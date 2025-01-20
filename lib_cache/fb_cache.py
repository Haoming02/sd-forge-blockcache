from functools import wraps
from typing import Callable

import torch
from ldm_patched.ldm.modules.diffusionmodules.openaimodel import apply_control
from ldm_patched.ldm.modules.diffusionmodules.util import timestep_embedding


def patch(BlockCache, forward: Callable):
    """First Block Cache"""

    @wraps(forward)
    @torch.inference_mode()
    def patched_forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        skip = False
        first_block = True
        assert (y is not None) == (self.num_classes is not None)

        if transformer_options["cond_or_uncond"] == [1, 0]:
            # both
            residual = BlockCache.previous_residual
            previous = BlockCache.previous
            distance = BlockCache.accumulated_distance
            BlockCache.this_step += 1
        elif transformer_options["cond_or_uncond"] == [0]:
            # cond
            residual = BlockCache.previous_residualP
            previous = BlockCache.previousP
            distance = BlockCache.accumulated_distanceP
            BlockCache.this_step += 0.5
        elif transformer_options["cond_or_uncond"] == [1]:
            # uncond
            residual = BlockCache.previous_residualN
            previous = BlockCache.previousN
            distance = BlockCache.accumulated_distanceN
            BlockCache.this_step += 0.5
        else:
            first_block = False

        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})
        block_modifiers = transformer_options.get("block_modifiers", [])

        hs = []
        t_emb = timestep_embedding(
            timesteps, self.model_channels, repeat_only=False
        ).to(x.dtype)
        emb = self.time_embed(t_emb)
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
        h = x

        original_h = h.clone()

        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            for block_modifier in block_modifiers:
                h = block_modifier(h, "before", transformer_options)
            h = module(h, emb, context, transformer_options)
            h = apply_control(h, control, "input")
            for block_modifier in block_modifiers:
                h = block_modifier(h, "after", transformer_options)
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)
            hs.append(h)
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)

            if first_block:
                first_block = False
                if (
                    previous is not None
                    and BlockCache.this_step > BlockCache.nocache_steps
                    and BlockCache.this_step + 0.5 < BlockCache.last_step
                ):
                    distance += (
                        ((h - previous).abs().mean() / previous.abs().mean())
                        .cpu()
                        .item()
                    )
                    previous = h.clone()
                    if distance < BlockCache.threshold:
                        h = original_h + residual
                        skip = True
                        break
                else:
                    previous = h.clone()

        if not skip:
            transformer_options["block"] = ("middle", 0)
            for block_modifier in block_modifiers:
                h = block_modifier(h, "before", transformer_options)
            h = self.middle_block(h, emb, context, transformer_options)
            h = apply_control(h, control, "middle")
            for block_modifier in block_modifiers:
                h = block_modifier(h, "after", transformer_options)
            for id, module in enumerate(self.output_blocks):
                transformer_options["block"] = ("output", id)
                hsp = hs.pop()
                hsp = apply_control(hsp, control, "output")
                if "output_block_patch" in transformer_patches:
                    patch = transformer_patches["output_block_patch"]
                    for p in patch:
                        h, hsp = p(h, hsp, transformer_options)
                h = torch.cat([h, hsp], dim=1)
                del hsp
                if len(hs) > 0:
                    output_shape = hs[-1].shape
                else:
                    output_shape = None
                for block_modifier in block_modifiers:
                    h = block_modifier(h, "before", transformer_options)
                h = module(h, emb, context, transformer_options, output_shape)
                for block_modifier in block_modifiers:
                    h = block_modifier(h, "after", transformer_options)
            transformer_options["block"] = ("last", 0)
            for block_modifier in block_modifiers:
                h = block_modifier(h, "before", transformer_options)
            if "group_norm_wrapper" in transformer_options:
                out_norm, out_rest = self.out[0], self.out[1:]
                h = transformer_options["group_norm_wrapper"](
                    out_norm, h, transformer_options
                )
                h = out_rest(h)
            else:
                h = self.out(h)
            for block_modifier in block_modifiers:
                h = block_modifier(h, "after", transformer_options)

            residual = h - original_h
            distance = 0

        if transformer_options["cond_or_uncond"] == [1, 0]:
            # both
            BlockCache.previous_residual = residual
            BlockCache.previous = previous
            BlockCache.accumulated_distance = distance
        elif transformer_options["cond_or_uncond"] == [0]:
            # cond
            BlockCache.previous_residualP = residual
            BlockCache.previousP = previous
            BlockCache.accumulated_distanceP = distance
        elif transformer_options["cond_or_uncond"] == [1]:
            # uncond
            BlockCache.previous_residualN = residual
            BlockCache.previousN = previous
            BlockCache.accumulated_distanceN = distance

        return h.type(x.dtype)

    return patched_forward
