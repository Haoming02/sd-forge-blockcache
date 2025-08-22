from functools import wraps
from typing import TYPE_CHECKING, Callable

import torch

from ldm_patched.ldm.modules.diffusionmodules.openaimodel import apply_control
from ldm_patched.ldm.modules.diffusionmodules.util import timestep_embedding

if TYPE_CHECKING:
    from scripts.blockcache import BlockCache


def patch(cls: "BlockCache", forward: Callable):
    """Tea Cache"""

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

        if x.size(1) != 4:
            return forward(self, x, timesteps, context, y, control, transformer_options, **kwargs)

        skip = False
        assert (y is not None) == (self.num_classes is not None)

        thisSigma = transformer_options["sigmas"][0].item()
        if cls.previousSigma == thisSigma:
            cls.index += 1
            if cls.index == len(cls.distance):
                cls.distance.append(0)
                cls.residual.append(None)
                cls.previous.append(None)
                cls.skipped.append(0)
        else:
            cls.previousSigma = thisSigma
            cls.index = 0
            cls.this_step += 1

        index = cls.index
        residual = cls.residual[index]
        previous = cls.previous[index]
        distance = cls.distance[index]
        skipped = cls.skipped[index]

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

        if cls.this_step <= cls.nocache_steps:
            skip_check = False
        elif cls.ignore_last and cls.this_step == cls.last_step:
            skip_check = False
        else:
            skip_check = True

        if previous is None or previous.shape != original_h.shape:
            skip_check = False
        if residual is None:
            skip_check = False
        if cls.skip_limit > 0 and skipped >= cls.skip_limit:
            skip_check = False

        if skip_check:
            distance += (
                (
                    (original_h - cls.previous[index]).abs().mean()
                    / cls.previous[index].abs().mean()
                )
                .cpu()
                .item()
            )

            if distance < cls.threshold:
                skip = True

        if skip:
            h += residual
            skipped += 1
        else:
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
            skipped = 0

        cls.residual[index] = residual
        cls.previous[index] = original_h
        cls.distance[index] = distance
        cls.skipped[index] = skipped

        return h.type(x.dtype)

    return patched_forward
