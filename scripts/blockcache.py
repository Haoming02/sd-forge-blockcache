from typing import Callable

import gradio as gr
from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel
from modules import scripts

from lib_cache import parse_steps
from lib_cache.fb_cache import patch as fb_patch
from lib_cache.tea_cache import patch as t_patch


class BlockCache(scripts.Script):

    def title(self):
        return "Block Cache"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(label=self.title(), open=False):
            with gr.Row():
                enable = gr.Checkbox(False, label="Enable")
                method = gr.Radio(
                    label="Method",
                    choices=("First Block Cache", "Tea Cache"),
                    value="First Block Cache",
                )
            with gr.Row():
                nocache_ratio = gr.Slider(
                    label="Cache Start",
                    info="caching activation step; lower=faster",
                    minimum=0.1,
                    maximum=0.9,
                    value=0.6,
                    step=0.05,
                )
                threshold = gr.Slider(
                    label="Cache Threshold",
                    info="caching aggressiveness; higher=faster",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.4,
                    step=0.05,
                )

        self.paste_field_names = []
        self.infotext_fields = [
            (enable, "bc_enable"),
            (method, "bc_method"),
            (nocache_ratio, "bc_ratio"),
            (threshold, "bc_threshold"),
        ]

        for comp, name in self.infotext_fields:
            comp.do_not_save_to_config = True
            self.paste_field_names.append(name)

        return [enable, method, nocache_ratio, threshold]

    def process(
        self,
        p,
        enable: bool,
        method: str,
        nocache_ratio: float,
        threshold: float,
        *args,
        **kwargs,
    ):
        if not enable:
            return

        self.original_forward: Callable = UNetModel.forward

        match method:
            case "First Block Cache":
                UNetModel.forward = fb_patch(BlockCache, UNetModel.forward)
            case "Tea Cache":
                UNetModel.forward = t_patch(BlockCache, UNetModel.forward)
            case _:
                raise ValueError

        p.extra_generation_params.update(
            {
                "bc_enable": enable,
                "bc_method": method,
                "bc_ratio": nocache_ratio,
                "bc_threshold": threshold,
            }
        )

    def process_before_every_sampling(
        self,
        p,
        enable: bool,
        method: str,
        nocache_ratio: float,
        threshold: float,
        *args,
        **kwargs,
    ):
        if not enable:
            return

        total_steps = parse_steps(p)

        setattr(BlockCache, "this_step", 0)
        setattr(BlockCache, "last_step", total_steps)
        setattr(BlockCache, "nocache_steps", int(total_steps * nocache_ratio))
        setattr(BlockCache, "threshold", threshold)
        setattr(BlockCache, "accumulated_distance", 0)
        setattr(BlockCache, "accumulated_distanceP", 0)
        setattr(BlockCache, "accumulated_distanceN", 0)
        setattr(BlockCache, "previous_residual", None)
        setattr(BlockCache, "previous_residualP", None)
        setattr(BlockCache, "previous_residualN", None)
        setattr(BlockCache, "previous", None)
        setattr(BlockCache, "previousP", None)
        setattr(BlockCache, "previousN", None)

    def postprocess(self, p, processed, enable: bool, *args, **kwargs):
        if not enable:
            return

        UNetModel.forward = self.original_forward
