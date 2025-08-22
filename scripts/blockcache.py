from typing import Callable

import gradio as gr
from lib_cache import parse_steps, StableDiffusionProcessingTxt2Img
from lib_cache.fb_cache import patch as fb_patch
from lib_cache.settings import settings
from lib_cache.tea_cache import patch as t_patch

from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel
from modules import scripts
from modules.script_callbacks import on_ui_settings
from modules.shared import opts

VERSION = "0.3.0"


class BlockCache(scripts.Script):
    index: int
    this_step: int
    last_step: int
    nocache_steps: int
    threshold: float
    distance: list[float]
    residual: list[float]
    previous: list[float]
    previousSigma: float
    skipped: list[int]
    skip_limit: int
    ignore_last: bool

    logged: bool = False

    def title(self):
        return "Block Cache"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        always: str = getattr(opts, "bc_always", None)
        if always:
            import json

            try:
                args: list = json.loads(always)

                return [
                    gr.State(True),
                    gr.State(str(args[0])),
                    gr.State(float(args[1])),
                    gr.State(float(args[2])),
                    gr.State(bool(args[3])),
                    gr.State(int(args[4])),
                ]

            except ValueError:
                if not is_img2img:
                    print("\n[BlockCache] Failed to parse Always Enabled Settings...\n")
            except json.JSONDecodeError:
                if not is_img2img:
                    print("\n[BlockCache] Failed to parse JSON...\n")

        with gr.Accordion(label=f"{self.title()} v{VERSION}", open=False):
            with gr.Row():
                with gr.Column():
                    enable = gr.Checkbox(False, label="Enable")
                    ignore_last = gr.Checkbox(False, label="Do not Cache on last step")
                method = gr.Radio(
                    label="Method",
                    choices=("First Block Cache", "Tea Cache"),
                    value="First Block Cache",
                )
            with gr.Row():
                nocache_ratio = gr.Slider(
                    label="Cache Start",
                    info="caching activation step; lower=faster",
                    minimum=0.0,
                    maximum=1.0,
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
                max_cached = gr.Slider(
                    label="Cache Limit",
                    info="max consecutive cache; 0=process all",
                    minimum=0,
                    maximum=4,
                    value=0,
                    step=1,
                )

        self.paste_field_names = []
        self.infotext_fields = [
            (enable, "bc_enable"),
            (method, "bc_method"),
            (nocache_ratio, "bc_ratio"),
            (threshold, "bc_threshold"),
            (ignore_last, "bc_ignore_last"),
            (max_cached, "bc_max_cached"),
        ]

        for comp, name in self.infotext_fields:
            comp.do_not_save_to_config = True
            self.paste_field_names.append(name)

        return [enable, method, nocache_ratio, threshold, ignore_last, max_cached]

    def process(
        self,
        p,
        enable: bool,
        method: str,
        nocache_ratio: float,
        threshold: float,
        ignore_last: bool,
        max_cached: int,
        *args,
        **kwargs,
    ):
        if not enable:
            return

        if not p.sd_model.is_sdxl:
            if not BlockCache.logged:
                BlockCache.logged = True
                print("\n[BlockCache] Only SDXL is supported...\n")
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
                "bc_ignore_last": ignore_last,
                "bc_max_cached": max_cached,
            }
        )

    def process_before_every_sampling(
        self,
        p: "StableDiffusionProcessingTxt2Img",
        enable: bool,
        method: str,
        nocache_ratio: float,
        threshold: float,
        ignore_last: bool,
        max_cached: int,
        *args,
        **kwargs,
    ):
        if not enable or getattr(self, "original_forward", None) is None:
            return

        if getattr(p, "is_hr_pass", False) and not getattr(opts, "bc_hires", False):
            if hasattr(self, "original_forward"):
                UNetModel.forward = self.original_forward
                delattr(self, "original_forward")
            return

        total_steps = parse_steps(p)

        setattr(BlockCache, "index", 0)
        setattr(BlockCache, "this_step", 0)
        setattr(BlockCache, "last_step", total_steps)
        setattr(BlockCache, "nocache_steps", int(total_steps * nocache_ratio))
        setattr(BlockCache, "threshold", threshold)
        setattr(BlockCache, "distance", [0])
        setattr(BlockCache, "residual", [None])
        setattr(BlockCache, "previous", [None])
        setattr(BlockCache, "previousSigma", None)
        setattr(BlockCache, "skipped", [0])
        setattr(BlockCache, "skip_limit", max_cached)
        setattr(BlockCache, "ignore_last", ignore_last)

    def postprocess(self, p, processed, enable: bool, *args, **kwargs):
        if not enable or getattr(self, "original_forward", None) is None:
            return

        UNetModel.forward = self.original_forward
        delattr(self, "original_forward")


on_ui_settings(settings)
