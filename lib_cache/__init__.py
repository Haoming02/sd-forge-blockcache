from modules.sd_samplers_common import setup_img2img_steps
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
    StableDiffusionProcessingTxt2Img,
)


def parse_steps(p: StableDiffusionProcessing) -> int:
    if isinstance(p, StableDiffusionProcessingImg2Img):
        _, total_steps = setup_img2img_steps(p)

    else:
        assert isinstance(p, StableDiffusionProcessingTxt2Img)
        total_steps = p.steps
        if getattr(p, "is_hr_pass", False):
            if (hr := getattr(p, "hr_second_pass_steps", 0)) > 0:
                total_steps = hr

    return total_steps
