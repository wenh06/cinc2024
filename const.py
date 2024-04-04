"""Constants for the project."""

from typing import List, Union

import numpy as np
import PIL
import torch

__all__ = [
    "INPUT_IMAGE_TYPES",
    "REMOTE_HEADS_URLS",
]


INPUT_IMAGE_TYPES = Union[
    torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray], PIL.Image.Image, List[PIL.Image.Image]
]


REMOTE_HEADS_URLS = {
    "hf--facebook/convnextv2-large-22k-384": {
        "dropbox": (
            "https://www.dropbox.com/scl/fi/c38dgecawfy7rhg1gjjgq/"
            "hf-facebook-convnextv2-large-22k-384-dx-headonly4_04-04_07-32_epochloss_202.66414_metric_0.78.pth.tar"
            "?rlkey=7za4y2o7ayarjuyyi4dawjrcq&dl=1"
        ),
        "deep-psp": (
            "https://deep-psp.tech/Models/CinC2024/"
            "hf-facebook-convnextv2-large-22k-384-dx-headonly4_04-04_07-32_epochloss_202.66414_metric_0.78.pth.tar"
        ),
    },
    "hf--facebook/convnextv2-atto-1k-224": {
        "dropbox": None,
        "deep-psp": None,
    },
    "hf--facebook/convnextv2-nano-22k-384": {
        "dropbox": None,
        "deep-psp": None,
    },
    "hf--microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft": {
        "dropbox": None,
        "deep-psp": None,
    },
}
