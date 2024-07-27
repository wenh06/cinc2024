"""Constants for the project."""

import os
from pathlib import Path
from typing import List, Union

import numpy as np
import PIL.Image
import torch

__all__ = [
    "INPUT_IMAGE_TYPES",
    "MODEL_CACHE_DIR",
    "DATA_CACHE_DIR",
    "REMOTE_HEADS_URLS",
]


INPUT_IMAGE_TYPES = Union[
    torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray], PIL.Image.Image, List[PIL.Image.Image]
]


MODEL_CACHE_DIR = str(
    Path(
        # "~/.cache/revenger_model_dir_cinc2024"
        os.environ.get("MODEL_CACHE_DIR", "~/.cache/cinc2024/revenger_model_dir")
    )
    .expanduser()
    .resolve()
)
Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)


DATA_CACHE_DIR = str(
    Path(
        # "~/.cache/revenger_data_dir_cinc2024"
        os.environ.get("DATA_CACHE_DIR", "~/.cache/cinc2024/revenger_data_dir")
    )
    .expanduser()
    .resolve()
)
Path(DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)


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
        "dropbox": (
            "https://www.dropbox.com/scl/fi/u62mdypjculkgrkm4skj2/"
            "hf-facebook-convnextv2-atto-1k-224-dx-headonly__15_04-05_07-06_epochloss_224.66350_metric_0.76.pth.tar"
            "?rlkey=iotmbhnnzwicf14b5cpjiymda&dl=1"
        ),
        "deep-psp": (
            "https://deep-psp.tech/Models/CinC2024/"
            "hf-facebook-convnextv2-atto-1k-224-dx-headonly__15_04-05_07-06_epochloss_224.66350_metric_0.76.pth.tar"
        ),
    },
    "hf--facebook/convnextv2-nano-22k-384": {
        "dropbox": (
            "https://www.dropbox.com/scl/fi/0z4gcs4acyd39mjzkcuj2/"
            "hf-facebook-convnextv2-nano-22k-384-dx-headonly__3_04-05_15-49_epochloss_296.05308_metric_0.79.pth.tar"
            "?rlkey=afir38rv2s4fghfb4l24jzjg4&dl=1"
        ),
        "deep-psp": (
            "https://deep-psp.tech/Models/CinC2024/"
            "hf-facebook-convnextv2-nano-22k-384-dx-headonly__3_04-05_15-49_epochloss_296.05308_metric_0.79.pth.tar"
        ),
    },
    "hf--microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft": {
        "dropbox": (
            "https://www.dropbox.com/scl/fi/876002r87cm756mvaweca/"
            "hf-microsoft-swinv2-base-patch4-window12to24-192to384-22kto1k-ft-dx-headonly__1_04-05_11-30_epochloss_395.73687_metric_0.75.pth.tar"
            "?rlkey=pvc4vekswnmuldj3syopck9r8&dl=1"
        ),
        "deep-psp": (
            "https://deep-psp.tech/Models/CinC2024/"
            "hf-microsoft-swinv2-base-patch4-window12to24-192to384-22kto1k-ft-dx-headonly__1_04-05_11-30_epochloss_395.73687_metric_0.75.pth.tar"
        ),
    },
}
