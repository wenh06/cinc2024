"""Constants for the project."""

import os
from pathlib import Path
from typing import List, Union

import numpy as np
import PIL.Image
import torch

__all__ = [
    "INPUT_IMAGE_TYPES",
    "PROJECT_DIR",
    "MODEL_CACHE_DIR",
    "DATA_CACHE_DIR",
    "TEST_DATA_CACHE_DIR",
    "SUBSET_DATA_CACHE_DIR",
    "FULL_DATA_CACHE_DIR",
    "REMOTE_MODELS",
]


INPUT_IMAGE_TYPES = Union[
    torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray], PIL.Image.Image, List[PIL.Image.Image]
]


PROJECT_DIR = str(Path(__file__).resolve().parent)


MODEL_CACHE_DIR = str(
    Path(
        # ~/.cache/revenger_model_dir_cinc2024
        # /challenge/cache/revenger_model_dir
        os.environ.get("MODEL_CACHE_DIR", "~/.cache/cinc2024/revenger_model_dir")
    )
    .expanduser()
    .resolve()
)
Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)


DATA_CACHE_DIR = str(
    Path(
        # ~/.cache/revenger_data_dir_cinc2024
        # /challenge/cache/revenger_data_dir
        os.environ.get("DATA_CACHE_DIR", "~/.cache/cinc2024/revenger_data_dir")
    )
    .expanduser()
    .resolve()
)
Path(DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
TEST_DATA_CACHE_DIR = str(Path(DATA_CACHE_DIR) / "cinc2024_action_test_data")
SUBSET_DATA_CACHE_DIR = str(Path(DATA_CACHE_DIR) / "cinc2024_subset_data")
FULL_DATA_CACHE_DIR = str(Path(DATA_CACHE_DIR) / "cinc2024_full_data")


REMOTE_MODELS = {
    "hf--facebook/convnextv2-nano-22k-384": {
        "url": {
            "google-drive": ("https://drive.google.com/u/0/uc?id=1KKioeOkUYHNXPGRhnXHWz9nmjudVjNEq"),
            "deep-psp": (
                "https://deep-psp.tech/Models/CinC2024/"
                "BestModel_hf-facebook-convnextv2-nano-22k-384-dx_20_08-09_13-18_metric_0.32.pth.tar"
            ),
        },
        "threshold": 0.9,
        "filename": "BestModel_hf-facebook-convnextv2-nano-22k-384-dx_20_08-09_13-18_metric_0.32.pth.tar",
    },
    "hf--facebook/detr-resnet-50": {
        "url": {
            "google-drive": ("https://drive.google.com/u/0/uc?id=19bt05pNy7u-6uKz2WPyP3OYdTS9f3mrB"),
            "deep-psp": (
                "https://deep-psp.tech/Models/CinC2024/"
                "BestModel_facebook-detr-resnet-50-merge_horizontal_3_08-09_07-18_metric_0.90.pth.tar"
            ),
        },
        "filename": "BestModel_facebook-detr-resnet-50-merge_horizontal_3_08-09_07-18_metric_0.90.pth.tar",
    },
    "custom--unet": {
        "url": {
            "google-drive": ("https://drive.google.com/u/0/uc?id=1vrfUtSGUr5YHghwfFpxLxYFDp_4M--R1"),
            "deep-psp": (
                "https://deep-psp.tech/Models/CinC2024/BestModel_custom-unet-roi_only_5_08-14_05-48_metric_0.73.pth.tar"
            ),
        },
        "filename": "BestModel_custom-unet-roi_only_5_08-14_05-48_metric_0.73.pth.tar",
    },
}
