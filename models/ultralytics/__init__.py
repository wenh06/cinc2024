# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))  # make `ultralytics` visible in this submodule

__version__ = "8.1.34"

from ultralytics.data.explorer.explorer import Explorer
from ultralytics.models import RTDETR, SAM, YOLO, YOLOWorld, YOLOv10
from ultralytics.models.fastsam import FastSAM
from ultralytics.models.nas import NAS
from ultralytics.utils import ASSETS, SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
    "YOLOv10"
)