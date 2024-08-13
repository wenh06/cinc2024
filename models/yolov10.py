import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import yaml
from torch_ecg.cfg import CFG
from torch_ecg.utils.utils_nn import CkptMixin, SizeMixin

from cfg import ModelCfg

from .ultralytics.engine.model import Model
from .ultralytics.models.yolov10.predict import YOLOv10DetectionPredictor
from .ultralytics.models.yolov10.train import YOLOv10DetectionTrainer
from .ultralytics.models.yolov10.val import YOLOv10DetectionValidator
from .ultralytics.nn.tasks import YOLOv10DetectionModel
from .ultralytics.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8SegmentationLoss, v10DetectLoss  # noqa: F401

_ULTRALYTICS_DIR = Path(__file__).parent / "ultralytics"
YOLOV10_CONFIGS = {
    scale: yaml.safe_load((_ULTRALYTICS_DIR / f"cfg/models/v10/yolov10{scale}.yaml").read_text()) for scale in list("nsmblx")
}


__all__ = ["YOLOv10_Detector"]


class YOLOv10_Detector(Model, SizeMixin, CkptMixin):

    __DEFAULT_CONFIGS__ = deepcopy(YOLOV10_CONFIGS)

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        self.__config = deepcopy(ModelCfg.object_detection)
        if config is not None:
            self.__config.update(deepcopy(config))
        verbose = kwargs.get("verbose", False)
        self.__config.update(kwargs)
        self.__yolo_cfg = self.__config.get("yolo_cfg", self.__DEFAULT_CONFIGS__)[self.config["scale"]]
        self.__yolo_cfg["nc"] = self.config["num_classes"]
        self.__yolo_cfg["scale"] = self.config["scale"]
        super().__init__(model=self.yolo_cfg, task="detect", verbose=verbose)
        setattr(self.model, "names", self.config["class_names"])
        setattr(self.model, "criterion", v10DetectLoss(self.model))

    @property
    def task_map(self) -> dict:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOv10DetectionModel,
                "trainer": YOLOv10DetectionTrainer,
                "validator": YOLOv10DetectionValidator,
                "predictor": YOLOv10DetectionPredictor,
            },
        }

    @property
    def config(self) -> CFG:
        return self.__config

    @property
    def yolo_cfg(self) -> dict:
        return self.__yolo_cfg

    # @property
    # def doi(self) -> str:
    #     return "10.48550/ARXIV.2405.14458"
