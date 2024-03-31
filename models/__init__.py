"""
CINC2024 models

It is a multi-head model for CINC2024. The backbone is a pre-trained image model, e.g., ResNet, DenseNet, etc.
"""

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import CitationMixin, add_docstring
from torch_ecg.utils.utils_nn import SizeMixin

from cfg import ModelCfg
from outputs import CINC2024Outputs
from utils.misc import url_is_reachable

from .backbone import _INPUT_IMAGE_TYPES, ImageBackbone
from .heads import DigitizationHead, DxHead
from .loss import get_loss_func

__all__ = [
    "MultiHead_CINC2024",
    "ImageBackbone",
    "DxHead",
    "DigitizationHead",
    "get_loss_func",
]


if os.environ.get("HF_ENDPOINT", None) is not None and (not url_is_reachable("https://huggingface.co")):
    # workaround for using huggingface hub in China
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class MultiHead_CINC2024(nn.Module, SizeMixin, CitationMixin):
    """Multi-head model for CINC2024.

    Parameters
    ----------
    config : dict
        Hyper-parameters, including backbone_name, etc.
        ref. the corresponding config file.

    """

    __DEBUG__ = True
    __name__ = "MultiHead_CINC2024"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        super().__init__()
        self.__config = deepcopy(ModelCfg)
        if config is not None:
            self.__config.update(deepcopy(config))
        self.image_backbone = ImageBackbone(
            self.__config.backbone_name,
            source=self.__config.backbone_source,
            pretrained=True,
        )
        backbone_output_shape = self.image_backbone.compute_output_shape()
        self.dx_head = DxHead(inp_features=backbone_output_shape[0], config=self.config.dx_head)
        self.digitization_head = DigitizationHead(inp_shape=backbone_output_shape, config=self.config.digitization_head)

    def forward(
        self,
        img: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        img : torch.Tensor
            Input image tensor.
        labels : dict, optional
            Labels for training, including
            - "dx": required for training the Dx classification head.
            - "digitization": required for training the digitization head.
            - "mask": optional for training the digitization head.

        Returns
        -------
        dict
            Predictions, including "dx" and "digitization",
            and the loss if any of the labels is provided.

        """
        features = self.image_backbone(img)
        dx_pred = self.dx_head(features, labels)
        digitization_pred = self.digitization_head(features, labels)
        total_loss = None
        if "loss" in dx_pred:
            total_loss = dx_pred["loss"]
        if "loss" in digitization_pred:
            if total_loss is None:
                total_loss = digitization_pred["loss"]
            else:
                total_loss += digitization_pred["loss"]
        return {
            "dx": dx_pred["preds"],
            "dx_logits": dx_pred["logits"],
            "dx_loss": dx_pred.get("loss", None),
            "digitization": digitization_pred["preds"],
            "digitization_loss": digitization_pred.get("loss", None),
            "total_loss": total_loss,
        }

    @add_docstring(ImageBackbone.get_input_tensors.__doc__)
    def get_input_tensors(self, x: _INPUT_IMAGE_TYPES) -> torch.Tensor:
        return self.image_backbone.get_input_tensors(x)

    @add_docstring(ImageBackbone.list_backbones.__doc__)
    @staticmethod
    def list_backbones(architectures: Optional[Union[str, Sequence[str]]] = None, source: Optional[str] = None) -> List[str]:
        return ImageBackbone.list_backbones(architectures=architectures, source=source)

    @torch.no_grad()
    def inference(self, img: _INPUT_IMAGE_TYPES) -> CINC2024Outputs:
        """Inference on a single image or a batch of images.

        Parameters
        ----------
        img : numpy.ndarray or torch.Tensor or PIL.Image.Image, or list
            Input image.

        Returns
        -------
        CINC2024Outputs
            Predictions, including "dx" and "digitization".

        """
        original_mode = self.training
        self.eval()
        output = self.forward(self.image_backbone.get_input_tensors(img))
        self.train(original_mode)
        return CINC2024Outputs(
            dx=output["dx"],
            dx_logits=output["dx_logits"],
            dx_classes=self.config.dx_head.classes,
            digitization=output["digitization"],
        )

    @add_docstring(inference.__doc__)
    def inference_CINC2024(self, img: _INPUT_IMAGE_TYPES) -> CINC2024Outputs:
        """
        alias for `self.inference`
        """
        return self.inference(img)

    @property
    def config(self) -> CFG:
        return self.__config
