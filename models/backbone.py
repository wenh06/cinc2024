"""
"""

import re
from typing import List, Optional, Sequence, Union

import timm
import torch
import torch.nn as nn
import torchvision as tv
import transformers
from torch_ecg.utils.misc import CitationMixin
from torch_ecg.utils.utils_nn import SizeMixin

__all__ = ["ImageBackbone"]


class ImageBackbone(nn.Module, SizeMixin, CitationMixin):
    """Backbone for extracting features from images.

    Parameters
    ----------
    backbone_name : str
        Name of the backbone model.
    source : {"timm", "hf", "tv"}, default "timm"
        Source of the backbone model:
        - "timm": models from `timm` package.
        - "hf": models from `transformers` package.
        - "tv": models from `torchvision` package.
    pretrained : bool, default True
        Whether to load pre-trained weights.

    """

    __timm_models__ = timm.list_models()
    __tv_models__ = tv.models.list_models()
    __DEBUG__ = True
    __name__ = "ImageBackbone"

    def __init__(self, backbone_name: str, source: str = "timm", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.source = source.lower()
        if self.source == "timm":
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        elif self.source == "tv":
            self.backbone = getattr(tv.models, backbone_name)(pretrained=pretrained)
        elif self.source == "hf":
            self.backbone = transformers.AutoModel.from_pretrained(backbone_name)
        else:
            raise ValueError(f"source: {source} not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @staticmethod
    def list_backbones(architectures: Optional[Union[str, Sequence[str]]] = None, source: Optional[str] = None) -> List[str]:
        """List available backbones.

        Parameters
        ----------
        architectures : str or sequence of str, optional
            Specific architectures to list (e.g. "resnet", "convnext", etc.). If None, list all available architectures.
        source : {"timm", "tv", "hf"}, optional
            Source of the backbone models. If None, "timm" is used.

        Returns
        -------
        list
            List of available architectures.

        """
        if architectures is None:
            architectures = []
        if source is None:
            source = "timm"
        source = source.lower()
        if source == "timm":
            model_names = ImageBackbone.__timm_models__
        elif source == "tv":
            model_names = ImageBackbone.__tv_models__

        elif source == "hf":
            print(
                "transformers does not provide a list of available models. Please refer to https://huggingface.co/models for available models."
            )
        else:
            raise ValueError(f"source: {source} not supported")

        if architectures:
            model_names = [
                name for name in model_names if any(list(re.search(arch, name) is not None for arch in architectures))
            ]

        return model_names

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze the backbone model.

        Parameters
        ----------
        freeze : bool, default True
            Whether to freeze the backbone model.

        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
