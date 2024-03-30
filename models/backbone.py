"""
"""

import os
import re
from typing import List, Optional, Sequence, Union

import numpy as np
import PIL
import timm
import torch
import torch.nn as nn
import torchvision as tv
import transformers
from torch_ecg.utils.misc import CitationMixin
from torch_ecg.utils.utils_nn import SizeMixin

from utils.misc import url_is_reachable

__all__ = ["ImageBackbone"]


_INPUT_IMAGE_TYPES = Union[
    torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray], PIL.Image.Image, List[PIL.Image.Image]
]


if not url_is_reachable("https://huggingface.co"):
    # workaround for using huggingface hub in China
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class ImageBackbone(nn.Module, SizeMixin, CitationMixin):
    """Backbone for extracting features from images.

    Parameters
    ----------
    backbone_name_or_path : str or `path_like`
        Name or path of the backbone model.
    source : {"timm", "hf", "tv"}, default "hf"
        Source of the backbone model:
        - "timm": models from `timm` package.
        - "hf": models from `transformers` package.
        - "tv": models from `torchvision` package.
    pretrained : bool, default True
        Whether to load pre-trained weights.

    .. note::

        ONLY `hf` is tested.

        Reasons for choosing huggingface transformers as the default source:

        - It has `AutoBackbone` class which produces feature maps from images,
          without the need to handle extra pooling layers and classifier layers.
          The rest two sources do not have a method for creating feature map extractors directly,
          and the models does not in general have a common method for extracting feature maps (e.g. calling methods like `forward_features`)
        - It has `AutoImageProcessor` class which can be used to preprocess images before feeding them to the model,
          so that one does not need to manually preprocess the images before feeding them to the model.
          The rest two sources do not have a method for creating image preprocessors directly (`timm` is better).
          One has to use for example `IMAGENET_DEFAULT_MEAN` and `IMAGENET_DEFAULT_STD` to normalize the images manually.

    """

    __timm_models__ = timm.list_models()
    __tv_models__ = tv.models.list_models()
    __DEBUG__ = True
    __name__ = "ImageBackbone"

    def __init__(self, backbone_name_or_path: Union[str, bytes, os.PathLike], source: str = "hf", pretrained: bool = True):
        super().__init__()
        self.backbone_name_or_path = backbone_name_or_path
        self.source = source.lower()
        if self.source == "hf":
            # preprocessor accepts batched images or single image,
            # in the format of numpy array or torch tensor or PIL image (single image)
            # or a list of single images (all 3 formats)
            self.preprocessor = transformers.AutoImageProcessor.from_pretrained(backbone_name_or_path)
            self.augmentor = None
            self.backbone = transformers.AutoBackbone.from_pretrained(backbone_name_or_path)
        elif self.source == "timm":
            self.backbone = timm.create_model(backbone_name_or_path, pretrained=pretrained)
            data_config = timm.data.resolve_model_data_config(self.backbone)
            self.preprocessor = timm.data.create_transform(**data_config, is_training=self.training)
            self.augmentor = None  # preprocessor is responsible for data augmentation
        elif self.source == "tv":
            self.preprocessor = None
            self.augmentor = None
            self.backbone = tv.models.get_model(backbone_name_or_path, pretrained=pretrained)
        else:
            raise ValueError(f"source: {source} not supported")

    def train(self, mode: bool = True) -> nn.Module:
        """Set the model and preprocessor to corresponding mode.

        Parameters
        ----------
        mode : bool, default True
            Whether to set the model and preprocessor to training mode.

        Returns
        -------
        nn.Module
            The model itself.

        """
        self.training = mode
        if self.source == "hf":
            self.backbone.train(mode)
        elif self.source == "timm":
            self.preprocessor.train(mode)
            self.backbone.train(mode)
        elif self.source == "tv":
            self.backbone.train(mode)
        super().train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Preprocessed input tensor.

        Returns
        -------
        torch.Tensor
            Output feature map tensor.

        """
        if self.training and self.augmentor is not None:
            x = self.augmentor(x)
        return self.backbone(x)

    def pipeline(self, x: _INPUT_IMAGE_TYPES) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.

        """
        assert self.preprocessor is not None, "Set up the preprocessor first."
        x_ndim = x.ndim
        if self.source == "hf":
            x = self.preprocessor(x)
        elif self.source == "timm":
            if x_ndim == 3:
                x = self.preprocessor(x)
            elif x_ndim == 4:
                x = torch.stack([self.preprocessor(img) for img in x])
            else:
                raise ValueError(f"Input tensor has invalid shape: {x.shape}")
        elif self.source == "tv":
            x = self.preprocessor(x)

        features = self.forward(x)

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
