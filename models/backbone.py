"""
"""

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision as tv
import transformers
from PIL import Image
from torch_ecg.utils.download import url_is_reachable
from torch_ecg.utils.utils_nn import CkptMixin, SizeMixin

from const import INPUT_IMAGE_TYPES, MODEL_CACHE_DIR

# workaround for using huggingface hub in China
if os.environ.get("HF_ENDPOINT", None) is not None and (not url_is_reachable(os.environ["HF_ENDPOINT"])):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
elif os.environ.get("HF_ENDPOINT", None) is None and (not url_is_reachable("https://huggingface.co")):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HOME"] = str(Path(MODEL_CACHE_DIR).parent)

__all__ = ["ImageBackbone"]


class ImageBackbone(nn.Module, SizeMixin, CkptMixin):
    """Backbone for extracting features from images.

    Parameters
    ----------
    backbone_name_or_path : str or `path_like`
        Name or path of the backbone model.
    source : {"timm", "hf", "tv"}, default "hf"
        Source of the backbone model:
        - "hf": models from `transformers` package.
        - "timm": models from `timm` package.
        - "tv": models from `torchvision` package.
    pretrained : bool, default True
        Whether to load pre-trained weights.

    .. note::

        ONLY `hf` is tested.

        Reasons for choosing huggingface transformers as the default source:

        - It has `AutoBackbone` class which produces feature maps from images,
          without the need to handle extra pooling layers and classifier layers.
          The rest two sources do not have a method for creating feature map extractors directly,
          and the models does not in general have a common method for extracting feature maps (e.g. calling methods like `forward_features`).
        - It has `AutoImageProcessor` class which can be used to preprocess images before feeding them to the model,
          so that one does not need to manually preprocess the images before feeding them to the model.
          The rest two sources do not have a method for creating image preprocessors directly (`timm` is better).
          One has to use for example `IMAGENET_DEFAULT_MEAN` and `IMAGENET_DEFAULT_STD` to normalize the images manually.

    """

    __timm_models__ = timm.list_models()
    __tv_models__ = tv.models.list_models()
    __DEBUG__ = True
    __name__ = "ImageBackbone"

    def __init__(
        self,
        backbone_name_or_path: Union[str, bytes, os.PathLike],
        source: Literal["timm", "hf", "tv"] = "hf",
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name_or_path = backbone_name_or_path
        self.source = source.lower()
        if self.source == "hf":
            # preprocessor accepts batched images or single image,
            # in the format of numpy array or torch tensor or PIL image (single image)
            # or a list of single images (all 3 formats)
            self.preprocessor = transformers.AutoImageProcessor.from_pretrained(
                backbone_name_or_path,
                cache_dir=MODEL_CACHE_DIR,
            )
            self.augmentor = None
            self.backbone = transformers.AutoBackbone.from_pretrained(
                backbone_name_or_path,
                cache_dir=MODEL_CACHE_DIR,
            )
        elif self.source == "timm":
            warnings.warn("backbone source 'timm' is not fully tested. Use it with caution.")
            self.backbone = timm.create_model(backbone_name_or_path, pretrained=pretrained)
            data_config = timm.data.resolve_model_data_config(self.backbone)
            self.preprocessor = timm.data.create_transform(**data_config, is_training=self.training)
            self.augmentor = None  # preprocessor is responsible for data augmentation
        elif self.source == "tv":
            warnings.warn("backbone source 'tv' is not fully tested. Use it with caution.")
            self.preprocessor = None
            self.augmentor = None
            self.backbone = tv.models.get_model(backbone_name_or_path, pretrained=pretrained)
        else:
            raise ValueError(f"source: {source} not supported")
        self.__default_output_shape = None

        # resolve issues related to backbone loading
        self.__post_init()

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
        if self.source == "hf":
            return self.backbone(x).feature_maps[-1]
        return self.backbone(x)

    def pipeline(self, x: INPUT_IMAGE_TYPES) -> torch.Tensor:
        """Pipeline of the backbone.

        This method accepts various types of input images
        and returns the output feature map tensor (with batch dimension).

        Parameters
        ----------
        x : numpy.ndarray or torch.Tensor or PIL.Image.Image or list
            Input image(s).

        Returns
        -------
        torch.Tensor
            Output feature map tensor.

        """
        x = self.get_input_tensors(x)
        return self.forward(x)

    def get_input_tensors(self, x: INPUT_IMAGE_TYPES) -> torch.Tensor:
        """Get input tensors for the model.

        Parameters
        ----------
        x : numpy.ndarray or torch.Tensor or PIL.Image.Image or list
            Input image(s).

        Returns
        -------
        torch.Tensor
            Input tensor for the image backbone model.

        """
        assert self.preprocessor is not None, "Set up the preprocessor first."
        if isinstance(x, (np.ndarray, torch.Tensor)):
            x_ndim = x.ndim
        elif isinstance(x, (Image.Image)):
            x_ndim = 3
        elif isinstance(x, (list, tuple)):
            x_ndim = 4
        else:
            raise ValueError(f"Input tensor has invalid type: {type(x)}")
        if self.source == "hf":
            x = self.preprocessor(x).convert_to_tensors("pt")["pixel_values"].to(self.device)
        elif self.source == "timm":
            if x_ndim == 3:
                x = self.preprocessor(x).to(self.device)
            elif x_ndim == 4:
                x = torch.stack([self.preprocessor(img).to(self.device) for img in x])
            else:
                raise ValueError(f"Input tensor has invalid shape: {x.shape}")
        elif self.source == "tv":
            x = self.preprocessor(x).to(self.device)
        return x

    @staticmethod
    def list_backbones(
        architectures: Optional[Union[str, Sequence[str]]] = None, source: Optional[Literal["timm", "hf", "tv"]] = None
    ) -> List[str]:
        """List available backbones.

        Parameters
        ----------
        architectures : str or sequence of str, optional
            Specific architectures to list (e.g. "resnet", "convnext", etc.). If None, list all available architectures.
        source : {"hf", "timm", "tv"}, optional
            Source of the backbone models. If None, "hf" is used.

        Returns
        -------
        list
            List of available architectures.

        """
        if architectures is None:
            architectures = []
        if source is None:
            source = "hf"
        source = source.lower()
        if source == "timm":
            model_names = ImageBackbone.__timm_models__
        elif source == "tv":
            model_names = ImageBackbone.__tv_models__
        elif source == "hf":
            print(
                "transformers does not provide a list of available models. Please refer to https://huggingface.co/models for available models."
            )
            return []
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

        Returns
        -------
        None

        Examples
        --------
        >>> bkb = ImageBackbone("microsoft/resnet-18")
        >>> bkb.freeze_backbone()
        >>> bkb.module_size_
        '0.0B'
        >>> bkb.sizeof_
        '42.7MB'
        >>> bkb.freeze_backbone(False)
        >>> bkb.module_size_
        '42.6MB'
        >>> bkb.sizeof_
        '42.7MB'

        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    @property
    def num_features(self) -> int:
        """Number of output channels of the backbone model.

        Returns
        -------
        int
            Number of output channels.

        """
        if self.source == "hf":
            return self.backbone.config.hidden_sizes[-1]
        elif self.source == "timm":
            return self.backbone.num_features
        else:
            return self.backbone.fc.in_features

    def compute_output_shape(self, input_shape: Optional[Union[Sequence[int], int]] = None) -> List[int]:
        """Compute the output shape of the backbone model, not including the batch dimension.

        Parameters
        ----------
        input_shape : int or sequence of int, optional
            Input shape of the image.

        Returns
        -------
        list
            Output shape of the backbone model.

        """
        if input_shape is None:
            if self.__default_output_shape is not None:
                return self.__default_output_shape
            _input_shape = [3, 224, 224]
        else:
            if isinstance(input_shape, int):
                _input_shape = [3, input_shape, input_shape]
            else:
                _input_shape = input_shape
        test_input = torch.randint(0, 256, (1, *_input_shape), dtype=torch.uint8)
        with torch.no_grad():
            output = self.pipeline(test_input)
        del test_input
        if input_shape is None:
            self.__default_output_shape = list(output.shape[1:])
        return list(output.shape[1:])

    def __post_init(self) -> None:
        """Post initialization process.

        Resolves known issues related to backbone loading.

        Returns
        -------
        None

        Known Issues
        ------------
        - `facebook/convnextv2` models from huggingface transformers package.
          The last layer (layer norm) has keys "hidden_states_norms.stage4.weight" and "hidden_states_norms.stage4.bias",
          but the model weights downloaded from huggingface model hub have keys "convnextv2.layernorm.weight" and
          "convnextv2.layernorm.bias".

        """
        if self.source == "hf":
            if re.search("facebook(\\/|\\-\\-)convnextv2", self.backbone_name_or_path):
                if Path(self.backbone_name_or_path).exists():
                    weight_file = list(Path(self.backbone_name_or_path).rglob("pytorch_model.bin"))[0]
                else:
                    weight_file = list(
                        (Path(MODEL_CACHE_DIR) / Path(f"""models--{self.backbone_name_or_path.replace("/", "--")}"""))
                        .expanduser()
                        .resolve()
                        .rglob("pytorch_model.bin")
                    )[0]
                state_dict = torch.load(weight_file)
                new_state_dict = {
                    "stage4.weight": state_dict["convnextv2.layernorm.weight"].detach().clone(),
                    "stage4.bias": state_dict["convnextv2.layernorm.bias"].detach().clone(),
                }
                self.backbone.hidden_states_norms.load_state_dict(new_state_dict)
                print(
                    "Loaded layer norm weights from the model weights for the last hidden_states_norms layer from "
                    f"weights file: {str(weight_file)}"
                )
                # remove `state_dict` to avoid potential memory leak
                del state_dict

    def set_input_size(self, input_size: Dict[str, int]) -> None:
        """Set the input size of the backbone model.

        Works for `hf` source only. `self.preprocessor.size` is updated using this method.

        Parameters
        ----------
        input_size : dict
            Input size of the backbone model.
            Example: {"height": 512, "width": 1024, "shortest_edge": 768}.
            It depends on the processor used by the backbone model to
            use "width" and "height" or "shortest_edge" to resize the input image.

        Returns
        -------
        None

        """
        if "shortest_edge" in self.preprocessor.size:
            self.preprocessor.size["shortest_edge"] = input_size["shortest_edge"]
        elif "height" in self.preprocessor.size and "width" in self.preprocessor.size:
            self.preprocessor.size["height"] = input_size["height"]
            self.preprocessor.size["width"] = input_size["width"]
        else:  # not found in the processor (typically won't happen for huggingface transformers)
            raise ValueError(f"Input size not found in the processor: {self.preprocessor.size}")

        # reset the default output shape
        self.__default_output_shape = None
        self.compute_output_shape()
