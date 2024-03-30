"""
CINC2024 models

It is a multi-head model for CINC2024. The backbone is a pre-trained image model, e.g., ResNet, DenseNet, etc.
"""

import os
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import CitationMixin, add_docstring
from torch_ecg.utils.utils_nn import SizeMixin

from cfg import ModelCfg
from outputs import CINC2024Outputs
from utils.misc import url_is_reachable

from .backbone import ImageBackbone
from .heads import DigitizationHead, DxHead

__all__ = [
    "MultiHead_CINC2024",
]


if not url_is_reachable("https://huggingface.co"):
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
        if config is None:
            self.__config = deepcopy(ModelCfg)
        else:
            self.__config = deepcopy(config)
        self.image_backbone = ImageBackbone(
            self.__config.backbone_name,
            source=self.__config.backbone_source,
            pretrained=True,
        )
        self.dx_head = DxHead()
        self.digitization_head = DigitizationHead()

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
            Labels for training, including "dx" and "digitization".

        Returns
        -------
        dict
            Predictions, including "dx" and "digitization".

        """
        features = self.image_backbone(img)
        dx_pred = self.dx_head(features)
        digitization_pred = self.digitization_head(features)
        return {
            "dx": dx_pred,
            "digitization": digitization_pred,
        }

    @torch.no_grad()
    def inference(self, img: Union[np.ndarray, torch.Tensor]) -> CINC2024Outputs:
        raise NotImplementedError

    @add_docstring(inference.__doc__)
    def inference_CINC2024(self, img: Union[np.ndarray, torch.Tensor]) -> CINC2024Outputs:
        """
        alias for `self.inference`
        """
        return self.inference(img)

    @property
    def config(self) -> CFG:
        return self.__config
