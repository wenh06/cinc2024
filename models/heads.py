"""
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import CitationMixin
from torch_ecg.utils.utils_nn import SizeMixin


class DxHead(nn.Module, SizeMixin, CitationMixin):
    """Head for making Dx classification (binary) predictions, including loss computation.

    The output features of the backbone model is fed into this head to make predictions.
    There are two Dx classes in CINC2024, "Normal" and "Abnormal".
    Typically, this module includes a few fully connected layers, followed by a softmax layer.

    Parameters
    ----------
    config : dict
        Hyper-parameters, including number of layers, hidden units, etc.
        ref. the corresponding config file.

    """

    def __init__(self, config: CFG, **kwargs: Any) -> None:
        super().__init__()
        self.__config = config
        self.dx_head = None  # TODO: implement the head
        self.dx_criterion = nn.CrossEntropyLoss()

    def forward(self, img_features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        img_features : torch.Tensor
            Features extracted from the backbone model.
        labels : torch.Tensor, optional
            Ground truth labels.

        Returns
        -------
        Dict[str, torch.Tensor]
            Output dictionary containing the predictions and loss.

        """
        logits = self.dx_head(img_features)
        preds = torch.argmax(logits, dim=1)
        output = {"preds": preds, "logits": logits}
        if labels is not None:
            loss = self.dx_criterion(logits, labels["dx"])
            output["loss"] = loss
        return output

    @property
    def config(self) -> CFG:
        return self.__config


class DigitizationHead(nn.Module, SizeMixin, CitationMixin):
    """Head for making digitization predictions, including loss computation.

    Fundamentally, this head is a sequence generation head,
    which predicts the digitized values of the ECG signals from the image features.

    Parameters
    ----------
    config : dict
        Hyper-parameters, including number of layers, hidden units, etc.
        ref. the corresponding config file.

    """

    def __init__(self, config: CFG, **kwargs: Any) -> None:
        super().__init__()
        self.__config = config
        self.digitization_head = None  # TODO: implement the head
        self.digitization_criterion = None  # TODO: implement the criterion

    def forward(self, img_features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        img_features : torch.Tensor
            Features extracted from the backbone model.
        labels : torch.Tensor, optional
            Ground truth labels.

        Returns
        -------
        Dict[str, torch.Tensor]
            Output dictionary containing the predictions and loss.

        """
        preds = self.digitization_head(img_features)
        output = {"preds": preds}
        if labels is not None:
            loss = self.digitization_criterion(preds, labels["digitization"])
            output["loss"] = loss
        return output
