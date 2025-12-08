""" """

import math
import os
from functools import reduce
from typing import Any, Dict, Optional, Sequence

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch_ecg.cfg import CFG
from torch_ecg.models._nets import MLP
from torch_ecg.models.loss import AsymmetricLoss, BCEWithLogitsWithClassWeightLoss, FocalLoss
from torch_ecg.utils.utils_nn import CkptMixin, SizeMixin, compute_conv_output_shape

from .loss import get_loss_func

__all__ = ["ClassificationHead", "DigitizationHead"]


class ClassificationHead(nn.Module, SizeMixin, CkptMixin):
    """Head for making Dx classification (binary) predictions, including loss computation.

    The output features of the backbone model is fed into this head to make classification predictions.
    There are 10 Dx classes in CINC2024, and an extra class ("OTHER") for those that do not belong to any of the 10 classes.

    Typically, this module includes a few fully connected layers, followed by a softmax layer.
    The input image features are of shape ``(batch_size, num_features, height, width)``.
    A global average pooling layer is applied to the input features to get a feature vector of shape ``(batch_size, num_features)``.
    Then, a few fully connected layers are applied to the feature vector to get the logits for the two classes,
    which are of shape ``(batch_size, num_classes)``.
    The class number predictions are obtained by taking the argmax of the logits, which are of shape ``(batch_size,)``.

    Parameters
    ----------
    inp_features : int
        Number of input features.
    config : dict
        Hyper-parameters, including number of layers, hidden units, etc.
        ref. the corresponding config file.

    """

    def __init__(self, inp_features: int, config: CFG, **kwargs: Any) -> None:
        super().__init__()
        self.__inp_features = inp_features
        self.__config = config
        if self.__config.get("classification_head", None) is not None:
            self.classification_head = self.__config.classification_head
        # create the Dx classification head
        # self.classification_head = None  # TODO: implement the head
        self.classification_head = nn.Sequential()
        self.classification_head.add_module("global_pool", nn.AdaptiveAvgPool2d((1, 1)))
        self.classification_head.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        self.classification_head.add_module(
            "mlp",
            MLP(
                in_channels=self.inp_features,
                out_channels=self.config.out_channels + [self.config.num_classes],
                activation=self.config.activation,
                dropouts=self.config.dropouts,
            ),
        )
        # self.classification_criterion = nn.CrossEntropyLoss()
        loss_kw = self.config.get("criterion_kw", {})
        if self.config.criterion == "BCEWithLogitsLoss":
            self.classification_criterion = nn.BCEWithLogitsLoss(**loss_kw)
        elif self.config.criterion == "BCEWithLogitsWithClassWeightLoss":
            self.classification_criterion = BCEWithLogitsWithClassWeightLoss(**loss_kw)
        elif self.config.criterion == "BCELoss":
            self.classification_criterion = nn.BCELoss(**loss_kw)
        elif self.config.criterion == "FocalLoss":
            self.classification_criterion = FocalLoss(**loss_kw)
        elif self.config.criterion == "AsymmetricLoss":
            self.classification_criterion = AsymmetricLoss(**loss_kw)
        elif self.config.criterion == "CrossEntropyLoss":
            self.classification_criterion = nn.CrossEntropyLoss(**loss_kw)
        else:
            raise NotImplementedError(f"loss `{self.config.criterion}` not implemented!")

    def forward(self, img_features: torch.Tensor, labels: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        img_features : torch.Tensor
            Features extracted from the backbone model.
        labels : dict of torch.Tensor, optional
            Ground truth labels, including "dx" (required), etc.

        Returns
        -------
        Dict[str, torch.Tensor]
            Output dictionary containing the predictions and loss.

        """
        logits = self.classification_head(img_features)
        # preds = torch.argmax(logits, dim=1)
        probs = torch.sigmoid(logits)
        output = {"probs": probs, "logits": logits}
        if labels is not None and "dx" in labels:
            loss = self.classification_criterion(logits, labels["dx"].to(self.device))
            output["loss"] = loss
        return output

    @property
    def config(self) -> CFG:
        return self.__config

    @property
    def inp_features(self) -> int:
        return self.__inp_features


class DigitizationHead(nn.Module, SizeMixin, CkptMixin):
    """Head for making digitization predictions, including loss computation.

    Fundamentally, this head is a sequence generation head,
    which predicts the digitized values of the ECG signals from the image features.

    The digitization head typically consists of a few fully connected layers, followed by a softmax layer, like the Dx head.
    The difference is that to generate the digitized values,
    the image features (of shape ``(batch_size, num_features, height, width)``) are mapped to
    feature tensors of shape ``(batch_size, num_leads, max_len)``, where ``num_leads`` is the number of ECG leads,
    and ``max_len`` is the maximum length of the ECG signals.
    This is done by the following steps:

    1. The image features are flattened to get a feature vector
       of shape ``(batch_size, num_features * height * width)``.
    2. The feature vector is fed into a few fully connected (or convolutional) layers to get a feature tensor
       of shape ``(batch_size, num_leads * max_len)``.
    3. The feature tensor is reshaped to get the final feature tensor of shape ``(batch_size, num_leads, max_len)``.

    .. note::

        For running the model on the Challenge data, some information about the ECG signals is provided in the header files,
        including the sampling frequency and the length of the signal (number of samples).

    Parameters
    ----------
    config : dict
        Hyper-parameters, including number of layers, hidden units, etc.
        ref. the corresponding config file.

    """

    def __init__(self, inp_shape: Sequence[int], config: CFG, **kwargs: Any) -> None:
        super().__init__()
        self.__inp_shape = inp_shape
        self.__config = config
        if self.__config.get("digitization_head", None) is not None:
            self.digitization_head = self.__config.digitization_head
        self.digitization_head = nn.Sequential()
        # self.digitization_head.add_module("flatten", Rearrange("b c h w -> b (c h w)"))
        # in_channels = reduce(lambda x, y: x * y, self.inp_shape)
        # self.digitization_head.add_module(
        #     "mlp",
        #     MLP(
        #         in_channels=in_channels,
        #         out_channels=self.config.out_channels + [self.config.num_leads * self.config.max_len],
        #         activation=self.config.activation,
        #         dropouts=self.config.dropouts,
        #     ),
        # )
        # instead of using MLP, we use a convolutional layer which has much fewer parameters
        # namely, we are using "local" features to predict the digitized values
        # instead of the entire image feature set
        total_features = reduce(lambda x, y: x * y, self.inp_shape)
        self.digitization_head.add_module("transpose", Rearrange("b c h w -> b (h w) c"))
        # extra_len makes the output length of the next convolutional layer to be exactly max_len
        # so that we do not do any padding or cropping
        # NOTE that the stride is 1, so we can do this
        extra_len = (
            self.config.max_len
            - compute_conv_output_shape(
                input_shape=[None, None, self.config.max_len],
                kernel_size=self.config.kernel_size,
                dilation=self.config.dilation,
            )[-1]
        )
        self.digitization_head.add_module(
            "linear",
            nn.Linear(
                in_features=self.inp_shape[0],
                out_features=self.config.max_len + extra_len,
                bias=True,
            ),  # out shape: b (h w) c -> b (h w) (max_len + extra_len)
        )
        self.digitization_head.add_module(
            "conv1d",
            nn.Conv1d(
                in_channels=self.inp_shape[1] * self.inp_shape[2],
                out_channels=self.config.num_leads,
                kernel_size=self.config.kernel_size,
                stride=1,
                dilation=self.config.dilation,
                padding=0,
                bias=True,
            ),  # out shape: b (h w) (max_len + extra_len) -> b num_leads max_len
        )
        # self.digitization_head.add_module("reshape", Rearrange("b (n m) -> b n m", n=self.config.num_leads, m=self.config.max_len))
        self.digitization_criterion = get_loss_func(**self.config.criterion)

    def forward(self, img_features: torch.Tensor, labels: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        img_features : torch.Tensor
            Features extracted from the backbone model.
        labels : dict of torch.Tensor, optional
            Ground truth labels, including "digitization" (required), "mask" (optional), etc.

        Returns
        -------
        Dict[str, torch.Tensor]
            Output dictionary containing the predictions and loss.

        """
        preds = self.digitization_head(img_features)
        output = {"preds": preds}
        if labels is not None and "digitization" in labels:
            loss = self.digitization_criterion(preds, labels["digitization"], labels.get("mask", None))
            output["loss"] = loss
        return output

    def pipeline(self, img_features: torch.Tensor, fs: int, siglen: int) -> torch.Tensor:
        """Pipeline of the model.

        Parameters
        ----------
        img_features : torch.Tensor
            Features extracted from the backbone model.
        fs : int
            Sampling frequency of the ECG signal.
        siglen : int
            Length of the ECG signal.

        Returns
        -------
        torch.Tensor
            Predicted digitized values.

        """
        sig = self.digitization_head(img_features)
        # interpolate the signal to the required sampling frequency
        scale_factor = fs / self.config.fs
        sampto = math.ceil(siglen / scale_factor) + 1
        # print(f"{sampto=}")
        sig = F.interpolate(sig[..., :sampto], scale_factor=scale_factor, mode="linear")
        if sig.shape[-1] < siglen:
            sig = F.pad(sig, (0, siglen - sig.shape[-1]), "constant", 0)
        elif sig.shape[-1] > siglen:
            sig = sig[..., :siglen]
        return sig

    @property
    def config(self) -> CFG:
        return self.__config

    @property
    def inp_shape(self) -> int:
        return self.__inp_shape
