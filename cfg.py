"""
Configurations for models, training, etc., as well as some constants.
"""

import pathlib
from copy import deepcopy

import numpy as np
import torch
from torch_ecg.cfg import CFG
from torch_ecg.model_configs import linear

__all__ = [
    "BaseCfg",
    "TrainCfg",
    "ModelCfg",
]


_BASE_DIR = pathlib.Path(__file__).absolute().parent


###############################################################################
# Base Configs,
# including path, data type, classes, etc.
###############################################################################

BaseCfg = CFG()
BaseCfg.db_dir = None
BaseCfg.working_dir = None
BaseCfg.project_dir = _BASE_DIR
BaseCfg.log_dir = _BASE_DIR / "log"
BaseCfg.model_dir = _BASE_DIR / "saved_models"
BaseCfg.log_dir.mkdir(exist_ok=True)
BaseCfg.model_dir.mkdir(exist_ok=True)

BaseCfg.torch_dtype = torch.float32  # "double"
BaseCfg.np_dtype = np.float32
BaseCfg.num_leads = 12
BaseCfg.normal_class = "Normal"
BaseCfg.abnormal_class = "Abnormal"
BaseCfg.classes = [BaseCfg.normal_class, BaseCfg.abnormal_class]


###############################################################################
# training configurations for machine learning and deep learning
###############################################################################

TrainCfg = deepcopy(BaseCfg)


###############################################################################
# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
###############################################################################


_BASE_MODEL_CONFIG = CFG()
_BASE_MODEL_CONFIG.num_leads = BaseCfg.num_leads
_BASE_MODEL_CONFIG.torch_dtype = BaseCfg.torch_dtype


ModelCfg = deepcopy(_BASE_MODEL_CONFIG)

# a list of candidate backbones
# microsoft/resnet-18  (46.8MB in memory consumption, including the classification head, pretrained on ImageNet-1k)
# facebook/convnextv2-atto-1k-224  (14.9 MB)
# facebook/convnextv2-femto-1k-224  (21.0 MB)
# facebook/convnextv2-pico-1k-224  (36.3 MB)
# facebook/convnextv2-nano-22k-384  (62.5 MB)
# facebook/convnextv2-tiny-22k-384  (115 MB)
# facebook/convnextv2-base-22k-384  (355 MB)
# facebook/convnextv2-large-22k-384  (792 MB)
# facebook/convnextv2-huge-22k-512  (2.64 GB)
# microsoft/swinv2-tiny-patch4-window16-256  (113 MB, pretrained on ImageNet-1k)
# microsoft/swinv2-small-patch4-window16-256  (199 MB, pretrained on ImageNet-1k)
# microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft  (352 MB)
# microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft  (787MB)
ModelCfg.backbone_name = "microsoft/resnet-18"
ModelCfg.backbone_source = "hf"

ModelCfg.dx_head = deepcopy(linear)

ModelCfg.dx_head.out_channels = [
    # containing just the intermediate features
    # not including the input features and the output features
    # 1024,
    256,
]
ModelCfg.dx_head.dropouts = 0.3
ModelCfg.dx_head.activation = "mish"

ModelCfg.dx_head.num_classes = len(BaseCfg.classes)
ModelCfg.dx_head.criterion = "CrossEntropyLoss"
ModelCfg.dx_head.label_smoothing = 0.1

# ModelCfg.digitization_head = deepcopy(linear)
# ModelCfg.digitization_head.out_channels = [
#     # no intermediate features
#     # the (flattened) input features, whose number equals
#     # (backbone output channels) * (backbone output height) * (backbone output width),
#     # are fed into ONE fully connected layer,
#     # and further reshaped to the final output shape
# ]

# digitization_head now use 1D convolutional layer instead
ModelCfg.digitization_head = CFG()
ModelCfg.digitization_head.kernel_size = 51
ModelCfg.digitization_head.dilation = 1

ModelCfg.digitization_head.num_leads = ModelCfg.num_leads
ModelCfg.digitization_head.fs = 100
ModelCfg.digitization_head.max_len = 10 * ModelCfg.digitization_head.fs
ModelCfg.digitization_head.criterion = CFG()
ModelCfg.digitization_head.criterion.name = "snr_loss"
ModelCfg.digitization_head.criterion.eps = 1e-7
ModelCfg.digitization_head.criterion.reduction = "mean"
