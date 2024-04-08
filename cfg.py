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
BaseCfg.checkpoints = _BASE_DIR / "checkpoints"
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

TrainCfg.checkpoints = BaseCfg.checkpoints
TrainCfg.checkpoints.mkdir(exist_ok=True)

TrainCfg.train_ratio = 0.9

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 27
# TODO: automatic adjust batch size according to GPU capacity
# https://stackoverflow.com/questions/45132809/how-to-select-batch-size-automatically-to-fit-gpu
# GPU memory limit of the Challenge is 64GB
TrainCfg.batch_size = 48  # 64, 128, 256

# configs of optimizers and lr_schedulers
TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
TrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

TrainCfg.learning_rate = 3.3e-3  # 5e-4, 1e-3
TrainCfg.lr = TrainCfg.learning_rate

TrainCfg.lr_scheduler = "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1
TrainCfg.max_lr = 8.7e-3  # for "one_cycle" scheduler, to adjust via expriments

# configs of callbacks, including early stopping, checkpoint, etc.
TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
TrainCfg.early_stopping.patience = TrainCfg.n_epochs // 3
TrainCfg.keep_checkpoint_max = 10

# configs of loss function
# TrainCfg.loss = "AsymmetricLoss"  # "FocalLoss", "BCEWithLogitsLoss"
# TrainCfg.loss_kw = CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")
TrainCfg.flooding_level = 0.0  # flooding performed if positive,

# configs of logging
TrainCfg.log_step = 10
# TrainCfg.eval_every = 20


TrainCfg.predict_dx = True
TrainCfg.predict_digitization = False  # TODO: implement digitization prediction

TrainCfg.monitor = "dx_f_measure"

TrainCfg.debug = True


###############################################################################
# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
###############################################################################


ModelCfg = CFG()
ModelCfg.num_leads = BaseCfg.num_leads
ModelCfg.torch_dtype = BaseCfg.torch_dtype
ModelCfg.model_dir = BaseCfg.model_dir
ModelCfg.checkpoints = BaseCfg.checkpoints


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
# ModelCfg.backbone_name = "facebook/convnextv2-large-22k-384"
ModelCfg.backbone_name = "facebook/convnextv2-atto-1k-224"
ModelCfg.backbone_source = "hf"
ModelCfg.backbone_freeze = True

ModelCfg.dx_head = deepcopy(linear)

ModelCfg.dx_head.out_channels = [
    # containing just the intermediate features
    # not including the input features and the output features
    # 1024,
    256,
]
ModelCfg.dx_head.dropouts = 0.3
ModelCfg.dx_head.activation = "mish"

ModelCfg.dx_head.classes = BaseCfg.classes
ModelCfg.dx_head.num_classes = len(ModelCfg.dx_head.classes)
ModelCfg.dx_head.criterion = "CrossEntropyLoss"
ModelCfg.dx_head.label_smoothing = 0.1

ModelCfg.dx_head.remote_checkpoints = {
    # stores the checkpoints of the dx head
}
ModelCfg.dx_head.remote_checkpoints_name = None  # None for not loading from remote checkpoints

ModelCfg.dx_head.include = TrainCfg.predict_dx

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
ModelCfg.digitization_head.max_len = 12 * ModelCfg.digitization_head.fs
ModelCfg.digitization_head.criterion = CFG()
ModelCfg.digitization_head.criterion.name = "snr_loss"
ModelCfg.digitization_head.criterion.eps = 1e-7
ModelCfg.digitization_head.criterion.reduction = "mean"

ModelCfg.digitization_head.remote_checkpoints = {
    # stores the checkpoints of the digitization head
}
ModelCfg.digitization_head.remote_checkpoints_name = None  # None for not loading from remote checkpoints

ModelCfg.digitization_head.include = TrainCfg.predict_digitization
