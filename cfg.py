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

# fmt: off
BaseCfg.default_class = "OTHER"  # an extra class for empty labels
BaseCfg.classes = [
    # diagnostic superclass
    "NORM", "Acute MI", "Old MI", "STTC", "CD", "HYP", "PAC", "PVC", "AFIB/AFL", "TACHY", "BRADY",
    BaseCfg.default_class,
]
# fmt: on

BaseCfg.lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


###############################################################################
# training configurations for machine learning and deep learning
###############################################################################

TrainCfg = deepcopy(BaseCfg)

TrainCfg.checkpoints = BaseCfg.checkpoints
TrainCfg.checkpoints.mkdir(exist_ok=True)

TrainCfg.train_ratio = 0.9

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 25
# TODO: automatic adjust batch size according to GPU capacity
# https://stackoverflow.com/questions/45132809/how-to-select-batch-size-automatically-to-fit-gpu
# GPU memory limit of the Challenge is 64GB
TrainCfg.batch_size = 16  # 64, 128, 256

# configs of optimizers and lr_schedulers
TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
TrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

TrainCfg.learning_rate = 5e-5  # 5e-4, 1e-3
TrainCfg.lr = TrainCfg.learning_rate

TrainCfg.lr_scheduler = "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1
TrainCfg.max_lr = 1e-4  # for "one_cycle" scheduler, to adjust via expriments

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
TrainCfg.log_step = 100
# TrainCfg.eval_every = 20

# data augmentation
# https://pytorch.org/vision/stable/transforms.html
TrainCfg.use_augmentation = True


TrainCfg.backbone_freeze = False  # whether to freeze the backbone

# tasks to be performed
TrainCfg.predict_dx = True
TrainCfg.predict_digitization = False  # TODO: implement digitization prediction
TrainCfg.predict_bbox = True
TrainCfg.bbox_format = "coco"  # "coco", "pascal_voc", "yolo"
TrainCfg.predict_mask = False  # TODO: implement mask prediction from ROI obtained by object detection

TrainCfg.bbox_mode = "merge_horizontal"  # "roi_only", "merge_horizontal", "full"

TrainCfg.roi_only = False
TrainCfg.roi_padding = 0.0

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
ModelCfg.backbone_name = "facebook/convnextv2-nano-22k-384"
ModelCfg.backbone_source = "hf"
ModelCfg.backbone_freeze = TrainCfg.backbone_freeze

ModelCfg.classification_head = deepcopy(linear)

ModelCfg.classification_head.out_channels = [
    # containing just the intermediate features
    # not including the input features and the output features
    # 1024,
    512,
]
ModelCfg.classification_head.dropouts = 0.3
ModelCfg.classification_head.activation = "mish"

ModelCfg.classification_head.classes = BaseCfg.classes
ModelCfg.classification_head.num_classes = len(ModelCfg.classification_head.classes)
ModelCfg.classification_head.criterion = "AsymmetricLoss"  # "FocalLoss", "BCEWithLogitsLoss"
ModelCfg.classification_head.label_smoothing = 0.1
ModelCfg.classification_head.threshold = 0.5  # threshold for multi-label classification

ModelCfg.classification_head.remote_checkpoints = {
    # stores the checkpoints of the classification head
}
ModelCfg.classification_head.remote_checkpoints_name = None  # None for not loading from remote checkpoints

ModelCfg.classification_head.include = TrainCfg.predict_dx
ModelCfg.classification_head.monitor = "dx_f_measure"

# ModelCfg.digitization_head = deepcopy(linear)
# ModelCfg.digitization_head.out_channels = [
#     # no intermediate features
#     # the (flattened) input features, whose number equals
#     # (backbone output channels) * (backbone output height) * (backbone output width),
#     # are fed into ONE fully connected layer,
#     # and further reshaped to the final output shape
# ]

# digitization_head (NOT used) now use 1D convolutional layer instead
ModelCfg.digitization_head = CFG()
ModelCfg.digitization_head.kernel_size = 51
ModelCfg.digitization_head.dilation = 1

ModelCfg.digitization_head.num_leads = ModelCfg.num_leads
ModelCfg.digitization_head.fs = 500
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


# model config for object detection
ModelCfg.object_detection = CFG()
ModelCfg.object_detection.model_name = "facebook/detr-resnet-50"  # "jozhang97/deta-resnet-50-24-epochs"
ModelCfg.object_detection.source = "hf"
ModelCfg.object_detection.freeze = False
ModelCfg.object_detection.scale = "n"
ModelCfg.object_detection.mode = TrainCfg.bbox_mode

ModelCfg.object_detection.class_names = BaseCfg.lead_names + ["waveform"]
ModelCfg.object_detection.num_classes = len(ModelCfg.object_detection.class_names)
ModelCfg.object_detection.label2id = {label: i for i, label in enumerate(ModelCfg.object_detection.class_names)}

ModelCfg.object_detection.num_queries = 50

ModelCfg.object_detection.bbox_thr = 0.5
ModelCfg.object_detection.nms_thr = 0.4

ModelCfg.object_detection.monitor = "detection_map"


ModelCfg.digitizer = CFG()
ModelCfg.digitizer.source = "custom"
ModelCfg.digitizer.model_name = "unet"

ModelCfg.digitizer.input_mode = "raw"  # "thresholded", "raw", "both"
ModelCfg.digitizer.input_shape = {"height": 512, "width": 512 * 2}
ModelCfg.digitizer.input_norm = {"mean": (0.5,), "std": (0.5,), "max_pixel_value": 255}  # NOT used currently
ModelCfg.digitizer.threshold_percentile = 1.0  # for mode "thresholded" and "both", NOT used currently
ModelCfg.digitizer.highest_weight = 10

ModelCfg.digitizer.num_classes = 1

ModelCfg.digitizer.monitor = "segmentation_dice"
