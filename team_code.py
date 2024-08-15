#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import median_filter  # noqa: F401
from sklearn.cluster import KMeans
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import str2bool
from torch_ecg.utils.utils_signal import resample_irregular_timeseries

from cfg import BaseCfg, ModelCfg, TrainCfg  # noqa: F401
from const import MODEL_CACHE_DIR, PROJECT_DIR, REMOTE_MODELS
from data_reader import CINC2024Reader
from dataset import CinC2024Dataset
from helper_code import (  # noqa: F401
    compute_one_hot_encoding,
    find_records,
    get_header_file,
    get_num_samples,
    get_num_signals,
    get_sampling_frequency,
    get_signal_names,
    get_signal_units,
    load_images,
    load_labels,
    load_text,
)
from models import ECGWaveformDetector, ECGWaveformDigitizer, MultiHead_CINC2024
from trainer import CINC2024Trainer
from utils.ecg_simulator import evolve_standard_12_lead_ecg

################################################################################
# environment variables

os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HOME"] = str(Path(MODEL_CACHE_DIR).parent)

try:
    TEST_FLAG = os.environ.get("CINC2024_REVENGER_TEST", False)
    TEST_FLAG = str2bool(TEST_FLAG)
except Exception:
    TEST_FLAG = False

# MODEL_DIR = "revenger_model_dir"
# SYNTHETIC_IMAGE_DIR = "revenger_synthetic_image_dir"
# MODEL_DIR and SYNTHETIC_IMAGE_DIR are subfolders of model_folder passed to the functions

################################################################################


################################################################################
# NOTE: configurable options

# model choices, ref. const.py

SubmissionCfg = CFG()
SubmissionCfg.detector = "hf--facebook/detr-resnet-50"
SubmissionCfg.classifier = "hf--facebook/convnextv2-nano-22k-384"
SubmissionCfg.digitizer = "custom--unet"

SubmissionCfg.final_model_name = {
    "detector": "detector.pth.tar",
    "classifier": "classifier.pth.tar",
    "digitizer": "digitizer.pth.tar",
}

################################################################################


################################################################################
# NOTE: constants

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32

CINC2024Reader.__DEBUG__ = False
CinC2024Dataset.__DEBUG__ = False
MultiHead_CINC2024.__DEBUG__ = False
CINC2024Trainer.__DEBUG__ = False

################################################################################

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################


def train_models(
    data_folder: Union[str, bytes, os.PathLike], model_folder: Union[str, bytes, os.PathLike], verbose: bool = False
) -> None:
    """Train the models.

    Parameters
    ----------
    data_folder : `path_like`
        The path to the folder containing the training data.
    model_folder : `path_like`
        The path to the folder where the model will be saved.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    None

    """
    print("\n" + "*" * 100)
    msg = "   CinC2024 challenge training entry starts   ".center(100, "#")
    print(msg)
    print("*" * 100 + "\n")

    # Find data files.
    if verbose:
        print("Training the models...")
        print("Finding the Challenge data...")

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError("No data was provided.")

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    model_folder = Path(model_folder).expanduser().resolve()
    data_folder = Path(data_folder).expanduser().resolve()
    (Path(model_folder) / "working_dir").mkdir(parents=True, exist_ok=True)

    reader_kwargs = {
        # "db_dir": Path(DATA_CACHE_DIR),
        "db_dir": Path(data_folder).expanduser().resolve(),
        "working_dir": (Path(model_folder) / "working_dir"),
        "synthetic_images_dir": Path(model_folder) / "working_dir" / "synthetic_images",
        # "aux_files_dir": Path(DATA_CACHE_DIR) / "aux_files",  # ref. post_docker_build.py
        "aux_files_dir": Path(PROJECT_DIR) / "aux_files",  # ref. post_docker_build.py
    }

    # generate the synthetic images
    print("Preparing the synthetic images...")
    dr = CINC2024Reader(**reader_kwargs)
    # gen_img_config = dr.__gen_img_extra_configs__[0].copy()  # requires much more storage and is much slower
    gen_img_config = dr.__gen_img_default_config__.copy()
    gen_img_config["write_signal_file"] = True
    dr.prepare_synthetic_images(parallel=True, force_recompute=True, **gen_img_config)
    del dr
    print("Done.")

    # Train the models
    if SubmissionCfg.classifier is not None:
        train_classification_model(data_folder, model_folder, verbose)
    if SubmissionCfg.detector is not None:
        train_object_detection_model(data_folder, model_folder, verbose)
    if SubmissionCfg.digitizer is not None:
        train_digitization_model(data_folder, model_folder, verbose)

    print("\n" + "*" * 100)
    msg = "   CinC2024 challenge training entry ends   ".center(100, "#")
    print(msg)


def load_models(
    model_folder: Union[str, bytes, os.PathLike], verbose: bool = False
) -> Tuple[Dict[str, Union[dict, nn.Module]], Dict[str, Union[dict, nn.Module]]]:
    """Load the trained models.

    Parameters
    ----------
    model_folder : `path_like`
        The path to the folder containing the trained model.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    digitization_model : Dict[str, Union[dict, nn.Module]]
        The trained digitization model, object detection model and their training configurations.
    classification_model : Dict[str, Union[dict, nn.Module]]
        The trained classification model and its training configurations.

    """
    model_folder = Path(model_folder).expanduser().resolve()

    print("Loading the trained models...")

    digitization_model = {}
    if SubmissionCfg.detector is not None:
        # detector, detector_train_cfg = ECGWaveformDetector.from_checkpoint(
        #     Path(MODEL_CACHE_DIR) / REMOTE_MODELS[SubmissionCfg.detector]["filename"]
        # )
        model_path = Path(model_folder) / SubmissionCfg.final_model_name["detector"]
        detector, detector_train_cfg = ECGWaveformDetector.from_checkpoint(model_path, device=DEVICE)
        digitization_model["detector"] = detector
        digitization_model["detector_train_cfg"] = detector_train_cfg

        print(f"Object detection model loaded from {str(model_path)}")
    if SubmissionCfg.digitizer is not None:
        # digitizer, digitizer_train_cfg = ECGWaveformDigitizer.from_checkpoint(
        #     Path(MODEL_CACHE_DIR) / REMOTE_MODELS[SubmissionCfg.digitizer]["filename"]
        # )
        model_path = Path(model_folder) / SubmissionCfg.final_model_name["digitizer"]
        digitizer, digitizer_train_cfg = ECGWaveformDigitizer.from_checkpoint(model_path, device=DEVICE)
        digitization_model["digitizer"] = digitizer
        digitization_model["digitizer_train_cfg"] = digitizer_train_cfg

        print(f"Digitization model loaded from {str(model_path)}")

    classification_model = {}
    if SubmissionCfg.classifier is not None:
        # classifier, classifier_train_cfg = MultiHead_CINC2024.from_checkpoint(
        #     Path(MODEL_CACHE_DIR) / REMOTE_MODELS[SubmissionCfg.classifier]["filename"]
        # )
        model_path = Path(model_folder) / SubmissionCfg.final_model_name["classifier"]
        classifier, classifier_train_cfg = MultiHead_CINC2024.from_checkpoint(model_path, device=DEVICE)
        classification_model["classifier"] = classifier
        classification_model["classifier_train_cfg"] = classifier_train_cfg

        print(f"Classification model loaded from {str(model_path)}")

    return digitization_model, classification_model


# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
@torch.no_grad()
def run_models(
    record: Union[str, bytes, os.PathLike],
    digitization_model: Dict[str, Union[dict, nn.Module]],
    classification_model: Dict[str, Union[dict, nn.Module]],
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Run the digitization model on a record.

    Parameters
    ----------
    record : `path_like`
        The path to the record to process, without the file extension.
    digitization_model : Dict[str, Union[dict, nn.Module]]
        The trained digitization model, object detection model and their training configurations.
    classification_model : Dict[str, Union[dict, nn.Module]]
        The trained classification model and its training configurations.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    numpy.ndarray
        The digitized signal.
    list of str
        The predicted labels.

    """
    input_images = load_images(record)  # a list of PIL.Image.Image
    # convert to RGB (it's possible that the images are RGBA format)
    input_images = [img.convert("RGB") for img in input_images]
    image_shapes = [{"width": img.width, "height": img.height} for img in input_images]

    if "detector" in digitization_model:
        detector = digitization_model["detector"]
        bbox = detector.inference(input_images).bbox  # a list of dict
        # crop the input images using the "roi" of each dict in the bbox
        # the "roi" is a list of 4 integers [xmin, ymin, xmax, ymax]
        if "classifier" in classification_model and classification_model["classifier_train_cfg"].roi_only:
            cropped_images_for_classifier = get_cropped_images(
                input_images, bbox, roi_padding=classification_model["classifier_train_cfg"].roi_padding
            )
        else:
            cropped_images_for_classifier = input_images

        if "digitizer" in digitization_model and digitization_model["digitizer_train_cfg"].roi_only:
            cropped_images_for_digitizer = get_cropped_images(
                input_images, bbox, roi_padding=digitization_model["digitizer_train_cfg"].roi_padding
            )
            shifted_bbox = get_shifted_bbox(bbox, roi_padding=digitization_model["digitizer_train_cfg"].roi_padding)
        else:
            cropped_images_for_digitizer = input_images
            shifted_bbox = bbox
    else:
        cropped_images_for_classifier = input_images
        cropped_images_for_digitizer = input_images
        bbox = None

    if "classifier" in classification_model:
        classifier = classification_model["classifier"]
        output = classifier.inference(
            cropped_images_for_classifier, threshold=REMOTE_MODELS[SubmissionCfg.classifier]["threshold"]
        )
        dx_classes = output.dx_classes
        dx_prob = np.asarray(output.dx_prob)  # of shape (n_samples, n_classes), n_samples is typically 1 but not always
        # take max pooling along the samples
        dx_prob = dx_prob.max(axis=0)
        labels = [
            dx_classes[idx] for idx, prob in enumerate(dx_prob) if prob > REMOTE_MODELS[SubmissionCfg.classifier]["threshold"]
        ]
        # remove the "OTHER" label (BaseCfg.default_class) if present
        labels = [label for label in labels if label != BaseCfg.default_class]
    else:
        labels = None

    if "digitizer" in digitization_model:
        digitizer = digitization_model["digitizer"]
        waveform_mask = digitizer.inference(cropped_images_for_digitizer).waveform_mask  # a list of np.ndarray

        try:
            signal = bbox_and_mask_to_signals(
                bbox=shifted_bbox,
                mask=waveform_mask,
                record=record,
            )
        except Exception as e:
            print(f"Error: {e}")
            signal = digitization_workaround(record)
    else:
        use_workaround = False  # True or False

        if use_workaround:
            signal = digitization_workaround(record)
        else:
            signal = None

    return signal, labels


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################


def get_cropped_images(
    images: List[Image.Image], bboxes: List[Dict[str, Union[int, List[int]]]], roi_padding: float
) -> List[Image.Image]:
    """Crop the images using the bounding boxes.

    Parameters
    ----------
    images : list of PIL.Image.Image
        The images to crop.
    bboxes : list of dict
        The bounding boxes.
    roi_padding : float
        The padding ratio for the bounding boxes.

    Returns
    -------
    list of PIL.Image.Image
        The cropped images.

    """
    if roi_padding > 0:
        roi = []
        for img, b_dict in zip(images, bboxes):
            roi_width = b_dict["roi"][2] - b_dict["roi"][0]
            roi_height = b_dict["roi"][3] - b_dict["roi"][1]
            width_padding = int(roi_width * roi_padding)
            height_padding = int(roi_height * roi_padding)
            roi.append(
                [
                    max(0, b_dict["roi"][0] - width_padding),
                    max(0, b_dict["roi"][1] - height_padding),
                    min(img.width, b_dict["roi"][2] + width_padding),
                    min(img.height, b_dict["roi"][3] + height_padding),
                ]
            )
    else:
        roi = [b_dict["roi"] for b_dict in bboxes]

    cropped_images = [img.crop(roi_) for img, roi_ in zip(images, roi)]

    return cropped_images


def get_shifted_bbox(
    bboxes: List[Dict[str, Union[int, List[int]]]], roi_padding: float
) -> List[Dict[str, Union[int, List[int]]]]:
    """Shift the bounding boxes.

    Parameters
    ----------
    bboxes : list of dict
        The bounding boxes.
    roi_padding : float
        The padding ratio for the bounding boxes.

    Returns
    -------
    list of dict
        The shifted bounding boxes.

    """
    shifted_bbox = []
    for b_dict in bboxes:
        if roi_padding == 0:
            shift_x = b_dict["roi"][0]
            shift_y = b_dict["roi"][1]
            new_dict = {
                "scores": b_dict["scores"],
                "image_size": (b_dict["roi"][3] - b_dict["roi"][1], b_dict["roi"][2] - b_dict["roi"][0]),
                "category_id": b_dict["category_id"],
                "category_name": b_dict["category_name"],
                "roi": [0, 0, b_dict["roi"][2] - shift_x, b_dict["roi"][3] - shift_y],
                "bbox": np.array(b_dict["bbox"]) - np.array([shift_x, shift_y, shift_x, shift_y]),
            }
            shifted_bbox.append(new_dict)
        else:
            roi_width = b_dict["roi"][2] - b_dict["roi"][0]
            roi_height = b_dict["roi"][3] - b_dict["roi"][1]
            width_padding = int(roi_width * roi_padding)
            height_padding = int(roi_height * roi_padding)
            new_roi = [
                max(0, b_dict["roi"][0] - width_padding),
                max(0, b_dict["roi"][1] - height_padding),
                min(b_dict["image_size"][1], b_dict["roi"][2] + width_padding),
                min(b_dict["image_size"][0], b_dict["roi"][3] + height_padding),
            ]
            shift_x = new_roi[0]
            shift_y = new_roi[1]
            new_dict = {
                "scores": b_dict["scores"],
                "image_size": (new_roi[3] - new_roi[1], new_roi[2] - new_roi[0]),
                "category_id": b_dict["category_id"],
                "category_name": b_dict["category_name"],
                "roi": [0, 0, new_roi[2] - shift_x, new_roi[3] - shift_y],
                "bbox": np.array(b_dict["bbox"]) - np.array([shift_x, shift_y, shift_x, shift_y]),
            }
            shifted_bbox.append(new_dict)

    return shifted_bbox


def train_classification_model(
    data_folder: Union[str, bytes, os.PathLike], model_folder: Union[str, bytes, os.PathLike], verbose: bool
) -> None:
    """Train the classification model.

    Parameters
    ----------
    data_folder : `path_like`
        The path to the folder containing the training data.
    model_folder : `path_like`
        The path to the folder where the model will be saved.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    None

    """
    model_folder = Path(model_folder).expanduser().resolve()
    data_folder = Path(data_folder).expanduser().resolve()

    print("Training the classification model...")

    model, train_config = MultiHead_CINC2024.from_checkpoint(
        Path(MODEL_CACHE_DIR) / REMOTE_MODELS[SubmissionCfg.classifier]["filename"],
        device=DEVICE,
    )
    model_config = model.config

    # adjust the training configuration
    train_config.db_dir = Path(data_folder).expanduser().resolve()
    train_config.model_dir = Path(model_folder).expanduser().resolve()
    train_config.working_dir = Path(model_folder) / "working_dir"
    train_config.checkpoints = train_config.working_dir / "checkpoints"
    train_config.log_dir = train_config.working_dir / "log"
    train_config.synthetic_images_dir = train_config.working_dir / "synthetic_images"

    train_config.final_model_name = SubmissionCfg.final_model_name["classifier"]
    train_config.debug = False

    # the learning rate is set low so that the fine-tuned model
    # is not too different from the pretrained model
    train_config.n_epochs = 1
    train_config.learning_rate = 5e-6  # 5e-4, 1e-3
    train_config.lr = train_config.learning_rate
    train_config.max_lr = 1e-5
    train_config.early_stopping.patience = train_config.n_epochs // 3

    train_config.predict_dx = True
    train_config.predict_bbox = False
    train_config.predict_mask = False
    # train_config.roi_only = False
    # train_config.roi_padding = 0.0

    train_config.backbone_freeze = False

    if TEST_FLAG:
        train_config.batch_size = 4
        train_config.log_step = 5
    else:
        train_config.batch_size = 16  # 16G (Tesla T4)
        train_config.log_step = 120

    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)
    model = model.to(device=DEVICE)
    if verbose:
        if isinstance(model, DP):
            print("model size:", model.module.module_size, model.module.module_size_)
        else:
            print("model size:", model.module_size, model.module_size_)

    reader_kwargs = {
        "db_dir": Path(data_folder).expanduser().resolve(),
        "working_dir": (Path(model_folder) / "working_dir"),
        "synthetic_images_dir": Path(model_folder) / "working_dir" / "synthetic_images",
        # "aux_files_dir": Path(DATA_CACHE_DIR) / "aux_files",  # ref. post_docker_build.py
        "aux_files_dir": Path(PROJECT_DIR) / "aux_files",  # ref. post_docker_build.py
    }

    ds_train = CinC2024Dataset(train_config, training=True, lazy=True, **reader_kwargs)
    ds_test = CinC2024Dataset(train_config, training=False, lazy=True, **reader_kwargs)
    if verbose:
        print(f"train size: {len(ds_train)}, test size: {len(ds_test)}")

    trainer = CINC2024Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=True,
    )
    if TEST_FLAG:
        # switch the dataloaders to make the test faster
        # the first dataloader is used for both training and evaluation
        # the second dataloader is used for validation only
        # trainer._setup_dataloaders(ds_test, ds_train)
        trainer._setup_dataloaders(ds_test, None)
    else:
        trainer._setup_dataloaders(ds_train, ds_test)

    best_state_dict = trainer.train()  # including saving model

    trainer.log_manager.flush()
    trainer.log_manager.close()

    del trainer
    del model
    del best_state_dict

    torch.cuda.empty_cache()

    print("Classification model training completed.")


def train_object_detection_model(
    data_folder: Union[str, bytes, os.PathLike], model_folder: Union[str, bytes, os.PathLike], verbose: bool
) -> None:
    """Train the object detection model.

    Parameters
    ----------
    data_folder : `path_like`
        The path to the folder containing the training data.
    model_folder : `path_like`
        The path to the folder where the model will be saved.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    None

    """
    model_folder = Path(model_folder).expanduser().resolve()
    data_folder = Path(data_folder).expanduser().resolve()

    print("Training the object detection model...")

    model, train_config = ECGWaveformDetector.from_checkpoint(
        Path(MODEL_CACHE_DIR) / REMOTE_MODELS[SubmissionCfg.detector]["filename"],
        device=DEVICE,
    )
    model_config = model.config

    # adjust the training configuration
    train_config.db_dir = Path(data_folder).expanduser().resolve()
    train_config.model_dir = Path(model_folder).expanduser().resolve()
    train_config.working_dir = Path(model_folder) / "working_dir"
    train_config.checkpoints = train_config.working_dir / "checkpoints"
    train_config.log_dir = train_config.working_dir / "log"
    train_config.synthetic_images_dir = train_config.working_dir / "synthetic_images"

    train_config.final_model_name = SubmissionCfg.final_model_name["detector"]
    train_config.debug = False

    train_config.n_epochs = 1
    train_config.learning_rate = 5e-6  # 5e-4, 1e-3
    train_config.lr = train_config.learning_rate
    train_config.max_lr = 1e-5
    train_config.early_stopping.patience = train_config.n_epochs // 3

    train_config.predict_dx = False
    train_config.predict_bbox = True
    train_config.predict_mask = False
    # train_config.roi_only = False
    # train_config.roi_padding = 0.0

    # train_config.backbone_freeze = True

    if TEST_FLAG:
        train_config.batch_size = 1
        train_config.log_step = 5
    else:
        train_config.batch_size = 10  # 16G (Tesla T4)
        train_config.log_step = 120

    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)
    model = model.to(device=DEVICE)
    if verbose:
        if isinstance(model, DP):
            print("model size:", model.module.module_size, model.module.module_size_)
        else:
            print("model size:", model.module_size, model.module_size_)

    reader_kwargs = {
        "db_dir": Path(data_folder).expanduser().resolve(),
        "working_dir": (Path(model_folder) / "working_dir"),
        "synthetic_images_dir": Path(model_folder) / "working_dir" / "synthetic_images",
        # "aux_files_dir": Path(DATA_CACHE_DIR) / "aux_files",  # ref. post_docker_build.py
        "aux_files_dir": Path(PROJECT_DIR) / "aux_files",  # ref. post_docker_build.py
    }

    ds_train = CinC2024Dataset(train_config, training=True, lazy=True, **reader_kwargs)
    ds_test = CinC2024Dataset(train_config, training=False, lazy=True, **reader_kwargs)
    if verbose:
        print(f"train size: {len(ds_train)}, test size: {len(ds_test)}")

    trainer = CINC2024Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=True,
    )

    if TEST_FLAG:
        # switch the dataloaders to make the test faster
        # the first dataloader is used for both training and evaluation
        # the second dataloader is used for validation only
        # trainer._setup_dataloaders(ds_test, ds_train)
        trainer._setup_dataloaders(ds_test, None)
    else:
        trainer._setup_dataloaders(ds_train, ds_test)

    best_state_dict = trainer.train()  # including saving model

    trainer.log_manager.flush()
    trainer.log_manager.close()

    del trainer
    del model
    del best_state_dict

    torch.cuda.empty_cache()

    print("Object detection model training completed.")


def train_digitization_model(
    data_folder: Union[str, bytes, os.PathLike], model_folder: Union[str, bytes, os.PathLike], verbose: bool
) -> None:
    """Train the digitization model.

    Parameters
    ----------
    data_folder : `path_like`
        The path to the folder containing the training data.
    model_folder : `path_like`
        The path to the folder where the model will be saved.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    None

    """
    model_folder = Path(model_folder).expanduser().resolve()
    data_folder = Path(data_folder).expanduser().resolve()

    print("Training the digitization model...")

    model, train_config = ECGWaveformDigitizer.from_checkpoint(
        Path(MODEL_CACHE_DIR) / REMOTE_MODELS[SubmissionCfg.digitizer]["filename"],
        device=DEVICE,
    )
    model_config = model.config

    # adjust the training configuration
    train_config.db_dir = Path(data_folder).expanduser().resolve()
    train_config.model_dir = Path(model_folder).expanduser().resolve()
    train_config.working_dir = Path(model_folder) / "working_dir"
    train_config.checkpoints = train_config.working_dir / "checkpoints"
    train_config.log_dir = train_config.working_dir / "log"
    train_config.synthetic_images_dir = train_config.working_dir / "synthetic_images"

    train_config.final_model_name = SubmissionCfg.final_model_name["digitizer"]
    train_config.debug = False
    train_config.n_epochs = 1
    train_config.learning_rate = 5e-6  # 5e-4, 1e-3
    train_config.lr = train_config.learning_rate
    train_config.max_lr = 1e-5
    train_config.early_stopping.patience = train_config.n_epochs // 3

    train_config.predict_dx = False
    train_config.predict_bbox = False
    train_config.predict_mask = True
    # train_config.roi_only = False
    # train_config.roi_padding = 0.0

    # train_config.backbone_freeze = True

    if TEST_FLAG:
        train_config.batch_size = 1
        train_config.log_step = 5
    else:
        train_config.batch_size = 3  # 16G (Tesla T4)
        train_config.log_step = 200

    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)
    model = model.to(device=DEVICE)
    if verbose:
        if isinstance(model, DP):
            print("model size:", model.module.module_size, model.module.module_size_)
        else:
            print("model size:", model.module_size, model.module_size_)

    reader_kwargs = {
        "db_dir": Path(data_folder).expanduser().resolve(),
        "working_dir": (Path(model_folder) / "working_dir"),
        "synthetic_images_dir": Path(model_folder) / "working_dir" / "synthetic_images",
        # "aux_files_dir": Path(DATA_CACHE_DIR) / "aux_files",  # ref. post_docker_build.py
        "aux_files_dir": Path(PROJECT_DIR) / "aux_files",  # ref. post_docker_build.py
    }

    ds_train = CinC2024Dataset(train_config, training=True, lazy=True, **reader_kwargs)
    ds_test = CinC2024Dataset(train_config, training=False, lazy=True, **reader_kwargs)
    if verbose:
        print(f"train size: {len(ds_train)}, test size: {len(ds_test)}")

    trainer = CINC2024Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=True,
    )

    if TEST_FLAG:
        # switch the dataloaders to make the test faster
        # the first dataloader is used for both training and evaluation
        # the second dataloader is used for validation only
        # trainer._setup_dataloaders(ds_test, ds_train)
        trainer._setup_dataloaders(ds_test, None)
    else:
        trainer._setup_dataloaders(ds_train, ds_test)

    best_state_dict = trainer.train()  # including saving model

    trainer.log_manager.flush()
    trainer.log_manager.close()

    del trainer
    del model
    del best_state_dict

    torch.cuda.empty_cache()

    print("Digitization model training completed.")


def bbox_and_mask_to_signals(
    bbox: List[Dict[str, Union[int, List[int]]]],
    mask: List[np.ndarray],
    record: Union[str, bytes, os.PathLike],
) -> np.ndarray:
    """Convert the bounding boxes and masks to signals.

    Parameters
    ----------
    bbox : list of dict
        The bounding boxes.
    mask : list of np.ndarray
        The masks.
    record : `path_like`
        The path to the record to process, without file extension.

    """
    # a macro grid of the ECG paper corresponds to 0.2 second and 0.5 mV
    ts_to_mv = 0.5 / 0.2  # mV per ms

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)
    num_samples = get_num_samples(header)  # length of the signal
    num_signals = get_num_signals(header)  # channels of the signal
    signal_names = get_signal_names(header)  # names of the signals (lead names)
    signal_names = [sn.upper() for sn in signal_names]
    signal_fs = get_sampling_frequency(header)  # sampling frequency of the signal
    signal_duration = num_samples / signal_fs  # duration of the signal
    # signal_units = get_signal_units(header)  # units of the signals
    # median_filter_window = int(signal_fs * 1.5)  # 1.5 seconds

    signal = np.zeros((num_samples, num_signals))

    standard_lead_names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    if set(signal_names).issubset(standard_lead_names):
        indices_mapping = [standard_lead_names.index(sn) for sn in signal_names]
    else:
        indices_mapping = np.arange(num_signals, dtype=int).tolist()

    if bbox is not None:
        # process the bounding boxes
        bbox = [
            [
                {"bbox": box, "category_id": cid, "category_name": cn}
                for box, cid, cn in zip(b_dict["bbox"], b_dict["category_id"], b_dict["category_name"])
            ]
            for b_dict in bbox
        ]
    else:
        bbox = [[] for _ in range(len(mask))]

    for img_idx, (b_list, waveform_mask) in enumerate(zip(bbox, mask)):
        if img_idx >= num_signals:
            break
        waveform_pixels = np.where(waveform_mask > 0)
        # to array of shape (n_pixels, 2) of (y, x)
        waveform_pixels = np.array(waveform_pixels).T
        xmin = waveform_pixels[:, 1].min()
        xmax = waveform_pixels[:, 1].max()
        # xmax - xmin typically corresponds to 10 seconds of ECG signal
        sig_start = int(img_idx * signal_fs * 10)
        sig_len = int(min(10 * signal_fs, num_samples - sig_start))
        x_to_ts = sig_len / signal_fs / (xmax - xmin)  # seconds per pixel
        y_to_mv = ts_to_mv * x_to_ts  # mV per pixel

        waveform_bbox_indices = [idx for idx, b_dict in enumerate(b_list) if b_dict["category_name"] == "waveform"]
        waveform_bbox = sorted([b_list[idx]["bbox"] for idx in waveform_bbox_indices], key=lambda box: box[1])
        # good case: we have 4 bounding boxes which does not overlap
        if len(waveform_bbox) == 4 and not is_bbox_overlap(waveform_bbox):
            # print("good case")
            for box_idx, box in enumerate(waveform_bbox):
                xmin, ymin, xmax, ymax = box
                # print(f"box {box_idx}: ymin={ymin}, xmin={xmin}, ymax={ymax}, xmax={xmax}, waveform_mask.shape={waveform_mask.shape}")
                # find the pixels of the waveform in the bounding box
                waveform_pixels = np.where(waveform_mask[ymin:ymax, xmin:xmax] > 0)
                # viz_box = waveform_mask[ymin:ymax, xmin:xmax].copy()
                waveform_pixels = pd.DataFrame(np.array(waveform_pixels).T, columns=["y", "x"])
                # sort the pixels by the x-coordinates
                waveform_pixels = waveform_pixels.sort_values("x")
                # the y-axis of the image is from top to bottom, so we need to flip it
                waveform_pixels["y"] = -1 * waveform_pixels["y"]
                # drop duplicates in the x-coordinates, keep the mean of the y-coordinates
                waveform_pixels = waveform_pixels.groupby("x").mean().reset_index()
                # apply median filter to the signal
                waveform_pixels["y"] = waveform_pixels["y"] - np.median(waveform_pixels["y"].values)
                # transform the y-coordinates to mV, x-coordinates to seconds
                waveform_pixels["y"] = waveform_pixels["y"] * y_to_mv
                waveform_pixels["x"] = waveform_pixels["x"] * x_to_ts
                waveform_pixels = waveform_pixels[["x", "y"]]
                # interpolate the waveform_pixels to the length of the signal
                tnew = np.linspace(0, sig_len / signal_fs, num=sig_len)
                # waveform_pixels = waveform_pixels[waveform_pixels["x"] < tnew[-1]]
                y = resample_irregular_timeseries(
                    waveform_pixels.values, tnew=tnew, method="interp1d", interp_kw=dict(fill_value="extrapolate")
                )
                # leads from top to bottom and left to right:
                # I, aVR, V1, V4
                # II, aVL, V2, V5
                # III, aVF, V3, V6
                # II (full)
                if box_idx == 3:  # lead II
                    signal[sig_start : sig_start + sig_len, standard_lead_names.index("II")] = y
                elif box_idx == 0:  # lead I, aVR, V1, V4
                    lead_Len = sig_len // 4
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("I")] = y[:lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("AVR")] = y[lead_Len : 2 * lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("V1")] = y[2 * lead_Len : 3 * lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("V4")] = y[3 * lead_Len :]
                elif box_idx == 1:  # lead II, aVL, V2, V5
                    lead_Len = sig_len // 4
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("II")] = y[:lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("AVL")] = y[lead_Len : 2 * lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("V2")] = y[2 * lead_Len : 3 * lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("V5")] = y[3 * lead_Len :]
                elif box_idx == 2:  # lead III, aVF, V3, V6
                    lead_Len = sig_len // 4
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("III")] = y[:lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("AVF")] = y[lead_Len : 2 * lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("V3")] = y[2 * lead_Len : 3 * lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("V6")] = y[3 * lead_Len :]
        else:
            # bad case: we have less than 4 bounding boxes or the bounding boxes overlap
            # find 4 centroids of the y-coordinates of the waveform_pixels
            # the 4 centroids correspond to the 4 leads of the ECG signal
            kmeans = KMeans(n_clusters=4).fit(waveform_pixels[:, 0].reshape(-1, 1))
            centroids = sorted(kmeans.cluster_centers_.flatten().astype(int).tolist())
            diff = np.diff(centroids).mean().astype(int).item()
            # print(centroids, diff)
            for axis_idx, axis in enumerate(centroids):
                top = 0 if axis_idx == 0 else centroids[axis_idx - 1] + int(0.3 * diff)
                bottom = waveform_mask.shape[0] if axis_idx == 3 else centroids[axis_idx + 1] - int(0.3 * diff)
                waveform_pixels = np.where(waveform_mask[top:bottom, :] > 0)
                waveform_pixels = pd.DataFrame(np.array(waveform_pixels).T, columns=["y", "x"])
                # sort the pixels by the x-coordinates
                waveform_pixels = waveform_pixels.sort_values("x")
                # the y-axis of the image is from top to bottom, so we need to flip it
                waveform_pixels["y"] = -1 * waveform_pixels["y"]
                waveform_pixels["dist_to_centroid"] = np.abs(waveform_pixels["y"] - axis)
                # drop duplicates in the x-coordinates, keep the item with the smallest distance to the centroid
                waveform_pixels = waveform_pixels.loc[waveform_pixels.groupby("x")["dist_to_centroid"].idxmin()].reset_index(
                    drop=True
                )
                waveform_pixels = waveform_pixels.sort_values("x").drop(columns="dist_to_centroid")
                # apply median filter to the signal
                waveform_pixels["y"] = waveform_pixels["y"] - np.median(waveform_pixels["y"].values)
                # transform the y-coordinates to mV, x-coordinates to seconds
                waveform_pixels["y"] = waveform_pixels["y"] * y_to_mv
                waveform_pixels["x"] = waveform_pixels["x"] * x_to_ts
                waveform_pixels = waveform_pixels[["x", "y"]]
                # remove possible outliers in the y-coordinates
                # waveform_pixels = remove_curve_outlier(waveform_pixels, window_size=int(signal_fs * 0.1))
                tnew = np.linspace(0, sig_len / signal_fs, num=sig_len)
                y = resample_irregular_timeseries(
                    waveform_pixels.values, tnew=tnew, method="interp1d", interp_kw=dict(fill_value="extrapolate")
                )
                # leads from top to bottom and left to right:
                # I, aVR, V1, V4
                # II, aVL, V2, V5
                # III, aVF, V3, V6
                # II (full)
                if axis_idx == 3:
                    signal[sig_start : sig_start + sig_len, standard_lead_names.index("II")] = y
                elif axis_idx == 0:
                    lead_Len = sig_len // 4
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("I")] = y[:lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("AVR")] = y[lead_Len : 2 * lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("V1")] = y[2 * lead_Len : 3 * lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("V4")] = y[3 * lead_Len :]
                elif axis_idx == 1:
                    lead_Len = sig_len // 4
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("II")] = y[:lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("AVL")] = y[lead_Len : 2 * lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("V2")] = y[2 * lead_Len : 3 * lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("V5")] = y[3 * lead_Len :]
                elif axis_idx == 2:
                    lead_Len = sig_len // 4
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("III")] = y[:lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("AVF")] = y[lead_Len : 2 * lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("V3")] = y[2 * lead_Len : 3 * lead_Len]
                    signal[sig_start : sig_start + lead_Len, standard_lead_names.index("V6")] = y[3 * lead_Len :]

    # if signal_units.lower() == "mv":
    #     return signal
    # else:
    #     return np.asarray(signal * 1000, dtype=np.int16)

    return signal


def digitization_workaround(record: Union[str, bytes, os.PathLike]) -> np.ndarray:
    """workaround for digitization errors by returning a random signal (or nan values).

    Parameters
    ----------
    record : `path_like`
        The path to the record to process, without file extension.

    Returns
    -------
    numpy.ndarray
        A randomly generated digitized signal.

    """
    # The following code block comes from the official baseline.

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)  # length of the signal
    num_signals = get_num_signals(header)  # channels of the signal
    signal_names = get_signal_names(header)  # names of the signals (lead names)
    signal_names = [sn.upper() for sn in signal_names]
    signal_fs = get_sampling_frequency(header)  # sampling frequency of the signal
    signal_duration = num_samples / signal_fs  # duration of the signal
    # signal_units = get_signal_units(header)  # units of the signals

    standard_lead_names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    if set(signal_names).issubset(standard_lead_names):
        indices_mapping = [standard_lead_names.index(sn) for sn in signal_names]
    else:
        indices_mapping = np.arange(num_signals, dtype=int).tolist()

    bpm = np.clip(np.random.default_rng().normal(80, 23), 50, 120)
    start_idx = np.random.default_rng().integers(low=int(2.5 * signal_fs), high=int(5.5 * signal_fs))
    try:
        signal = evolve_standard_12_lead_ecg(
            signal_duration + 8,
            fs=signal_fs,
            bpm=bpm,
            remove_baseline=0.8,
            return_phase=False,
            return_format="lead_last",
        )["ecg"][start_idx : start_idx + num_samples]
        signal = signal[:, indices_mapping]
    except Exception:
        print("Failed to generate the signal. Return random signal.")
        # official baseline uses random signal
        signal = np.random.default_rng().uniform(low=-1000, high=1000, size=(num_samples, num_signals))

    # if signal_units.lower() == "mv":
    #     signal = signal / 1000.0
    # else:
    #     signal = np.asarray(signal, dtype=np.int16)

    return signal / 1000.0


def is_bbox_overlap(bbox: Sequence[Sequence[int]]) -> bool:
    """Check if the bounding boxes overlap with the region of interest.

    Parameters
    ----------
    bbox : sequence of sequence of int
        The bounding boxes of VOC format.
    roi : sequence of int
        The region of interest.

    Returns
    -------
    bool
        Whether the bounding boxes overlap with the region of interest.

    """
    bbox = np.array(bbox)  # shape (n_boxes, 4)
    mask = np.zeros((bbox[:, 3].max(), bbox[:, 2].max()), dtype=np.uint8)
    for box in bbox:
        mask[box[1] : box[3], box[0] : box[2]] += 1
    return mask.max() > 1


def remove_curve_outlier(curve: pd.DataFrame, window_size: int, threshold: float = 1.5) -> pd.DataFrame:
    """Remove the outliers in the curve using IQR

    Parameters
    ----------
    curve : pd.DataFrame
        The curve to remove outliers. Columns: x, y.
    window_size : int
        The window size for the median filter.
    threshold : float, default 1.5
        The threshold for the outlier.
        1.5 is the standard value for IQR.

    Returns
    -------
    pd.DataFrame
        The curve without outliers.

    """
    y = curve["y"].values
    rolling_median = curve["y"].rolling(window=window_size, min_periods=1, center=True).median()
    rolling_q1 = curve["y"].rolling(window=window_size, min_periods=1, center=True).quantile(0.25)
    rolling_q3 = curve["y"].rolling(window=window_size, min_periods=1, center=True).quantile(0.75)
    rolling_iqr = rolling_q3 - rolling_q1
    mask = (y < rolling_q1 - threshold * rolling_iqr) | (y > rolling_q3 + threshold * rolling_iqr)
    curve = curve[~mask]
    return curve
