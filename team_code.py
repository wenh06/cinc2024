#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from deprecated import deprecated
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch_ecg.cfg import CFG
from torch_ecg.utils.download import url_is_reachable
from torch_ecg.utils.misc import str2bool

from cfg import BaseCfg, ModelCfg, TrainCfg
from const import DATA_CACHE_DIR, MODEL_CACHE_DIR, REMOTE_HEADS_URLS, REMOTE_MODELS
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
    load_images,
    load_labels,
    load_text,
)
from models import ECGWaveformDetector, MultiHead_CINC2024
from trainer import CINC2024Trainer
from utils.ecg_simulator import evolve_standard_12_lead_ecg

################################################################################
# environment variables

os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)

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

# choices of the backbone models:
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
# NOTE: DO NOT change the backbone_name and backbone_source here
# change them in cfg.py
# as one has to run post_docker_build.py to cache the pretrained models
# in which ModelCfg is imported from cfg.py, not from this script
# ModelCfg.backbone_name = "facebook/convnextv2-large-22k-384"
# ModelCfg.backbone_source = "hf"

SubmissionCfg = CFG()
SubmissionCfg.detector = None  # "hf--facebook/detr-resnet-50"
SubmissionCfg.classifier = "hf--facebook/convnextv2-nano-22k-384"
SubmissionCfg.digitizer = None

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
    data_folder: Union[str, bytes, os.PathLike], model_folder: Union[str, bytes, os.PathLike], verbose: bool
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
        print("Training the digitization model...")
        print("Finding the Challenge data...")

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError("No data was provided.")

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    (Path(model_folder) / "working_dir").mkdir(parents=True, exist_ok=True)

    reader_kwargs = {
        "db_dir": Path(DATA_CACHE_DIR),
        "working_dir": (Path(model_folder) / "working_dir"),
        "synthetic_images_dir": Path(DATA_CACHE_DIR) / "synthetic_images",
    }

    # Download the synthetic images
    # dr = CINC2024Reader(**reader_kwargs)
    # dr.download_synthetic_images(set_name="subset")  # "full" is too large, not uploaded to any cloud storage
    # del dr

    # Train the model
    train_config = deepcopy(TrainCfg)
    # train_config.db_dir = data_folder
    train_config.db_dir = Path(DATA_CACHE_DIR)
    train_config.synthetic_images_dir = Path(DATA_CACHE_DIR) / "synthetic_images"
    # train_config.model_dir = model_folder
    train_config.working_dir = Path(model_folder) / "working_dir"
    if TEST_FLAG:
        # train_config.debug = True
        train_config.debug = False

        train_config.n_epochs = 1
        train_config.batch_size = 4  # 16G (Tesla T4)
        train_config.log_step = 5
        # # train_config.max_lr = 1.5e-3
        # train_config.early_stopping.patience = 20
    else:
        train_config.debug = False

        train_config.n_epochs = 27
        train_config.batch_size = 48  # 16G (Tesla T4)
        train_config.log_step = 100
        # train_config.max_lr = 1.5e-3
        train_config.early_stopping.patience = train_config.n_epochs // 3

    model_config = deepcopy(ModelCfg)
    # model_config.backbone_name = "facebook/convnextv2-atto-1k-224"

    model = MultiHead_CINC2024(config=model_config)
    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)
    model = model.to(device=DEVICE)
    if verbose:
        if isinstance(model, DP):
            print("model size:", model.module.module_size, model.module.module_size_)
        else:
            print("model size:", model.module_size, model.module_size_)

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

    # NOTE that this training process only ensures
    # that the training pipeline is correct and the model can be trained
    # the model used in Challenge evaluation will be loaded from the remote heads

    best_state_dict = trainer.train()  # including saving model

    del trainer
    del model
    del best_state_dict

    torch.cuda.empty_cache()

    if verbose:
        print(f"""Saved models: {list((Path(__file__).parent / "saved_models").iterdir())}""")

    print("\n" + "*" * 100)
    msg = "   CinC2024 challenge training entry ends   ".center(100, "#")
    print(msg)


def load_models(
    model_folder: Union[str, bytes, os.PathLike], verbose: bool
) -> Tuple[Dict[str, nn.Module], Dict[str, nn.Module]]:
    """Load the trained models.

    Parameters
    ----------
    model_folder : `path_like`
        The path to the folder containing the trained model.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    digitization_model : Dict[str, nn.Module]
        The trained digitization models.
    classification_model : Dict[str, nn.Module]
        The trained classification models.

    """
    if SubmissionCfg.digitalizer is not None:
        raise NotImplementedError("Digitalizer is not implemented yet.")
    else:
        digitization_model = None

    classification_model = {}
    if SubmissionCfg.detector is not None:
        detector, _ = ECGWaveformDetector.from_checkpoint(
            Path(MODEL_CACHE_DIR) / REMOTE_MODELS[SubmissionCfg.detector]["filename"]
        )
        classification_model["detector"] = detector

    if SubmissionCfg.classifier is not None:
        classifier, _ = MultiHead_CINC2024.from_checkpoint(
            Path(MODEL_CACHE_DIR) / REMOTE_MODELS[SubmissionCfg.classifier]["filename"]
        )
        classification_model["classifier"] = classifier

    return digitization_model, classification_model


@deprecated(reason="Use `load_models` instead.", action="error")
def load_models_bak(
    model_folder: Union[str, bytes, os.PathLike], verbose: bool
) -> Tuple[Dict[str, nn.Module], Dict[str, nn.Module]]:
    """Load the trained models.

    Legacy function for the unofficial phase.

    Parameters
    ----------
    model_folder : `path_like`
        The path to the folder containing the trained model.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    digitization_model : Dict[str, nn.Module]
        The trained digitization models.
    classification_model : Dict[str, nn.Module]
        The trained classification models.

    """
    key = f"{ModelCfg.backbone_source}--{ModelCfg.backbone_name}"
    if url_is_reachable("https://www.dropbox.com/"):
        remote_heads_url = REMOTE_HEADS_URLS[key]["dropbox"]
    else:
        remote_heads_url = REMOTE_HEADS_URLS[key]["deep-psp"]
    # model_dir = Path(model_folder).resolve() / MODEL_DIR / key.replace("/", "--")
    model_dir = Path(MODEL_CACHE_DIR) / key.replace("/", "--")
    model = MultiHead_CINC2024.from_remote_heads(
        url=remote_heads_url,
        model_dir=model_dir,
        device=DEVICE,
    )
    return model


# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(
    record: Union[str, bytes, os.PathLike],
    digitization_model: Dict[str, nn.Module],
    classification_model: Dict[str, nn.Module],
    verbose: bool,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Run the digitization model on a record.

    Parameters
    ----------
    digitization_model : MultiHead_CINC2024
        The trained digitization model.
    record : Union[str, bytes, os.PathLike]
        The path to the record to process.
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

    if classification_model.get("detector") is not None:
        detector = classification_model["detector"]
        bbox = detector.inference(input_images).bbox  # a list of dict
        # crop the input images using the "roi" of each dict in the bbox
        # the "roi" is a list of 4 integers [xmin, ymin, xmax, ymax]
        cropped_images = [img.crop(b_dict["roi"]) for img, b_dict in zip(input_images, bbox)]
    else:
        cropped_images = input_images

    if classification_model.get("classifier") is not None:
        classifier = classification_model["classifier"]
        output = classifier.inference(cropped_images, threshold=REMOTE_MODELS[SubmissionCfg.classifier]["threshold"])
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

    if digitization_model is not None:
        raise NotImplementedError("Digitalizer is not implemented yet.")
    else:
        use_workaround = False  # True or False

        if use_workaround:
            # workaround for Dx prediction only by returning a random signal (or nan values),
            # to avoid the FileNotFoundError in the evaluation script.
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

            signal = np.asarray(signal, dtype=np.int16)
        else:
            signal = None

    return signal, labels


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# NOT used currently.
