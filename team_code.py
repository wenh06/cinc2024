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
from typing import Union

import numpy as np
import torch
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch_ecg.utils.misc import str2bool

from cfg import ModelCfg, TrainCfg
from const import REMOTE_HEADS_URLS
from data_reader import CINC2024Reader
from dataset import CinC2024Dataset
from helper_code import (  # noqa: F401
    compute_one_hot_encoding,
    find_records,
    get_header_file,
    get_num_samples,
    get_num_signals,
    load_dx,
    load_image,
    load_text,
)
from models import MultiHead_CINC2024
from trainer import CINC2024Trainer
from utils.misc import url_is_reachable

################################################################################
# environment variables

try:
    TEST_FLAG = os.environ.get("CINC2024_REVENGER_TEST", False)
    TEST_FLAG = str2bool(TEST_FLAG)
except Exception:
    TEST_FLAG = False

MODEL_DIR = "revenger_model_dir"
SYNTHETIC_IMAGE_DIR = "revenger_synthetic_image_dir"
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
ModelCfg.backbone_name = "facebook/convnextv2-large-22k-384"
ModelCfg.backbone_source = "hf"

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


# Train your digitization model.
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
    print("\n" + "*" * 100)
    msg = "   CinC2023 challenge training entry starts   ".center(100, "#")
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

    # Create the MODEL_DIR and SYNTHETIC_IMAGE_DIR subfolders
    (Path(model_folder) / MODEL_DIR).mkdir(parents=True, exist_ok=True)
    (Path(model_folder) / SYNTHETIC_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    (Path(model_folder) / "working_dir").mkdir(parents=True, exist_ok=True)

    reader_kwargs = {
        "db_dir": data_folder,
        "working_dir": (Path(model_folder) / "working_dir"),
        "synthetic_images_dir": (Path(model_folder) / SYNTHETIC_IMAGE_DIR),
    }

    # Download the synthetic images
    dr = CINC2024Reader(**reader_kwargs)
    dr.download_synthetic_images(set_name="subset")  # "full" is too large, not uploaded to any cloud storage
    del dr

    # Train the model
    train_config = deepcopy(TrainCfg)
    train_config.db_dir = data_folder
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

        train_config.n_epochs = 25
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
    msg = "   CinC2023 challenge training entry ends   ".center(100, "#")
    print(msg)


# Train your dx model.
def train_dx_model(
    data_folder: Union[str, bytes, os.PathLike], model_folder: Union[str, bytes, os.PathLike], verbose: bool
) -> None:
    """Train the dx model.

    This function will not be used,
    since the dx classification model and the digitization model will be trained simultaneously,
    in which case they share a same backbone, with different heads.

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
    msg = "   Dx model is trained simultaneously with the digitization model   ".center(100, "#")
    print(msg)
    print("*" * 100 + "\n")


# Load your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a digitization model, then you can return None.
def load_digitization_model(model_folder: Union[str, bytes, os.PathLike], verbose: bool) -> MultiHead_CINC2024:
    """Load the trained digitization model.

    Parameters
    ----------
    model_folder : `path_like`
        The path to the folder containing the trained model.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    MultiHead_CINC2024
        The trained digitization model.

    """
    if url_is_reachable("https://www.dropbox.com/"):
        remote_heads_url = REMOTE_HEADS_URLS[f"{ModelCfg.backbone_source}--{ModelCfg.backbone_name}"]["dropbox"]
    else:
        remote_heads_url = REMOTE_HEADS_URLS[f"{ModelCfg.backbone_source}--{ModelCfg.backbone_name}"]["deep-psp"]
    model = MultiHead_CINC2024.from_remote_heads(
        url=remote_heads_url,
        model_dir=model_folder,
        device=DEVICE,
    )
    return model


# Load your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function. If you do not train a dx classification model, then you can return None.
def load_dx_model(model_folder: Union[str, bytes, os.PathLike], verbose: bool) -> MultiHead_CINC2024:
    """Load the trained dx classification model.

    This function will do the same thing as load_digitization_model,
    where both heads are included in the same model

    Parameters
    ----------
    model_folder : `path_like`
        The path to the folder containing the trained model.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    MultiHead_CINC2024
        The trained dx classification model.

    """
    return load_digitization_model(model_folder, verbose)


# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.
def run_digitization_model(
    digitization_model: MultiHead_CINC2024, record: Union[str, bytes, os.PathLike], verbose: bool
) -> np.ndarray:
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

    """
    # Load the dimensions of the signal.
    # header_file = get_header_file(record)
    # header = load_text(header_file)

    # num_samples = get_num_samples(header)
    # num_signals = get_num_signals(header)

    input_images = load_image(record)  # a list of PIL.Image.Image
    # convert to RGB (it's possible that the images are RGBA format)
    input_images = [img.convert("RGB") for img in input_images]
    output = digitization_model.inference(input_images)  # of type CINC2024Outputs
    return output.digitization


# Run your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function.
def run_dx_model(
    dx_model: MultiHead_CINC2024, record: Union[str, bytes, os.PathLike], signal: np.ndarray, verbose: bool
) -> list[str]:
    """Run the dx classification model on a record.

    Parameters
    ----------
    dx_model : MultiHead_CINC2024
        The trained dx classification model.
    record : Union[str, bytes, os.PathLike]
        The path to the record to process.
    signal : numpy.ndarray
        The digitized signal.
    verbose : bool
        Whether to display progress information.

    Returns
    -------
    list[str]
        The predicted labels.

    """
    input_images = load_image(record)  # a list of PIL.Image.Image
    # convert to RGB (it's possible that the images are RGBA format)
    input_images = [img.convert("RGB") for img in input_images]
    output = dx_model.inference(input_images)  # of type CINC2024Outputs
    return output.dx


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# NOT used currently.
