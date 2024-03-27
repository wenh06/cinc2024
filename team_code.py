#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import os
from typing import Any, Union

import numpy as np

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

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################


# Train your digitization model.
def train_digitization_model(
    data_folder: Union[str, bytes, os.PathLike], model_folder: Union[str, bytes, os.PathLike], verbose: bool
) -> None:
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

    raise NotImplementedError("The digitization model is not implemented.")


# Train your dx model.
def train_dx_model(
    data_folder: Union[str, bytes, os.PathLike], model_folder: Union[str, bytes, os.PathLike], verbose: bool
) -> None:
    # This function will not be used,
    # since the dx classification model and the digitization model will be trained simultaneously,
    # in which case they share a same backbone, with different heads.
    pass


# Load your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a digitization model, then you can return None.
def load_digitization_model(model_folder: Union[str, bytes, os.PathLike], verbose: bool) -> Any:
    raise NotImplementedError("The digitization model is not implemented.")


# Load your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function. If you do not train a dx classification model, then you can return None.
def load_dx_model(model_folder: Union[str, bytes, os.PathLike], verbose: bool) -> Any:
    # this function will do the same thing as load_digitization_model
    # both heads are included in the same model
    return load_digitization_model(model_folder, verbose)


# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.
def run_digitization_model(digitization_model: Any, record: Union[str, bytes, os.PathLike], verbose: bool) -> np.ndarray:
    raise NotImplementedError("The digitization model is not implemented.")


# Run your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function.
def run_dx_model(dx_model: Any, record: Union[str, bytes, os.PathLike], signal: np.ndarray, verbose: bool) -> list[str]:
    raise NotImplementedError("The dx classification model is not implemented.")


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# NOT used currently.
