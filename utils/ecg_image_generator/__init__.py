"""
The main interface for the ECG image generator.
"""

from . import constants
from ._ecg_plot import ecg_plot
from .CreasesWrinkles import get_creased
from .extract_leads import get_paper_ecg
from .gen_ecg_image_from_data import run_single_file
from .gen_ecg_images_from_data_batch import run as run_batch
from .HandwrittenText import download_en_core_sci_sm, en_core_sci_sm_model, get_handwritten
from .ImageAugmentation import get_augment
from .TemplateFiles import get_template

__all__ = [
    "en_core_sci_sm_model",
    "get_creased",
    "get_handwritten",
    "download_en_core_sci_sm",
    "get_augment",
    "get_template",
    "ecg_plot",
    "get_paper_ecg",
    "run_single_file",
    "run_batch",
    "constants",
]
