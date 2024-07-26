"""
The main interface for the ECG image generator.
"""

from . import constants
from ._ecg_plot import ecg_plot
from .CreasesWrinkles import get_creased
from .extract_leads import get_paper_ecg
from .gen_ecg_image_from_data import run_single_file
from .gen_ecg_images_from_data_batch import run as run_batch
from .HandwrittenText import download_en_core_sci_sm, en_core_sci_sm_model_dir, get_handwritten, load_en_core_sci_sm
from .ImageAugmentation import get_augment
from .TemplateFiles import get_template

__all__ = [
    "en_core_sci_sm_model_dir",
    "load_en_core_sci_sm",
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


def report_extra_libs():
    not_installed = []
    try:
        from imgaug import augmenters as iaa  # noqa: F401
    except ImportError:
        not_installed.append("imgaug")
    try:
        import cv2  # noqa: F401
    except ImportError:
        not_installed.append("opencv-python")

    if not_installed:
        import warnings

        not_installed = ", ".join(not_installed)
        warnings.warn(f"The following libraries are not installed, but are required for some functions: {not_installed}")


report_extra_libs()
