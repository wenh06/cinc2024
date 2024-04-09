"""
OCR models for extracting text from ECG images, especially the lead names.
"""

from torch_ecg.utils.misc import CitationMixin
from torch_ecg.utils.utils_nn import SizeMixin


class OCRModel(CitationMixin, SizeMixin):
    """OCR model for extracting text from ECG images, especially the lead names.

    References
    ----------
    .. [1] to add
    """

    def __init__(self):
        raise NotImplementedError
