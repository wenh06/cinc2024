"""
Waveform detector model, which detects the bounding boxes of the waveforms in the ECG images.
"""

from torch_ecg.utils.misc import CitationMixin
from torch_ecg.utils.utils_nn import SizeMixin


class ECGWaveformDetector(CitationMixin, SizeMixin):
    """Waveform detector model, which detects the bounding boxes of the waveforms in the ECG images.

    References
    ----------
    .. [1] to add
    """

    def __init__(self):
        raise NotImplementedError
