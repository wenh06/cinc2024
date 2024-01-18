"""
"""
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch_ecg.cfg import CFG
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.utils import add_docstring

from outputs import CINC2024Outputs

__all__ = ["CRNN_CINC2024"]


class CRNN_CINC2024(ECG_CRNN):
    """ """

    __DEBUG__ = True
    __name__ = "CRNN_CINC2024"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        raise NotImplementedError

    def forward(
        self,
        waveforms: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def inference(self, waveforms: Union[np.ndarray, torch.Tensor]) -> CINC2024Outputs:
        raise NotImplementedError

    @add_docstring(inference.__doc__)
    def inference_CINC2024(
        self,
        waveforms: Union[np.ndarray, torch.Tensor],
        seg_thr: float = 0.5,
    ) -> CINC2024Outputs:
        """
        alias for `self.inference`
        """
        return self.inference(waveforms, seg_thr)
