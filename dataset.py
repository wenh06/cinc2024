"""
"""

from typing import Dict

import numpy as np
from torch.utils.data.dataset import Dataset
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import ReprMixin

__all__ = [
    "CinC2024Dataset",
]


class CinC2024Dataset(Dataset, ReprMixin):
    """Dataset for the CinC2024 Challenge."""

    __name__ = "CinC2024Dataset"

    def __init__(
        self,
        config: CFG,
        training: bool = True,
        lazy: bool = True,
        **reader_kwargs,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError
