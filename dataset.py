"""
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import ReprMixin

from data_reader import CINC2024Reader

__all__ = [
    "CinC2024Dataset",
]


class CinC2024Dataset(Dataset, ReprMixin):
    """Dataset for the CinC2024 Challenge.

    Parameters
    ----------
    config : CFG
        configuration for the dataset
    training : bool, default True
        whether the dataset is for training or validation
    lazy : bool, default True
        whether to load all data into memory at initialization
    reader_kwargs : dict, optional
        keyword arguments for the data reader class.

    """

    __name__ = "CinC2024Dataset"

    def __init__(
        self,
        config: CFG,
        training: bool = True,
        lazy: bool = True,
        **reader_kwargs,
    ) -> None:
        super().__init__()
        self.config = CFG(deepcopy(config))
        # self.task = task.lower()
        self.training = training
        self.lazy = lazy

        if self.config.get("db_dir", None) is None:
            self.config.db_dir = reader_kwargs.pop("db_dir", None)
            assert self.config.db_dir is not None, "db_dir must be specified"
        else:
            reader_kwargs.pop("db_dir", None)
        self.config.db_dir = Path(self.config.db_dir).expanduser().resolve()

        # updates reader_kwargs with the config
        for kw in ["fs", "working_dir"]:
            if kw not in reader_kwargs and hasattr(self.config, kw):
                reader_kwargs[kw] = getattr(self.config, kw)

        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        self.__cache = None

        self.reader = CINC2024Reader(**reader_kwargs)

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError
