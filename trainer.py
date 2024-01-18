"""
"""

from typing import Any, Optional

import torch
from torch import nn
from torch_ecg.components.trainer import BaseTrainer

from dataset import CinC2024Dataset

__all__ = [
    "CINC2024Trainer",
]


class CINC2024Trainer(BaseTrainer):
    """Trainer for the CinC2024 challenge.

    Parameters
    ----------
    model : torch.nnModule
        the model to be trained
    model_config : dict
        the configuration of the model,
        used to keep a record in the checkpoints
    train_config : dict
        the configuration of the training,
        including configurations for the data loader, for the optimization, etc.
        will also be recorded in the checkpoints.
        `train_config` should at least contain the following keys:

            - "monitor": obj:`str`,
            - "loss": obj:`str`,
            - "n_epochs": obj:`int`,
            - "batch_size": obj:`int`,
            - "learning_rate": obj:`float`,
            - "lr_scheduler": obj:`str`,
            - "lr_step_size": obj:`int`, optional, depending on the scheduler
            - "lr_gamma": obj:`float`, optional, depending on the scheduler
            - "max_lr": obj:`float`, optional, depending on the scheduler
            - "optimizer": obj:`str`,
            - "decay": obj:`float`, optional, depending on the optimizer
            - "momentum": obj:`float`, optional, depending on the optimizer

    device : torch.device, optional
        the device to be used for training
    lazy : bool, default True
        whether to initialize the data loader lazily

    """

    __DEBUG__ = True
    __name__ = "CINC2024Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            dataset_cls=CinC2024Dataset,
            model_config=model_config,
            train_config=train_config,
            device=device,
            lazy=lazy,
        )
        raise NotImplementedError
