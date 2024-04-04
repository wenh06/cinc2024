"""
"""

import argparse
import os
import sys
import textwrap
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch.utils.data import DataLoader, Dataset
from torch_ecg.cfg import CFG
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.utils.misc import str2bool
from tqdm.auto import tqdm

from cfg import ModelCfg, TrainCfg
from dataset import CinC2024Dataset, collate_fn
from models import MultiHead_CINC2024
from utils.scoring_metrics import compute_challenge_metrics

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

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
        """
        setup the dataloaders for training and validation

        Parameters
        ----------
        train_dataset: Dataset, optional,
            the training dataset
        val_dataset: Dataset, optional,
            the validation dataset

        """
        if train_dataset is None:
            train_dataset = self.dataset_cls(
                config=self.train_config,
                training=True,
                lazy=True,
            )

        if self.train_config.debug:
            val_train_dataset = train_dataset
        else:
            val_train_dataset = None
        if val_dataset is None:
            val_dataset = self.dataset_cls(
                config=self.train_config,
                training=False,
                lazy=True,
            )

        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
        if self.device == torch.device("cpu"):
            num_workers = 1
        else:
            num_workers = 4

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        if self.train_config.debug:
            self.val_train_loader = DataLoader(
                dataset=val_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            self.val_train_loader = None
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def train_one_epoch(self, pbar: tqdm) -> None:
        """Train one epoch, and update the progress bar

        Parameters
        ----------
        pbar : tqdm
            the progress bar for training

        """
        for epoch_step, input_tensors in enumerate(self.train_loader):
            input_tensors["image"] = self._model.get_input_tensors(input_tensors["image"])
            self.global_step += 1
            n_samples = input_tensors["image"].shape[self.batch_dim]

            out_tensors = self.run_one_step(input_tensors)

            # NOTE: loss is computed in the model, and kept in the out_tensors
            loss = out_tensors["total_loss"]

            if self.train_config.flooding_level > 0:
                flood = (loss - self.train_config.flooding_level).abs() + self.train_config.flooding_level
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                flood.backward()
            else:
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
            self.optimizer.step()
            self._update_lr()

            if self.global_step % self.train_config.log_step == 0:
                train_step_metrics = {"loss": loss.item()}
                if self.scheduler:
                    train_step_metrics.update({"lr": self.scheduler.get_last_lr()[0]})
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                            "lr": self.scheduler.get_last_lr()[0],
                        }
                    )
                else:
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                        }
                    )
                if self.train_config.flooding_level > 0:
                    train_step_metrics.update({"flood": flood.item()})
                self.log_manager.log_metrics(
                    metrics=train_step_metrics,
                    step=self.global_step,
                    epoch=self.epoch,
                    part="train",
                )
            pbar.update(n_samples)

    def run_one_step(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run one step (batch) of training

        Parameters
        ----------
        input_tensors : dict
            the tensors to be processed for training one step (batch), with the following items:
                - "image" (required): the input waveforms
                - "dx" (optional): the Dx classification labels
                - "digitization" (optional): the signal reconstruction labels
                - "mask" (optional): the mask for the signal reconstruction

        Returns
        -------
        out_tensors : dict
            with the following items (some are optional):
            - "dx": the Dx classification predictions, of shape ``(batch_size, n_classes)``.
            - "dx_logits": the Dx classification logits, of shape ``(batch_size, n_classes)``.
            - "dx_loss": the Dx classification loss
            - "digitization": the signal reconstruction predictions, of shape ``(batch_size, n_leads, n_samples)``.
            - "digitization_loss": the signal reconstruction loss
            - "total_loss": the total loss for the training step

        """
        image = input_tensors.pop("image")
        out_tensors = self.model(image, input_tensors)
        return out_tensors

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on the given data loader"""

        self.model.eval()

        all_outputs = []
        all_labels = []

        with tqdm(
            total=len(data_loader.dataset),
            desc="Evaluation",
            unit="signals",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
        ) as pbar:
            for input_tensors in data_loader:
                # input_tensors is assumed to be a dict of tensors, with the following items:
                # "image" (required): the input image list
                # "dx" (optional): the Dx classification labels
                # "digitization" (optional): the signal reconstruction labels
                # "mask" (optional): the mask for the signal reconstruction
                # image = self._model.get_input_tensors(input_tensors.pop("image"))
                image = input_tensors.pop("image")
                labels = {k: v.numpy() for k, v in input_tensors.items() if v is not None}
                if "dx" in labels:
                    # convert numeric labels to string labels
                    labels["dx"] = np.array([self._model.config.dx_head.classes[i] for i in labels["dx"]])

                all_labels.append(labels)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                all_outputs.append(self._model.inference(image))  # of type CINC2024Outputs
                pbar.update(len(image))

        if self.val_train_loader is not None and self.train_config.predict_dx:
            log_head_num = 5
            head_scalar_preds = all_outputs[0].dx_prob[:log_head_num]
            head_preds_classes = all_outputs[0].dx[:log_head_num]
            head_labels_classes = all_labels[0]["dx"][:log_head_num]
            log_head_num = min(log_head_num, len(head_scalar_preds))
            for n in range(log_head_num):
                msg = textwrap.dedent(
                    f"""
                ----------------------------------------------
                Dx scalar prediction:    {[round(item, 3) for item in head_scalar_preds[n].tolist()]}
                Dx predicted classes:    {head_preds_classes[n]}
                Dx label classes:        {head_labels_classes[n]}
                ----------------------------------------------
                """
                )
                self.log_manager.log_message(msg)

        metrics_keeps = []
        if self.train_config.predict_dx:
            metrics_keeps.append("dx")
        if self.train_config.predict_digitization:
            metrics_keeps.append("digitization")
        eval_res = compute_challenge_metrics(labels=all_labels, outputs=all_outputs, keeps=metrics_keeps)

        # in case possible memeory leakage?
        del all_labels
        del all_outputs

        self.model.train()

        return eval_res

    @property
    def batch_dim(self) -> int:
        """
        batch dimension, usually 0,
        but can be 1 for some models, e.g. RR_LSTM
        """
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        return ["predict_dx", "predict_digitization"]

    @property
    def save_prefix(self) -> str:
        prefix = f"""{self.model_config.backbone_source}-{self.model_config.backbone_name.replace("/", "-")}"""
        if self.model_config.dx_head.include:
            prefix = f"{prefix}-dx"
        if self.model_config.digitization_head.include:
            prefix = f"{prefix}-digitization"
        if self.model_config.backbone_freeze:
            prefix = f"{prefix}-headonly"
        return prefix + "_"

    def extra_log_suffix(self) -> str:
        suffix = f"""{self.model_config.backbone_source}-{self.model_config.backbone_name.replace("/", "-")}"""
        if self.model_config.dx_head.include:
            suffix = f"{suffix}-dx"
        if self.model_config.digitization_head.include:
            suffix = f"{suffix}-digitization"
        if self.model_config.backbone_freeze:
            suffix = f"{suffix}-headonly"
        suffix = f"{suffix}-{super().extra_log_suffix()}"
        return suffix

    def _setup_criterion(self) -> None:
        # since criterion is defined in the model,
        # override this method to do nothing
        pass

    def save_checkpoint(self, path: str) -> None:
        """Save the current state of the trainer to a checkpoint.

        Parameters
        ----------
        path : str
            Path to save the checkpoint

        """
        if not self.model_config.backbone_freeze:
            super().save_checkpoint(path)
            return
        # if the backbone is frozen, save only the state_dict of the head(s)
        checkpoint = {
            "model_config": self.model_config,
            "train_config": self.train_config,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
        }
        if self.model_config.dx_head.include:
            checkpoint.update({"dx_head_state_dict": self._model.dx_head.state_dict()})
        if self.model_config.digitization_head.include:
            checkpoint.update({"digitization_head_state_dict": self._model.digitization_head.state_dict()})
        torch.save(checkpoint, path)


def get_args(**kwargs: Any):
    """NOT checked,"""
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train the Model on CINC2024 database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=24,
        help="the batch size for training",
        dest="batch_size",
    )
    parser.add_argument(
        "--keep-checkpoint-max",
        type=int,
        default=10,
        help="maximum number of checkpoints to keep. If set 0, all checkpoints will be kept",
        dest="keep_checkpoint_max",
    )
    # parser.add_argument(
    #     "--optimizer", type=str, default="adam",
    #     help="training optimizer",
    #     dest="train_optimizer")
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="train with more debugging information",
        dest="debug",
    )

    args = vars(parser.parse_args())

    cfg.update(args)

    return CFG(cfg)


if __name__ == "__main__":
    # WARNING: most training were done in notebook,
    # NOT in cli
    train_config = get_args(**TrainCfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: adjust for CINC2024
    model_config = deepcopy(ModelCfg)
    # adjust the model configuration if necessary
    model = MultiHead_CINC2024(config=model_config)

    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)
    model.to(device=device)

    trainer = CINC2024Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
        lazy=True,
    )

    try:
        best_model_state_dict = trainer.train()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
