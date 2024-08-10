"""
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "CINC2024Outputs",
]


@dataclass
class CINC2024Outputs:
    """Output class for CinC2024.

    Attributes
    ----------
    dx : Sequence[Sequence[str]]
        Predicted class names of the Dx classification.
    dx_logits : Sequence[Sequence[float]]
        Logits of the Dx classification.
    dx_prob : Sequence[Sequence[float]]
        Probabilities of the Dx classification.
    dx_loss : Sequence[float]
        Loss for the Dx classification.
    dx_classes : Sequence[str]
        Class names for the Dx classification.
    dx_threshold : float
        Threshold for the Dx classification (multi-label).
        Default is 0.5.
    digitization : Sequence[numpy.ndarray]
        Predicted digitization results.
    digitization_loss : Sequence[float]
        Loss for the digitization.
    total_loss : Sequence[float]
        Total loss, sum of the `dx_loss` and `digitization_loss`.
    bbox : Sequence[Dict[str, np.ndarray]]
        Bounding boxes of the detected objects,
        keys are "bbox", "category_id", "category_name", "scores".
    bbox_loss : Sequence[Dict[str, float]], optional

    """

    dx: Optional[Sequence[Sequence[str]]] = None
    dx_logits: Optional[Sequence[Sequence[float]]] = None
    dx_prob: Optional[Sequence[Sequence[float]]] = None
    dx_loss: Optional[Sequence[float]] = None
    dx_classes: Optional[Sequence[str]] = None
    dx_threshold: float = 0.5
    digitization: Optional[Sequence[np.ndarray]] = None
    digitization_loss: Optional[Sequence[float]] = None
    total_loss: Optional[Sequence[float]] = None
    bbox: Optional[Sequence[Dict[str, np.ndarray]]] = None
    bbox_classes: Optional[Sequence[str]] = None
    bbox_loss: Optional[Sequence[Dict[str, float]]] = None

    def __post_init__(self) -> None:
        assert any(
            [
                self.dx is not None,
                self.dx_logits is not None,
                self.dx_prob is not None,
                self.digitization is not None,
                self.bbox is not None,
            ]
        ), "at least one of `dx`, `digitization`, `bbox` prediction should be provided"
        if self.dx is not None:
            assert self.dx_classes is not None, "dx_classes should be provided if `dx` is provided"
            idx2class = {idx: cl for idx, cl in enumerate(self.dx_classes)}
            # in case the dx is not converted to class names
            self.dx = [[idx2class.get(item, item) for item in items] for items in self.dx]
        if self.dx_prob is not None:
            assert self.dx_classes is not None, "dx_classes should be provided if `dx` is provided"
            if isinstance(self.dx_prob, torch.Tensor):
                self.dx_prob = self.dx_prob.cpu().detach().numpy()
            if self.dx is None:
                self.dx = [
                    [self.dx_classes[idx] for idx in np.where(np.array(items) > self.dx_threshold)[0]] for items in self.dx_prob
                ]
        elif self.dx_logits is not None:
            assert self.dx_classes is not None, "dx_classes should be provided if `dx` is provided"
            if isinstance(self.dx_logits, torch.Tensor):
                self.dx_logits = self.dx_logits.cpu().detach().numpy()
            self.dx_prob = torch.sigmoid(torch.from_numpy(self.dx_logits)).cpu().detach().numpy()
            if self.dx is None:
                self.dx = [
                    [self.dx_classes[idx] for idx in np.where(np.array(items) > self.dx_threshold)[0]] for items in self.dx_prob
                ]
        if self.dx_loss is not None:
            if isinstance(self.dx_loss, torch.Tensor):
                self.dx_loss = self.dx_loss.cpu().detach().numpy()
        if self.digitization is not None:
            if isinstance(self.digitization, torch.Tensor):
                self.digitization = self.digitization.cpu().detach().numpy()
        if self.digitization_loss is not None:
            if isinstance(self.digitization_loss, torch.Tensor):
                self.digitization_loss = self.digitization_loss.cpu().detach().numpy()

        if self.total_loss is None:
            if self.dx_loss is not None and self.digitization_loss is not None:
                self.total_loss = np.array(
                    [dx_loss + digitization_loss for dx_loss, digitization_loss in zip(self.dx_loss, self.digitization_loss)]
                )
            elif self.dx_loss is not None:
                self.total_loss = self.dx_loss
            elif self.digitization_loss is not None:
                self.total_loss = self.digitization_loss

        # TODO: further process the bbox:
        # the bounding boxes include the lead names boxes and the waveform boxes
        # one should assign the lead names to the detected waveform bounding boxes
        # based on spatial relations of the waveform boxes and the lead names boxes
        if self.bbox is not None:
            for b_dict in self.bbox:
                if "boxes" in b_dict:
                    b_dict["bbox"] = b_dict.pop("boxes")
                # extract ROI from the bounding boxes
                bbox_arr = np.array(b_dict["bbox"])
                b_dict["roi"] = [
                    bbox_arr[..., 0].min().astype(int).item(),
                    bbox_arr[..., 1].min().astype(int).item(),
                    bbox_arr[..., 2].max().astype(int).item(),
                    bbox_arr[..., 3].max().astype(int).item(),
                ]
