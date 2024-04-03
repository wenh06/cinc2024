"""
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "CINC2024Outputs",
]


@dataclass
class CINC2024Outputs:
    """Output class for CinC2024.

    Attributes
    ----------
    dx : Sequence[str]
        Predicted class names ("Normal", "Abnormal") of the Dx classification.
    dx_logits : Sequence[Sequence[float]]
        Logits of the Dx classification.
    dx_prob : Sequence[Sequence[float]]
        Probabilities of the Dx classification.
    dx_loss : Sequence[float]
        Loss for the Dx classification.
    dx_classes : Sequence[str]
        Class names for the Dx classification.
    digitization : Sequence[numpy.ndarray]
        Predicted digitization results.
    digitization_loss : Sequence[float]
        Loss for the digitization.
    total_loss : Sequence[float]
        Total loss, sum of the `dx_loss` and `digitization_loss`.

    """

    dx: Optional[Sequence[str]] = None
    dx_logits: Optional[Sequence[Sequence[float]]] = None
    dx_prob: Optional[Sequence[Sequence[float]]] = None
    dx_loss: Optional[Sequence[float]] = None
    dx_classes: Optional[Sequence[str]] = None
    digitization: Optional[Sequence[np.ndarray]] = None
    digitization_loss: Optional[Sequence[float]] = None
    total_loss: Optional[Sequence[float]] = None

    def __post_init__(self):
        assert any(
            [self.dx is not None, self.digitization is not None]
        ), "at least one of `dx`, `digitization` prediction should be provided"
        if self.dx is not None:
            assert self.dx_classes is not None, "dx_classes should be provided if `dx` is provided"
            idx2class = {idx: cl for idx, cl in enumerate(self.dx_classes)}
            if isinstance(self.dx, torch.Tensor):
                self.dx = self.dx.cpu().detach().numpy().tolist()
            self.dx = [idx2class[item] if isinstance(item, int) else item for item in self.dx]
        if self.dx_logits is not None:
            if isinstance(self.dx_logits, torch.Tensor):
                self.dx_logits = self.dx_logits.cpu().detach().numpy()
        if self.dx_prob is not None:
            if isinstance(self.dx_prob, torch.Tensor):
                self.dx_prob = self.dx_prob.cpu().detach().numpy()
        elif self.dx_logits is not None:
            self.dx_prob = F.softmax(torch.from_numpy(self.dx_logits), dim=-1).cpu().detach().numpy()
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
