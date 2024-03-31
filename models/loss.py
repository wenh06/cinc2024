"""
Digitization loss functions.

This module contains the loss functions for the digitized signal values.

- SNR loss: Signal-to-noise ratio loss.

- DTW loss: Dynamic time warping loss.

- MAE loss: Mean absolute error loss.

- RMSE loss: Root mean squared error loss.

- KS loss: Kolmogorov-Smirnov loss.

- ASCI loss: Adaptive signed correlation index loss.

"""

import re
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_ecg.utils import add_docstring, get_kwargs, remove_parameters_returns_from_docstring

__all__ = [
    "SNRLoss",
    "DTWLoss",
    "MAELoss",
    "RMSELoss",
    "KSLoss",
    "ASCILoss",
    "get_loss_func",
]


_LOSS_FUNCTIONS = {}


def _register_loss_fn(name: Optional[str] = None) -> Any:
    """Decorator to register a new loss function class.

    Parameters
    ----------
    name : str, optional
        Name of the loss function.
        If not specified, the class name will be used.

    Returns
    -------
    The decorated class.

    """

    def wrapper(cls_: Any) -> Any:
        if name is None:
            _name = cls_.__name__
        else:
            _name = name
        if _name in _LOSS_FUNCTIONS:
            raise ValueError(f"{_name} has already been registered")
        _LOSS_FUNCTIONS[_name] = cls_
        return cls_

    return wrapper


_snr_loss_docstring = """Signal-to-noise ratio loss.

    The signal-to-noise ratio (SNR) is defined as the ratio of the power of the signal to the power of the noise.
    The SNR loss is defined as the negative SNR.

    Parameters
    ----------
    inp : torch.Tensor
        Predicted signal values, of shape ``(batch_size, num_leads, sig_len)``.
    target : torch.Tensor
        Ground truth signal values, of shape ``(batch_size, num_leads, sig_len)``.
    mask : torch.Tensor, optional
        Mask tensor, of shape ``(batch_size, num_leads, sig_len)``.
        The mask tensor is used to mask out the padded values in the signal,
        so that the padded values do not contribute to the loss.
    eps : float, default 1e-7
        Small value to avoid division by zero.
    reduction : {"none", "mean", "sum"}, optional
        Specifies the reduction to apply to the output, by default "mean".

    Returns
    -------
    torch.Tensor
        SNR loss (negative SNR).

    .. note::

        `target` and `mask` should have the same shape.
        `inp` can have a different length (the last dimension) from `target` and `mask`.

    References
    ----------
    [#snr_wiki] https://en.wikipedia.org/wiki/Signal-to-noise_ratio

"""


@add_docstring(_snr_loss_docstring)
def snr_loss(
    inp: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-7, reduction: str = "mean"
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(target)
    mask = mask.bool().to(inp.device)
    target = target.to(inp.device)
    if target.shape[-1] > inp.shape[-1]:
        # pad the input signal
        inp = F.pad(inp, (0, target.shape[-1] - inp.shape[-1]), mode="constant", value=0)
    elif target.shape[-1] < inp.shape[-1]:
        # trim the input signal
        inp = inp[..., : target.shape[-1]]
    signal_power = torch.sum((inp**2) * mask, dim=-1)
    noise_power = torch.sum(((inp - target) ** 2) * mask, dim=-1) + eps
    snr = signal_power / noise_power
    neg_snr = -10 * torch.log10(snr)
    if reduction == "mean":
        return neg_snr.mean()
    elif reduction == "sum":
        return neg_snr.sum()
    return neg_snr


_dtw_loss_docstring = """Dynamic time warping loss.

    DTW is a method for measuring similarity between two temporal sequences that may vary in speed.

    Parameters
    ----------
    inp : torch.Tensor
        Predicted signal values, of shape ``(batch_size, num_leads, sig_len)``.
    target : torch.Tensor
        Ground truth signal values, of shape ``(batch_size, num_leads, sig_len)``.
    mask : torch.Tensor, optional
        Mask tensor, of shape ``(batch_size, num_leads, sig_len)``.
        The mask tensor is used to mask out the padded values in the signal,
        so that the padded values do not contribute to the loss.
    reduction : {"none", "mean", "sum"}, optional
        Specifies the reduction to apply to the output, by default "mean".

    Returns
    -------
    torch.Tensor
        DTW loss.

    References
    ----------
    [#dtw_wiki] https://en.wikipedia.org/wiki/Dynamic_time_warping

"""


@add_docstring(_dtw_loss_docstring)
def dtw_loss(
    inp: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, reduction: str = "mean"
) -> torch.Tensor:
    raise NotImplementedError


_mae_loss_docstring = """Mean absolute error loss.

    Parameters
    ----------
    inp : torch.Tensor
        Predicted signal values, of shape ``(batch_size, num_leads, sig_len)``.
    target : torch.Tensor
        Ground truth signal values, of shape ``(batch_size, num_leads, sig_len)``.
    mask : torch.Tensor, optional
        Mask tensor, of shape ``(batch_size, num_leads, sig_len)``.
        The mask tensor is used to mask out the padded values in the signal,
        so that the padded values do not contribute to the loss.
    reduction : {"none", "mean", "sum"}, optional
        Specifies the reduction to apply to the output, by default "mean".

    Returns
    -------
    torch.Tensor
        MAE loss.

    .. note::

        `target` and `mask` should have the same shape.
        `inp` can have a different length (the last dimension) from `target` and `mask`.

"""


@add_docstring(_mae_loss_docstring)
def mae_loss(
    inp: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, reduction: str = "mean"
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(target)
    mask = mask.bool().to(inp.device)
    target = target.to(inp.device)
    if target.shape[-1] > inp.shape[-1]:
        # pad the input signal
        inp = F.pad(inp, (0, target.shape[-1] - inp.shape[-1]), mode="constant", value=0)
    elif target.shape[-1] < inp.shape[-1]:
        # trim the input signal
        inp = inp[..., : target.shape[-1]]
    mae = torch.abs(inp - target) * mask
    mae = mae.sum(dim=-1) / mask.sum(dim=-1)  # DO NOT use mean() here, as the denominator may be zero
    if reduction == "mean":
        return mae.mean()
    elif reduction == "sum":
        return mae.sum()
    return mae


_rmse_loss_docstring = """Root mean squared error loss.

    Parameters
    ----------
    inp : torch.Tensor
        Predicted signal values, of shape ``(batch_size, num_leads, sig_len)``.
    target : torch.Tensor
        Ground truth signal values, of shape ``(batch_size, num_leads, sig_len)``.
    mask : torch.Tensor, optional
        Mask tensor, of shape ``(batch_size, num_leads, sig_len)``.
        The mask tensor is used to mask out the padded values in the signal,
        so that the padded values do not contribute to the loss.
    reduction : {"none", "mean", "sum"}, optional
        Specifies the reduction to apply to the output, by default "mean".

    Returns
    -------
    torch.Tensor
        RMSE loss.

    .. note::

        `target` and `mask` should have the same shape.
        `inp` can have a different length (the last dimension) from `target` and `mask`.

"""


@add_docstring(_rmse_loss_docstring)
def rmse_loss(
    inp: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, reduction: str = "mean"
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(target)
    mask = mask.bool().to(inp.device)
    target = target.to(inp.device)
    if target.shape[-1] > inp.shape[-1]:
        # pad the input signal
        inp = F.pad(inp, (0, target.shape[-1] - inp.shape[-1]), mode="constant", value=0)
    elif target.shape[-1] < inp.shape[-1]:
        # trim the input signal
        inp = inp[..., : target.shape[-1]]
    rmse = torch.sqrt(torch.mean(((inp - target) ** 2) * mask, dim=-1))
    if reduction == "mean":
        return rmse.mean()
    elif reduction == "sum":
        return rmse.sum()
    return rmse


_ks_loss_docstring = """Kolmogorov-Smirnov loss, inspired by the Kolmogorov-Smirnov test.

    KS test is a non-parametric test of the equality of continuous, one-dimensional probability distributions
    that can be used to test whether a sample came from a given reference probability distribution (one-sample KS test),
    or to test whether two samples came from the same distribution (two-sample KS test).

    Parameters
    ----------
    inp : torch.Tensor
        Predicted signal values, of shape ``(batch_size, num_leads, sig_len)``.
    target : torch.Tensor
        Ground truth signal values, of shape ``(batch_size, num_leads, sig_len)``.
    mask : torch.Tensor, optional
        Mask tensor, of shape ``(batch_size, num_leads, sig_len)``.
        The mask tensor is used to mask out the padded values in the signal,
        so that the padded values do not contribute to the loss.
    reduction : {"none", "mean", "sum"}, optional
        Specifies the reduction to apply to the output, by default "mean".

    Returns
    -------
    torch.Tensor
        KS loss.

    .. note::

        `target` and `mask` should have the same shape.
        `inp` can have a different length (the last dimension) from `target` and `mask`.

    References
    ----------
    [#ks_wiki] https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test

"""


@add_docstring(_ks_loss_docstring)
def ks_loss(
    inp: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, reduction: str = "mean"
) -> torch.Tensor:
    raise NotImplementedError


_asci_loss_docstring = """Adaptive signed correlation index loss.

    The Adaptive Signed Correlation Index (ASCI) is a measure to
    quantify the morphological similarity between signals.

    Parameters
    ----------
    inp : torch.Tensor
        Predicted signal values, of shape ``(batch_size, num_leads, sig_len)``.
    target : torch.Tensor
        Ground truth signal values, of shape ``(batch_size, num_leads, sig_len)``.
    mask : torch.Tensor, optional
        Mask tensor, of shape ``(batch_size, num_leads, sig_len)``.
        The mask tensor is used to mask out the padded values in the signal,
        so that the padded values do not contribute to the loss.
    beta : float, default 0.05
        Beta parameter for the ASCI loss, by default 0.05.
    reduction : {"none", "mean", "sum"}, optional
        Specifies the reduction to apply to the output, by default "mean".

    Returns
    -------
    torch.Tensor
        ASCI loss.

    .. note::

        `target` and `mask` should have the same shape.
        `inp` can have a different length (the last dimension) from `target` and `mask`.

    References
    ----------
    [#asci] https://www.sciencedirect.com/science/article/pii/S0165168409003119

"""


@add_docstring(_asci_loss_docstring)
def asci_loss(
    inp: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, beta: float = 0.05, reduction: str = "mean"
) -> torch.Tensor:
    raise NotImplementedError


@_register_loss_fn("snr")
@add_docstring(remove_parameters_returns_from_docstring(_snr_loss_docstring, parameters=["inp", "target", "mask"]))
class SNRLoss(nn.Module):

    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, inp: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the loss.

        Parameters
        ----------
        inp : torch.Tensor
            Predicted signal values.
        target : torch.Tensor
            Ground truth signal values.
        mask : torch.Tensor, optional
            Mask tensor.

        Returns
        -------
        torch.Tensor
            SNR loss.

        """
        return snr_loss(inp, target, mask, eps=self.eps, reduction=self.reduction)


@_register_loss_fn("dtw")
@add_docstring(remove_parameters_returns_from_docstring(_dtw_loss_docstring, parameters=["inp", "target", "mask"]))
class DTWLoss(nn.Module):

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        raise NotImplementedError

    def forward(self, inp: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the loss.

        Parameters
        ----------
        inp : torch.Tensor
            Predicted signal values.
        target : torch.Tensor
            Ground truth signal values.
        mask : torch.Tensor, optional
            Mask tensor.

        Returns
        -------
        torch.Tensor
            DTW loss.

        """
        pass


@_register_loss_fn("mae")
@add_docstring(remove_parameters_returns_from_docstring(_mae_loss_docstring, parameters=["inp", "target", "mask"]))
class MAELoss(nn.Module):

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, inp: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the loss.

        Parameters
        ----------
        inp : torch.Tensor
            Predicted signal values.
        target : torch.Tensor
            Ground truth signal values.
        mask : torch.Tensor, optional
            Mask tensor.

        Returns
        -------
        torch.Tensor
            MAE loss.

        """
        return mae_loss(inp, target, mask, reduction=self.reduction)


@_register_loss_fn("rmse")
@add_docstring(remove_parameters_returns_from_docstring(_rmse_loss_docstring, parameters=["inp", "target", "mask"]))
class RMSELoss(nn.Module):

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, inp: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the loss.

        Parameters
        ----------
        inp : torch.Tensor
            Predicted signal values.
        target : torch.Tensor
            Ground truth signal values.
        mask : torch.Tensor, optional
            Mask tensor.

        Returns
        -------
        torch.Tensor
            RMSE loss.

        """
        return rmse_loss(inp, target, mask, reduction=self.reduction)


@_register_loss_fn("ks")
@add_docstring(remove_parameters_returns_from_docstring(_asci_loss_docstring, parameters=["inp", "target", "mask"]))
class KSLoss(nn.Module):

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        raise NotImplementedError

    def forward(self, inp: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the loss.

        Parameters
        ----------
        inp : torch.Tensor
            Predicted signal values.
        target : torch.Tensor
            Ground truth signal values.
        mask : torch.Tensor, optional
            Mask tensor.

        Returns
        -------
        torch.Tensor
            KS loss.

        """
        pass


@_register_loss_fn("asci")
@add_docstring(remove_parameters_returns_from_docstring(_asci_loss_docstring, parameters=["inp", "target", "mask"]))
class ASCILoss(nn.Module):

    def __init__(self, beta: float = 0.05, reduction: str = "mean") -> None:
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        raise NotImplementedError

    def forward(self, inp: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the loss.

        Parameters
        ----------
        inp : torch.Tensor
            Predicted signal values.
        target : torch.Tensor
            Ground truth signal values.
        mask : torch.Tensor, optional
            Mask tensor.

        Returns
        -------
        torch.Tensor
            ASCI loss.

        """
        pass


def get_loss_func(name: str, **kwargs: Any) -> nn.Module:
    """Get the loss function.

    Parameters
    ----------
    name : str
        Name of the loss function, case-insensitive.
    **kwargs : dict
        Keyword arguments for the loss function.

    Returns
    -------
    nn.Module
        The loss function.

    """
    name = re.sub("(?:[\\_\\-])?loss", "", name.lower())
    if name.lower() in _LOSS_FUNCTIONS:
        # remove unwanted arguments
        if set(kwargs.keys()) - set(get_kwargs(_LOSS_FUNCTIONS[name.lower()])):
            print(
                f"Unexpected arguments for {name} loss: "
                f"{set(kwargs.keys()) - set(get_kwargs(_LOSS_FUNCTIONS[name.lower()]))} removed"
            )
        kwargs = {k: v for k, v in kwargs.items() if k in get_kwargs(_LOSS_FUNCTIONS[name.lower()])}
        return _LOSS_FUNCTIONS[name.lower()](**kwargs)
    raise ValueError(f"Unknown loss function: {name}")
