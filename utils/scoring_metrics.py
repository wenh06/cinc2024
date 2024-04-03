from typing import Dict, Sequence

import numpy as np
from torch_ecg.utils.utils_metrics import cls_to_bin  # noqa: F401

from helper_code import compute_f_measure
from outputs import CINC2024Outputs

__all__ = [
    "compute_challenge_metrics",
]


def compute_challenge_metrics(
    labels: Sequence[Dict[str, np.ndarray]],
    outputs: Sequence[CINC2024Outputs],
) -> Dict[str, float]:
    """Compute the challenge metrics.

    Parameters
    ----------
    labels : Sequence[Dict[str, np.ndarray]]
        The labels for the records.
    outputs : Sequence[CINC2024Outputs]
        The outputs for the records.

    Returns
    -------
    Dict[str, float]
        The computed challenge metrics for "dx" and "digitization" (at least one of them).
        nan values are returned for the metrics that are not computed due to missing outputs.

    """
    metrics = {f"dx_{metric}": value for metric, value in compute_dx_metrics(labels, outputs).items()}
    metrics.update({f"digitization_{metric}": value for metric, value in compute_digitization_metrics(labels, outputs).items()})
    return metrics


def compute_dx_metrics(
    labels: Sequence[Dict[str, np.ndarray]],
    outputs: Sequence[CINC2024Outputs],
) -> Dict[str, float]:
    """Compute the metrics for the "dx" task.

    Parameters
    ----------
    labels : Sequence[Dict[str, np.ndarray]]
        The labels for the records, containing the "dx" field.
        The "dx" field is a 1D numpy array of shape `(num_samples,)` with binary values,
        or a 2D numpy array of shape `(num_samples, num_classes)` with probabilities (0 or 1).
    outputs : Sequence[CINC2024Outputs]
        The outputs for the records, containing the "dx" field.
        The "dx" field is a 1D numpy array of shape `(num_samples,)` with binary values,
        or a 2D numpy array of shape `(num_samples, num_samples)` with probabilities (0 to 1).

    Returns
    -------
    Dict[str, float]
        The computed challenge metrics for the "dx" task.

    Examples
    --------
    >>> labels = [{"dx": ["Abnormal", "Normal", "Normal"]}, {"dx": ["Normal", "Normal"]}]
    >>> outputs = [
            CINC2024Outputs(dx=["Abnormal", "Normal", "Abnormal"], dx_classes=["Abnormal", "Normal"]),
            CINC2024Outputs(dx=["Abnormal", "Normal"], dx_classes=["Abnormal", "Normal"])
        ]
    >>> compute_dx_metrics(labels, outputs)
    {'f_measure': 0.5833333333333333}

    >>> from torch_ecg.components.metrics import ClassificationMetrics
    >>> cm = ClassificationMetrics(multi_label=False)
    >>> cm(labels=np.array([0, 1, 1, 1, 1]), outputs=np.array([0, 1, 0, 0, 1]), num_classes=2)
    >>> assert cm.f1_measure == compute_dx_metrics(labels, outputs)["f_measure"]
    True

    """
    assert len(labels) == len(outputs), "The number of labels and outputs should be the same"
    if not all([item.dx is not None for item in outputs]):
        return {"f_measure": np.nan}
    assert all(
        [len(label["dx"]) == len(output.dx) for label, output in zip(labels, outputs)]
    ), "The number of 'dx' labels and outputs should be the same"
    # concatenate the labels and outputs
    labels = np.concatenate([label["dx"] for label in labels], axis=0)
    outputs = np.concatenate([output.dx for output in outputs], axis=0)
    # convert the labels to binarized form (one-hot encoding) if they are not
    # labels, outputs = cls_to_bin(labels, outputs, num_classes=len(dx.dx_classes))
    # compute_f_measure accepts categorical labels and outputs,
    # (both in the form of sequence of sequences of a single categorical value)
    # and convert them to binarized form internally,
    # so we don't need to convert them here
    labels = labels.reshape(-1, 1)
    outputs = outputs.reshape(-1, 1)
    macro_f_measure, per_class_f_measure, classes = compute_f_measure(labels, outputs)
    return {"f_measure": macro_f_measure}


def compute_digitization_metrics(
    labels: Sequence[Dict[str, np.ndarray]],
    outputs: Sequence[CINC2024Outputs],
) -> Dict[str, float]:
    """Compute the metrics for the "digitization" task.

    Parameters
    ----------
    labels : Sequence[Dict[str, np.ndarray]]
        The labels for the records.
    outputs : Sequence[CINC2024Outputs]
        The outputs for the records.

    Returns
    -------
    Dict[str, float]
        The computed challenge metrics for the "digitization" task.

    """
    assert len(labels) == len(outputs), "The number of labels and outputs should be the same"
    if not all([item.digitization is not None for item in outputs]):
        return {"snr": np.nan, "snr_median": np.nan, "ks": np.nan, "asci": np.nan, "weighted_absolute_difference": np.nan}
    assert all(
        [len(label["digitization"]) == len(output.digitization) for label, output in zip(labels, outputs)]
    ), "The number of 'digitization' labels and outputs should be the same"
    # TODO: implement the computation of the digitization metrics
    raise NotImplementedError("compute_digitization_metrics is not implemented")
