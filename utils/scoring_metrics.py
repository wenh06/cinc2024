from typing import Dict, Optional, Sequence, Union

import numpy as np
from torch_ecg.utils.misc import list_sum

from helper_code import compute_f_measure
from outputs import CINC2024Outputs

__all__ = [
    "compute_challenge_metrics",
]


def compute_challenge_metrics(
    labels: Sequence[Dict[str, np.ndarray]],
    outputs: Sequence[CINC2024Outputs],
    keeps: Optional[Union[str, Sequence[str]]] = None,
) -> Dict[str, float]:
    """Compute the challenge metrics.

    Parameters
    ----------
    labels : Sequence[Dict[str, np.ndarray]]
        The labels for the records.
    outputs : Sequence[CINC2024Outputs]
        The outputs for the records.
    keeps : Union[str, Sequence[str]], optional
        Metrics to keep, available options are "dx" and "digitization".
        By default all metrics are computed.

    Returns
    -------
    Dict[str, float]
        The computed challenge metrics for "dx" and "digitization" (at least one of them).
        nan values are returned for the metrics that are not computed due to missing outputs.

    Examples
    --------
    >>> labels = [
            {"dx": [["Acute MI", "AFIB/AFL"], [], ["Normal"]]}, {"dx": [["Normal"], ["Old MI", "PVC"]]}
        ]
    >>> outputs = [
            CINC2024Outputs(
                dx=[["Old MI", "AFIB/AFL"], ["HYP"], ["Normal"]],
                dx_classes=["NORM", "Acute MI", "Old MI", "STTC", "CD", "HYP", "PAC", "PVC", "AFIB/AFL", "TACHY", "BRADY"],
            ),
            CINC2024Outputs(
                dx=[[], ["PVC"]],
                dx_classes=["NORM", "Acute MI", "Old MI", "STTC", "CD", "HYP", "PAC", "PVC", "AFIB/AFL", "TACHY", "BRADY"],
            )
        ]
    >>> compute_challenge_metrics(labels, outputs, keeps="dx")
    {'dx_f_measure': 0.5333333333333333}

    """
    metrics = {}
    if keeps is None:
        keeps = ["dx", "digitization"]
    elif isinstance(keeps, str):
        keeps = [keeps]
    keeps = [keep.lower() for keep in keeps]
    if "dx" in keeps:
        metrics.update({f"dx_{metric}": value for metric, value in compute_classification_metrics(labels, outputs).items()})
    if "digitization" in keeps:
        metrics.update(
            {f"digitization_{metric}": value for metric, value in compute_digitization_metrics(labels, outputs).items()}
        )
    return metrics


def compute_classification_metrics(
    labels: Sequence[Dict[str, np.ndarray]],
    outputs: Sequence[CINC2024Outputs],
    classes: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Compute the metrics for the "dx" classification task.

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
    classes : Optional[Sequence[str]], optional
        The class names for the "dx" task, by default None.
        If None, the class names are extracted from the "dx_classes" field of the outputs

    Returns
    -------
    Dict[str, float]
        The computed challenge metrics for the "dx" task.

    Examples
    --------
    >>> labels = [
            {"dx": [["Acute MI", "AFIB/AFL"], [], ["Normal"]]}, {"dx": [["Normal"], ["Old MI", "PVC"]]}
        ]
    >>> outputs = [
            CINC2024Outputs(
                dx=[["Old MI", "AFIB/AFL"], ["HYP"], ["Normal"]],
                dx_classes=["NORM", "Acute MI", "Old MI", "STTC", "CD", "HYP", "PAC", "PVC", "AFIB/AFL", "TACHY", "BRADY"],
            ),
            CINC2024Outputs(
                dx=[[], ["PVC"]],
                dx_classes=["NORM", "Acute MI", "Old MI", "STTC", "CD", "HYP", "PAC", "PVC", "AFIB/AFL", "TACHY", "BRADY"],
            )
        ]
    >>> compute_classification_metrics(labels, outputs)
    {'f_measure': 0.5333333333333333}

    """
    assert len(labels) == len(outputs), "The number of labels and outputs should be the same"
    if not all([item.dx is not None for item in outputs]):
        return {"f_measure": np.nan}
    assert all(
        [len(label["dx"]) == len(output.dx) for label, output in zip(labels, outputs)]
    ), "The number of 'dx' labels and outputs should be the same"
    if classes is None:
        classes = outputs[0].dx_classes
    # concatenate the labels and outputs
    # labels = np.concatenate([label["dx"] for label in labels], axis=0)
    # outputs = np.concatenate([output.dx for output in outputs], axis=0)
    labels = list_sum([label["dx"] for label in labels])
    outputs = list_sum([output.dx for output in outputs])
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
