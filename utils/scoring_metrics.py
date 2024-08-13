from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch_ecg.utils.misc import list_sum
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.segmentation.generalized_dice import GeneralizedDiceScore
from torchmetrics.segmentation.mean_iou import MeanIoU
from transformers.image_transforms import corners_to_center_format

from helper_code import compute_f_measure
from outputs import CINC2024Outputs

__all__ = [
    "compute_challenge_metrics",
]


def compute_challenge_metrics(
    labels: Sequence[Dict[str, Union[np.ndarray, List[dict]]]],
    outputs: Sequence[CINC2024Outputs],
    keeps: Optional[Union[str, Sequence[str]]] = None,
) -> Dict[str, float]:
    """Compute the challenge metrics.

    Parameters
    ----------
    labels : Sequence[Dict[str, Union[np.ndarray, List[dict]]]]
        The labels for the records.
        `labels` is produced by the dataset class (ref. dataset.py).
    outputs : Sequence[CINC2024Outputs]
        The outputs for the records.
    keeps : Union[str, Sequence[str]], optional
        Metrics to keep, available options are "dx", "digitization", "detection", "segmentation".
        By default all metrics are computed.

    Returns
    -------
    Dict[str, float]
        The computed challenge metrics for "dx", "digitization", "detection", "segmentation" (at least one of them).
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
        keeps = ["dx", "digitization", "detection", "segmentation"]
    elif isinstance(keeps, str):
        keeps = [keeps]
    keeps = [keep.lower() for keep in keeps]
    if "dx" in keeps:
        metrics.update({f"dx_{metric}": value for metric, value in compute_classification_metrics(labels, outputs).items()})
    if "digitization" in keeps:
        metrics.update(
            {f"digitization_{metric}": value for metric, value in compute_digitization_metrics(labels, outputs).items()}
        )
    if "detection" in keeps:
        metrics.update({f"detection_{metric}": value for metric, value in compute_detection_metrics(labels, outputs).items()})
    if "mask" in keeps:
        metrics.update(
            {f"segmentation_{metric}": value for metric, value in compute_segmentation_metrics(labels, outputs).items()}
        )
    return metrics


def compute_classification_metrics(
    labels: Sequence[Dict[str, Union[np.ndarray, List[dict]]]],
    outputs: Sequence[CINC2024Outputs],
    classes: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Compute the metrics for the "dx" classification task.

    Parameters
    ----------
    labels : Sequence[Dict[str, Union[np.ndarray, List[dict]]]]
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
    labels: Sequence[Dict[str, Union[np.ndarray, List[dict]]]],
    outputs: Sequence[CINC2024Outputs],
) -> Dict[str, float]:
    """Compute the metrics for the "digitization" task.

    Parameters
    ----------
    labels : Sequence[Dict[str, Union[np.ndarray, List[dict]]]]
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


def compute_detection_metrics(
    labels: Sequence[Dict[str, Union[np.ndarray, List[dict]]]],
    outputs: Sequence[CINC2024Outputs],
) -> Dict[str, float]:
    """Compute the metrics for the detection task.

    Parameters
    ----------
    labels : Sequence[Dict[str, Union[np.ndarray, List[dict]]]]
        The labels for the records.
    outputs : Sequence[CINC2024Outputs]
        The outputs for the records.

    Returns
    -------
    Dict[str, float]
        The computed challenge metrics for the detection task.

    """
    assert len(labels) == len(outputs), "The number of labels and outputs should be the same"
    if not all([item.bbox is not None for item in outputs]):
        return {"mAP": np.nan}
    assert all(
        [len(label["bbox"]) == len(output.bbox) for label, output in zip(labels, outputs)]
    ), "The number of 'bbox' labels and outputs should be the same"

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    classes = outputs[0].bbox_classes

    post_processed_targets = []  # A list consisting of dictionaries with keys "boxes", "labels"
    post_processed_predictions = []  # A list consisting of dictionaries with keys "boxes", "labels", "scores"
    for label, output in zip(labels, outputs):
        # label is a dictionary with keys "bbox" and other fields (not used here)
        for lb, op in zip(label["bbox"], output.bbox):
            # convert the labels to the required format ("voc", i.e. "xyxy")
            converted_targets = dict()
            converted_targets["labels"] = torch.Tensor([bbox_dict["category_id"] for bbox_dict in lb["annotations"]]).long()
            if lb["format"] == "coco":
                converted_targets["boxes"] = coco_to_voc_format(
                    torch.Tensor([bbox_dict["bbox"] for bbox_dict in lb["annotations"]])
                )
            elif lb["format"] == "voc":
                converted_targets["boxes"] = torch.Tensor([bbox_dict["bbox"] for bbox_dict in lb["annotations"]])
            else:  # yolo format
                height, width = lb["image_size"]
                converted_targets["boxes"] = corners_to_center_format(
                    torch.Tensor([bbox_dict["bbox"] for bbox_dict in lb["annotations"]])
                )
                converted_targets["boxes"] *= torch.Tensor([[width, height, width, height]])
            post_processed_targets.append(converted_targets)
            post_processed_predictions.append(
                {
                    "boxes": torch.Tensor(op["boxes"]),
                    "labels": torch.Tensor(op["category_id"]).long(),
                    "scores": torch.Tensor(op["scores"]),
                }
            )

    metric.update(post_processed_predictions, post_processed_targets)
    result = metric.compute()

    map_per_class = result.pop("map_per_class")
    mar_100_per_class = result.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(result.pop("classes"), map_per_class, mar_100_per_class):
        class_name = classes[class_id.item()]
        result[f"map_{class_name}"] = class_map
        result[f"mar_100_{class_name}"] = class_mar
    result = {k: v.item() for k, v in result.items()}
    return result


def coco_to_voc_format(boxes: torch.Tensor) -> torch.Tensor:
    """Convert the bounding boxes from COCO format to VOC format.

    Parameters
    ----------
    boxes : torch.Tensor
        The bounding boxes in COCO format (x1, y1, w, h).

    Returns
    -------
    torch.Tensor
        The bounding boxes in VOC format (x1, y1, x2, y2).

    """
    boxes_voc = boxes.clone()
    boxes_voc[:, [2, 3]] += boxes_voc[:, [0, 1]]
    return boxes_voc


def compute_segmentation_metrics(
    labels: Sequence[Dict[str, Union[np.ndarray, List[dict]]]],
    outputs: Sequence[CINC2024Outputs],
) -> Dict[str, float]:
    """Compute the metrics for the segmentation task.

    Parameters
    ----------
    labels : Sequence[Dict[str, Union[np.ndarray, List[dict]]]]
        The labels for the records.
    outputs : Sequence[CINC2024Outputs]
        The outputs for the records.

    Returns
    -------
    Dict[str, float]
        The computed challenge metrics for the segmentation task.

    """
    assert len(labels) == len(outputs), "The number of labels and outputs should be the same"
    if not all([item.waveform_mask is not None for item in outputs]):
        return {"mAP": np.nan}
    assert all(
        [len(label["mask"]) == len(output.waveform_mask) for label, output in zip(labels, outputs)]
    ), "The number of 'mask' labels and outputs should be the same"

    metric_miou = MeanIoU(num_classes=1, include_background=False, per_class=False)
    metric_dice = GeneralizedDiceScore(num_classes=1, include_background=False, per_class=False)

    count = 0
    for label, output in zip(labels, outputs):
        # label is a dictionary with keys "mask" and other fields (not used here)
        for lb, op in zip(label["mask"], output.waveform_mask):
            # convert the masks to the required format
            converted_targets = torch.Tensor(lb).long()
            for _ in range(4 - converted_targets.ndim):
                converted_targets = converted_targets.unsqueeze(0)
            converted_predictions = torch.Tensor(op).long()
            for _ in range(4 - converted_predictions.ndim):
                converted_predictions = converted_predictions.unsqueeze(0)
            metric_miou.update(converted_predictions, converted_targets)
            metric_dice.update(converted_predictions, converted_targets)
            count += 1

    # NOTE: the mean IoU is computed as the sum of mean IoU over all samples,
    # BUT not divided by the number of samples;
    # while the mean Dice is computed as the sum of mean Dice over all samples divided by the number of samples.
    result = {"miou": metric_miou.compute().item() / count, "dice": metric_dice.compute().item()}
    return result
