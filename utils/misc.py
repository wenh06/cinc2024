"""
Miscellaneous functions.
"""

from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from bib_lookup.utils import is_notebook
from PIL import Image, ImageDraw, ImageFont
from torch_ecg.utils.misc import list_sum

from const import INPUT_IMAGE_TYPES

__all__ = [
    "func_indicator",
    "predict_proba_ordered",
    "load_submission_log",
    "view_image_with_bbox",
    "view_roi",
    "get_target_sizes",
]


def func_indicator(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            print("\n" + "-" * 100)
            print(f"  Start {name}  ".center(100, "-"))
            print("-" * 100 + "\n")
            func(*args, **kwargs)
            print("\n" + "-" * 100)
            print(f"  End {name}  ".center(100, "-"))
            print("-" * 100 + "\n")

        return wrapper

    return decorator


def predict_proba_ordered(probs: np.ndarray, classes_: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    """Workaround for the problem that one can not set explicitly
    the list of classes for models in sklearn.

    Modified from https://stackoverflow.com/a/32191708

    Parameters
    ----------
    probs : numpy.ndarray
        Array of probabilities, output of `predict_proba` method of sklearn models.
    classes_ : numpy.ndarray
        Array of classes, output of `classes_` attribute of sklearn models.
    all_classes : numpy.ndarray
        All possible classes (superset of `classes_`).

    Returns
    -------
    numpy.ndarray
        Array of probabilities, ordered according to all_classes.

    """
    proba_ordered = np.zeros((probs.shape[0], all_classes.size), dtype=float)
    sorter = np.argsort(all_classes)  # http://stackoverflow.com/a/32191125/395857
    idx = sorter[np.searchsorted(all_classes, classes_, sorter=sorter)]
    proba_ordered[:, idx] = probs
    return proba_ordered


def load_submission_log() -> pd.DataFrame:
    """Load the submission log.

    Returns
    -------
    df_sub_log : pandas.DataFrame
        The submission log,
        sorted by challenge score in descending order.

    """
    path = Path(__file__).parents[1] / "submissions"
    df_sub_log = pd.DataFrame.from_dict(yaml.safe_load(path.read_text())["Official Phase"], orient="index").sort_values(
        "score", ascending=False
    )
    return df_sub_log


def view_image_with_bbox(
    image: Union[np.ndarray, torch.Tensor, Image.Image],
    bbox: Optional[Union[List[dict], Dict[str, list]]] = None,
    fmt: Literal["coco", "voc", "yolo"] = "coco",
    cat_names: Optional[List[str]] = None,
    mask: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Optional[Image.Image]:
    """View the image with bounding boxes.

    Parameters
    ----------
    image : Union[np.ndarray, torch.Tensor, Image.Image]
        The image to be viewed.
    bbox : Dict[str, list] or List[dict], optional
        The bounding boxes, which is a dictionary consisting of
        - "bbox" : list of shape `(n, 4)`.
        - "category_id" : list of shape `(n,)`.
        - "category_name" : list of shape `(n,)`.
        - "area" : list of shape `(n,)`.
        Or a list of dictionaries, each dictionary consists of
        - "bbox" : list of shape `(4,)`.
        - "category_id" : int.
    fmt : {"coco", "voc", "yolo"}, default "coco"
        Format of the bounding boxes in `bbox`.
    cat_names : List[str], optional
        The category names.
    mask : numpy.ndarray, optional
        The mask to be overlaid on the image.
        The mask is a binary (0 or 1) with shape `(height, width)`.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Image.Image
        The image with bounding boxes.

    """
    if isinstance(image, torch.Tensor):
        image = image.clone().numpy().transpose(1, 2, 0)
    if isinstance(image, np.ndarray):
        assert image.ndim == 3 and image.shape[-1] == 3, f"unsupported shape {image.shape}"
        img = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        # make a copy
        img = image.copy()
    else:
        raise ValueError(f"unsupported type {type(image)}")

    if mask is not None:
        # overlay the mask on the image with green color and 50% transparency
        # overlay_color = (0, 255, 0)
        overlay_color = kwargs.get("mask_color", (0, 255, 0))
        overlay = Image.new("RGBA", img.size, overlay_color + (128,))
        overlay = Image.fromarray(np.asarray(overlay) * mask[:, :, np.newaxis].astype(np.uint8))

    if bbox is None:
        if mask is not None:
            img = Image.alpha_composite(img.convert("RGBA"), overlay)
        if is_notebook():
            return img
        img.show()
        return

    if isinstance(bbox, list):
        # convert to dictionary
        bbox_dict = {
            "bbox": [item["bbox"] for item in bbox],
            "category_id": [item["category_id"] for item in bbox],
        }
        if cat_names is not None:
            bbox_dict["category_name"] = [cat_names[item["category_id"]] for item in bbox]
        else:
            bbox_dict["category_name"] = [str(item["category_id"]) for item in bbox]
    elif isinstance(bbox, dict):
        bbox_dict = deepcopy(bbox)
    else:
        raise ValueError(f"unsupported type {type(bbox)}")

    img_width, img_height = img.size
    if fmt == "coco":
        # (x, y, w, h)
        # to voc format (xmin, ymin, xmax, ymax)
        bbox_dict["bbox"] = np.array(bbox_dict["bbox"])
        bbox_dict["bbox"][..., [2, 3]] = bbox_dict["bbox"][..., [2, 3]] + bbox_dict["bbox"][..., [0, 1]]
    elif fmt == "voc":
        # (xmin, ymin, xmax, ymax)
        pass
    elif fmt == "yolo":
        # (x_center, y_center, w, h)
        # to voc format (xmin, ymin, xmax, ymax)
        bbox_dict["bbox"] = np.array(bbox_dict["bbox"])
        bbox_dict["bbox"][..., [0, 1]] = bbox_dict["bbox"][..., [0, 1]] - bbox_dict["bbox"][..., [2, 3]] / 2
        bbox_dict["bbox"][..., [2, 3]] = bbox_dict["bbox"][..., [0, 1]] + bbox_dict["bbox"][..., [2, 3]]
        bbox_dict["bbox"][..., [0, 2]] *= img_width
        bbox_dict["bbox"][..., [1, 3]] *= img_height
    else:
        raise ValueError(f"unsupported format {fmt}")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", int(min(img.size) * 0.025))
    for box, cat_name in zip(bbox_dict["bbox"], bbox_dict["category_name"]):
        draw.rectangle(box.tolist(), outline=kwargs.get("box_color", "green"), width=kwargs.get("box_width", 5))
        draw.text((box[0], box[1]), cat_name, fill=kwargs.get("box_color", "green"), font=font, anchor="lb")

    if mask is not None:
        img = Image.alpha_composite(img.convert("RGBA"), overlay)

    # if is jupyter notebook, show the image inline
    if is_notebook():
        return img
    img.show()


def view_roi(
    image: Union[np.ndarray, torch.Tensor, Image.Image],
    roi: Sequence[int],
    fmt: Literal["coco", "voc", "yolo"] = "coco",
    binarize: bool = False,
    binarize_percentile: float = 1.0,
) -> Optional[Image.Image]:
    """View the region of interest (ROI) of the image.

    Parameters
    ----------
    image : Union[np.ndarray, torch.Tensor, Image.Image]
        The image to be viewed.
    roi : Sequence[int]
        The region of interest, a list of integers of length 4.
    fmt : {"coco", "voc", "yolo"}, default "coco"
        Format of the bounding boxes in `bbox`.
    binarize : bool, default False
        Whether to binarize the ROI.
    binarize_percentile : float, default 1.0
        The percentile to binarize the ROI.

    Returns
    -------
    Image.Image
        The image with ROI.

    """
    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose(1, 2, 0)
    if isinstance(image, np.ndarray):
        assert image.ndim == 3 and image.shape[-1] == 3, f"unsupported shape {image.shape}"
        img = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise ValueError(f"unsupported type {type(image)}")

    img_width, img_height = img.size
    if fmt == "coco":
        # (x, y, w, h)
        # to voc format (xmin, ymin, xmax, ymax)
        roi = np.array(roi)
        roi[2:] = roi[:2] + roi[2:]
    elif fmt == "voc":
        # (xmin, ymin, xmax, ymax)
        roi = np.array(roi)
    elif fmt == "yolo":
        # (x_center, y_center, w, h)
        # to voc format (xmin, ymin, xmax, ymax)
        roi = np.array(roi)
        roi[[0, 1]] = roi[[0, 1]] - roi[[2, 3]] / 2
        roi[[2, 3]] = roi[[0, 1]] + roi[[2, 3]]
        roi[[0, 2]] *= img_width
        roi[[1, 3]] *= img_height
    else:
        raise ValueError(f"unsupported format {fmt}")
    img = img.crop(roi.tolist())

    if binarize:
        threshold = np.percentile(np.asarray(img.convert("L")), binarize_percentile)
        print(f"Binarization threshold: {threshold}")
        img = img.point(lambda p: 255 if p > threshold else 0).convert("L")

    # if is jupyter notebook, show the image inline
    if is_notebook():
        return img
    img.show()


def get_target_sizes(img: INPUT_IMAGE_TYPES, channels: int = 3) -> List[Tuple[int, int]]:
    """Get the target sizes of the input image(s).

    Parameters
    ----------
    img : numpy.ndarray, or torch.Tensor, or PIL.Image.Image, or list or tuple
        Input image.
    channels : int, default 3
        The number of channels of the input image.
        Used to determine the channel dimension of the input image.

    Returns
    -------
    List[Tuple[int, int]]
        The list containing the target size `(height, width)` of each image.

    """
    if isinstance(img, (list, tuple)):
        target_sizes = list_sum(get_target_sizes(item, channels) for item in img)
    elif isinstance(img, (np.ndarray, torch.Tensor)):
        if img.ndim == 3:
            if img.shape[0] == channels:  # channels first
                target_sizes = [tuple(img.shape[1:])]
            else:  # channels last
                target_sizes = [tuple(img.shape[:-1])]
        elif img.ndim == 4:
            target_sizes = list_sum(get_target_sizes(item, channels) for item in img)
    elif isinstance(img, Image.Image):
        target_sizes = [img.size[::-1]]
    return target_sizes


def schmidt_spike_removal(
    original_signal: np.ndarray,
    fs: int,
    window_size: float = 0.5,
    threshold: float = 3.0,
    eps: float = 1e-4,
) -> np.ndarray:
    """Spike removal using Schmidt algorithm.

    Parameters
    ----------
    original_signal : np.ndarray
        The original signal.
    fs : int
        The sampling frequency.
    window_size : float, default 0.5
        The sliding window size, with units in seconds.
    threshold : float, default 3.0
        The threshold (multiplier for the median value) for detecting spikes.
    eps : float, default 1e-4
        The epsilon for numerical stability.

    Returns
    -------
    despiked_signal : np.ndarray,
        The despiked signal.

    """
    window_size = round(fs * window_size)
    nframes, res = divmod(original_signal.shape[0], window_size)
    frames = original_signal[: window_size * nframes].reshape((nframes, window_size))
    if res > 0:
        nframes += 1
        frames = np.concatenate((frames, original_signal[-window_size:].reshape((1, window_size))), axis=0)
    MAAs = np.abs(frames).max(axis=1)  # of shape (nframes,)

    while len(np.where(MAAs > threshold * np.median(MAAs))[0]) > 0:
        frame_num = np.where(MAAs == MAAs.max())[0][0]
        spike_position = np.argmax(np.abs(frames[frame_num]))
        zero_crossings = np.where(np.diff(np.sign(frames[frame_num])))[0]
        spike_start = np.where(zero_crossings <= spike_position)[0]
        spike_start = zero_crossings[spike_start[-1]] if len(spike_start) > 0 else 0
        spike_end = np.where(zero_crossings >= spike_position)[0]
        spike_end = zero_crossings[spike_end[0]] + 1 if len(spike_end) > 0 else window_size
        frames[frame_num, spike_start:spike_end] = eps
        MAAs = np.abs(frames).max(axis=1)

    despiked_signal = original_signal.copy()
    if res > 0:
        despiked_signal[-window_size:] = frames[-1]
        nframes -= 1
    despiked_signal[: window_size * nframes] = frames[:nframes, ...].reshape((-1,))

    return despiked_signal
