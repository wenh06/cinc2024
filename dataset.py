"""
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Set, Union

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import ReprMixin
from tqdm.auto import tqdm

from cfg import TrainCfg
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
        self.config = CFG(deepcopy(TrainCfg))
        if config is not None:
            self.config.update(deepcopy(config))
        self.training = training
        self.lazy = lazy

        if self.config.predict_bbox:
            assert self.config.roi_only is False, "predict_bbox and roi_only cannot be True at the same time"

        A_kw = {}
        if self.config.predict_bbox:
            A_kw["bbox_params"] = A.BboxParams(format=self.config.bbox_format, label_fields=["category_id"])
        if self.config.use_augmentation:
            if self.training:
                self.transform = A.Compose(
                    [
                        A.RandomBrightnessContrast(p=0.5),
                        A.HueSaturationValue(p=0.1),
                    ],
                    **A_kw,
                )
            else:
                self.transform = A.Compose(
                    [
                        A.NoOp(),
                    ],
                    **A_kw,
                )
        else:
            self.transform = None

        if self.config.get("db_dir", None) is None:
            self.config.db_dir = reader_kwargs.pop("db_dir", None)
            assert self.config.db_dir is not None, "db_dir must be specified"
        else:
            reader_kwargs.pop("db_dir", None)
        self.config.db_dir = Path(self.config.db_dir).expanduser().resolve()

        # updates reader_kwargs with the config
        for kw in ["fs", "working_dir", "synthetic_images_dir"]:
            if kw not in reader_kwargs and hasattr(self.config, kw):
                reader_kwargs[kw] = getattr(self.config, kw)

        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        self.__cache = None
        self.reader = CINC2024Reader(db_dir=self.config.db_dir, **reader_kwargs)

        if self.config.predict_bbox:
            # process the bounding boxes
            self.reader._df_images["bbox_formatted"] = self.reader._df_images.index.map(
                lambda img_id: self.reader.load_bbox(img_id, fmt=self.config.bbox_format, return_dict=True)
            )

            # register `pandas.progress_apply` with `tqdm`
            tqdm.pandas(desc="Processing bounding boxes", dynamic_ncols=True, mininterval=1.0)
            # process the bounding boxes
            if self.config.bbox_mode.lower() == "full":
                pass  # make no change to "bbox_formatted"
            elif self.config.bbox_mode.lower() == "roi_only":
                # merge all waveform boxes into one ROI box
                self.reader._df_images["bbox_formatted"] = self.reader._df_images["bbox_formatted"].progress_apply(
                    lambda x: bbox2roi(x, fmt=self.config.bbox_format)
                )
            elif self.config.bbox_mode.lower() == "merge_horizontal":
                # merge the waveform bounding boxes that are horizontally adjacent
                self.reader._df_images["bbox_formatted"] = self.reader._df_images["bbox_formatted"].progress_apply(
                    lambda x: merge_horizontal_bbox(x, fmt=self.config.bbox_format)
                )
            else:
                raise ValueError(f"unsupported bbox_mode {self.config.bbox_mode}")

        ecg_ids = self._train_test_split(train_ratio=self.config.train_ratio)
        self._df_data = self.reader._df_images[self.reader._df_images.ecg_id.isin(ecg_ids)]
        self._df_data["ecg_path"] = self._df_data["ecg_id"].apply(lambda x: self.reader._df_records.loc[x, "path"])
        dx_class_map = {k: i for i, k in enumerate(self.config.classes)}
        self._df_data["dx"] = self._df_data["ecg_id"].apply(lambda x: self.reader.load_dx_ann(x, class_map=dx_class_map))
        # one-hot encode the dx
        self._df_data["dx"] = self._df_data["dx"].apply(lambda x: one_hot_encode(x, num_classes=len(self.config.classes)))

        if self.config.predict_bbox:
            # select only the images with bounding boxes of the waveforms
            self._df_data = self._df_data[self._df_data["lead_bbox_file"].apply(lambda x: x is not None)]
        if self.config.predict_mask:
            # select only the images with masks of the waveforms
            self._df_data = self._df_data[self._df_data["path"].apply(lambda x: x.with_suffix(".json.gz").exists())]

        self.fdr = FastDataReader(self.reader, self._df_data, self.config, self.transform)

        if not self.lazy:
            self._load_all_data()

    def __len__(self) -> int:
        if self.cache is None:
            # self._load_all_data()
            return len(self.fdr)
        return len(self.cache["images"])

    def __getitem__(self, index: Union[int, slice]) -> Dict[str, np.ndarray]:
        if self.cache is None:
            # self._load_all_data()
            return self.fdr[index]
        return {k: v[index] for k, v in self.cache.items()}

    def _load_all_data(self) -> None:
        """Load all data into memory.

        .. warning::

            caching all data into memory is not recommended, which would certainly cause OOM error.
            The RAM of the Challenge is only 64GB.

        """
        tmp_cache = []
        with tqdm(range(len(self.fdr)), desc="Loading data", unit="image") as pbar:
            for idx in pbar:
                tmp_cache.append(self.fdr[idx])
        keys = tmp_cache[0].keys()
        self.__cache = {
            k: np.concatenate([v[k] if v[k].shape == (1,) else v[k][np.newaxis, ...] for v in tmp_cache]) for k in keys
        }
        del tmp_cache

    def _train_test_split(self, train_ratio: float = 0.9, force_recompute: bool = False) -> List[str]:
        if train_ratio == 0.9:
            split = self.reader.default_train_val_split
            if self.training:
                return split["train"]
            else:
                return split["val"]
        elif train_ratio == 0.8:
            split = self.reader.default_train_val_test_split
            if self.training:
                return split["train"]
            else:
                return split["val"] + split["test"]
        else:
            raise NotImplementedError(f"split for {train_ratio=} is not implemented yet")

    @property
    def cache(self) -> Dict[str, np.ndarray]:
        return self.__cache

    @property
    def data_fields(self) -> Set[str]:
        # fmt: off
        return set([
            "image", "image_id",  # basic fields
            "dx",  # classification
            "digitization",  # digitization
            "bbox",  # object detection
            "mask",  # mask prediction
        ])
        # fmt: on

    def extra_repr_keys(self) -> List[str]:
        return ["reader", "training"]


class FastDataReader(ReprMixin, Dataset):
    def __init__(
        self,
        reader: CINC2024Reader,
        df_data: Sequence[str],
        config: CFG,
        transform: A.Compose,
    ) -> None:
        self.reader = reader
        self.df_data = df_data
        self.images = self.df_data.index.tolist()
        self.config = config
        self.transform = transform
        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: Union[int, slice]) -> Dict[str, np.ndarray]:
        if isinstance(index, slice):
            # note that the images are of the same size
            # return default_collate_fn([self[i] for i in range(*index.indices(len(self)))])
            if self.config.predict_bbox or self.config.predict_mask:
                return naive_collate_fn([self[i] for i in range(*index.indices(len(self)))])
            return collate_fn([self[i] for i in range(*index.indices(len(self)))])

        row = self.df_data.loc[self.images[index]]
        # load the image, of type numpy ndarray, of shape (H, W, C)
        image = self.reader.load_image(row.name, fmt="np", roi_only=self.config.roi_only, roi_padding=self.config.roi_padding)

        # image_id (of `int` type) required by some object detection models
        # `str` type is not supported by pytorch
        # data = {"image_id": index, "image_name": row.name}
        data = {"image_id": index}

        A_kw = {}
        if self.config.predict_bbox:
            A_kw["bboxes"] = row["bbox_formatted"]["bbox"]
            A_kw["category_id"] = row["bbox_formatted"]["category_id"]
        if self.config.predict_mask:
            # load the mask
            mask = self.reader.load_waveform_mask(row.name, roi_only=self.config.roi_only, roi_padding=self.config.roi_padding)
            A_kw["mask"] = mask

        if self.transform is not None:
            A_out = self.transform(image=image, **A_kw)
        else:
            A_out = {"image": image}
            if self.config.predict_bbox:
                A_out["bboxes"] = A_kw["bboxes"]
                A_out["category_id"] = A_kw["category_id"]
            if self.config.predict_mask:
                A_out["mask"] = A_kw["mask"]
        data["image"] = A_out.pop("image")

        if self.config.predict_dx:
            data["dx"] = row["dx"]  # int
        if self.config.predict_digitization:
            # one has to load the signal, pad or trim to self.config.max_len,
            # and also create a mask for it, since the ECG plot typically contains only 2.5s for each lead,
            # except for a default lead (lead II)
            # keys are "digitization" and "mask"
            raise NotImplementedError("data preparation for digitization prediction is not implemented yet")
        if self.config.predict_bbox:
            # load the bounding boxes
            if self.config.bbox_format == "coco":
                data["bbox"] = format_image_annotations_as_coco(
                    image_id=index,
                    bboxes=A_out["bboxes"],
                    categories=A_out["category_id"],
                    # the area should be the original area or the area after augmentation?
                    areas=row["bbox_formatted"]["area"],
                    # areas=[(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in A_out["bboxes"]],
                )
                data["bbox"]["image_size"] = list(image.shape[:2])  # H, W
                data["bbox"]["format"] = self.config.bbox_format
            else:
                raise NotImplementedError(f"bbox format {self.config.bbox_format} is not implemented yet")
        if self.config.predict_mask:
            data["mask"] = A_out.pop("mask")

        return data

    def extra_repr_keys(self) -> List[str]:
        return [
            "reader",
        ]


def collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, Union[torch.Tensor, List[np.ndarray]]]:
    """Collate function for the dataset.

    Parameters
    ----------
    batch : List[Dict[str, np.ndarray]]
        The batch of data.

    Returns
    -------
    Dict[str, Union[torch.Tensor, List[np.ndarray]]]
        The collated data.

    """
    out_tensors = {}
    for k in batch[0].keys():
        if k == "image":
            continue
        out_tensors[k] = torch.from_numpy(np.concatenate([[b[k]] for b in batch], axis=0))
    out_tensors["image"] = [b["image"] for b in batch]
    return out_tensors


def naive_collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, list]:
    """Naive collate function for the dataset.

    This function only concatenates the data into a list, typically used for
    object detection tasks. (Pre)Processors of the Huggingface object detection
    models can handle the data in this format.

    Parameters
    ----------
    batch : List[Dict[str, np.ndarray]]
        The batch of data.

    Returns
    -------
    Dict[str, list]
        The collated data.

    """
    batched_data = {}
    for k in batch[0].keys():
        batched_data[k] = [b[k] for b in batch]
        if k == "dx":
            batched_data[k] = torch.from_numpy(np.stack(batched_data[k], axis=0))
    return batched_data


def format_image_annotations_as_coco(
    image_id: Union[int, str], bboxes: Sequence[Sequence[float]], categories: Sequence[int], areas: Sequence[float]
) -> Dict[str, List[Dict[str, Union[int, float]]]]:
    """Format the image annotations as COCO format.

    Parameters
    ----------
    image_id : Union[int, str]
        The image ID.
    bboxes : Sequence[Sequence[float]]
        The bounding boxes.
    categories : Sequence[int]
        The category IDs.
    areas : Sequence[float]
        The areas of the bounding boxes.

    Returns
    -------
    Dict[str, List[Dict[str, Union[int, float]]]]
        The formatted annotations, keys are "annotations" and "image_id".
        "annotations" is a list of dictionaries, each of which contains the following keys:
        - "image_id" : int
        - "is_crowd" : 0
        - "bbox" : List[float]
        - "category_id" : int
        - "area" : float

    """
    annotations = []
    for bbox, area, category_id in zip(bboxes, areas, categories):
        annotations.append(
            {
                "image_id": image_id,
                "is_crowd": 0,
                "bbox": list(bbox),
                "category_id": category_id,
                "area": area,
            }
        )
    return {"annotations": annotations, "image_id": image_id}


def bbox2roi(bbox: Dict[str, list], fmt: str) -> Dict[str, List]:
    """Merge all waveform bounding boxes into one ROI box.

    Parameters
    ----------
    bbox : Dict[str, list]
        The bounding boxes, which is a dictionary consisting of
        - "bbox" : list of shape `(n, 4)`.
        - "category_id" : list of shape `(n,)`.
        - "category_name" : list of shape `(n,)`.
        - "area" : list of shape `(n,)`.
    fmt : {"coco", "voc", "yolo"}
        Format of the bounding boxes in `bbox`.

    Returns
    -------
    Dic[str, List]
        The bounding boxes with all waveform bounding boxes merged into one ROI box,
        "category_name" and "category_id" left unchanged.

    """
    indices = np.where(np.array(bbox["category_name"]) == "waveform")[0]
    roi_bbox = np.array(bbox["bbox"])[indices]
    if fmt == "coco":  # (x, y, w, h)
        # to voc format (xmin, ymin, xmax, ymax)
        roi_bbox[..., [2, 3]] = roi_bbox[..., [2, 3]] + roi_bbox[..., [0, 1]]
        roi_bbox = [
            roi_bbox[..., 0].min(),
            roi_bbox[..., 1].min(),
            roi_bbox[..., 2].max() - roi_bbox[..., 0].min(),
            roi_bbox[..., 3].max() - roi_bbox[..., 1].min(),
        ]
        roi_area = roi_bbox[2] * roi_bbox[3]
    elif fmt == "voc":  # (xmin, ymin, xmax, ymax)
        roi_bbox = [roi_bbox[..., 0].min(), roi_bbox[..., 1].min(), roi_bbox[..., 2].max(), roi_bbox[..., 3].max()]
        roi_area = (roi_bbox[2] - roi_bbox[0]) * (roi_bbox[3] - roi_bbox[1])
    elif fmt == "yolo":  # (x_center, y_center, w, h)
        # to voc format (xmin, ymin, xmax, ymax)
        roi_bbox[..., [0, 1]] = roi_bbox[..., [0, 1]] - roi_bbox[..., [2, 3]] / 2
        roi_bbox[..., [2, 3]] = roi_bbox[..., [0, 1]] + roi_bbox[..., [2, 3]]
        roi_bbox = [
            (roi_bbox[..., 0].min() + roi_bbox[..., 2].max()) / 2,
            (roi_bbox[..., 1].min() + roi_bbox[..., 3].max()) / 2,
            roi_bbox[..., 2].max() - roi_bbox[..., 0].min(),
            roi_bbox[..., 3].max() - roi_bbox[..., 1].min(),
        ]
        roi_area = roi_bbox[2] * roi_bbox[3] * (bbox["area"][0] / bbox["bbox"][0][2] / bbox["bbox"][0][3])
    else:
        raise ValueError(f"unsupported format {fmt}")

    return {
        "bbox": np.array(bbox["bbox"])[~indices].tolist() + [roi_bbox],
        "category_id": np.array(bbox["category_id"])[~indices].tolist() + [bbox["category_id"][indices[0]]],
        "category_name": np.array(bbox["category_name"])[~indices].tolist() + [bbox["category_name"][indices[0]]],
        "area": np.array(bbox["area"])[~indices].tolist() + [roi_area],
    }


def merge_horizontal_bbox(bbox: Dict[str, list], fmt: str) -> Dict[str, list]:
    """Merge all waveform bounding boxes that are horizontally adjacent.

    Adjacent bounding boxes are merged into one bounding box:
    - I, aVR, V1, V4
    - II, aVL, V2, V5
    - III, aVF, V3, V6

    Parameters
    ----------
    bbox : Dict[str, list]
        The bounding boxes, which is a dictionary consisting of
        - "bbox" : list of shape `(n, 4)`.
        - "category_id" : list of shape `(n,)`.
        - "category_name" : list of shape `(n,)`.
        - "area" : list of shape `(n,)`.
    fmt : {"coco", "voc", "yolo"}
        Format of the bounding boxes in `bbox`.

    Returns
    -------
    Dict[str, list]
        The bounding boxes with horizontally adjacent bounding boxes merged.

    """
    # convert to list of dictionaries
    waveform_boxes = CINC2024Reader.match_bbox(
        [
            {"bbox": b, "category_id": c, "category_name": n, "area": a}
            for b, c, n, a in zip(bbox["bbox"], bbox["category_id"], bbox["category_name"], bbox["area"])
        ],
        fmt=fmt,
    )
    # after `match_bbox`, an item with key "lead_name" will be added to each dict.

    # full waveform has the largest width
    if fmt == "coco":
        full_waveform_idx = np.argmax([box["bbox"][2] for box in waveform_boxes])
    elif fmt == "voc":
        full_waveform_idx = np.argmax([box["bbox"][2] - box["bbox"][0] for box in waveform_boxes])
    elif fmt == "yolo":
        full_waveform_idx = np.argmax([box["bbox"][2] for box in waveform_boxes])
        area_ratio = bbox["area"][0] / bbox["bbox"][0][2] / bbox["bbox"][0][3]
    else:
        raise ValueError(f"unsupported format {fmt}")

    lead_name_bbox_indices = np.append(np.where(np.array(bbox["category_name"]) != "waveform")[0], full_waveform_idx)
    waveform_bbox_indices = np.where(np.array(bbox["category_name"]) == "waveform")[0]

    new_bbox = {
        "bbox": np.array(bbox["bbox"])[lead_name_bbox_indices].tolist(),
        "category_id": np.array(bbox["category_id"])[lead_name_bbox_indices].tolist(),
        "category_name": np.array(bbox["category_name"])[lead_name_bbox_indices].tolist(),
        "area": np.array(bbox["area"])[lead_name_bbox_indices].tolist(),
    }

    # merge horizontally adjacent leads
    waveform_lead_names = np.array([box["lead_name"] for box in waveform_boxes])
    for set_of_lead_names in [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]:
        indices = np.where(np.isin(waveform_lead_names, set_of_lead_names))[0]
        # remove full_waveform_idx from indices if it is in the set
        indices = indices[indices != full_waveform_idx]
        boxes = np.array([box["bbox"] for box in waveform_boxes])[indices]
        if fmt == "coco":
            # (x, y, w, h)
            # to voc format (xmin, ymin, xmax, ymax)
            boxes[..., [2, 3]] = boxes[..., [2, 3]] + boxes[..., [0, 1]]
            new_bbox["bbox"].append(
                [
                    boxes[..., 0].min(),
                    boxes[..., 1].min(),
                    boxes[..., 2].max() - boxes[..., 0].min(),
                    boxes[..., 3].max() - boxes[..., 1].min(),
                ]
            )
            new_bbox["area"].append((boxes[..., 2].max() - boxes[..., 0].min()) * (boxes[..., 3].max() - boxes[..., 1].min()))
        elif fmt == "voc":
            # (xmin, ymin, xmax, ymax)
            new_bbox["bbox"].append(
                [
                    boxes[..., 0].min(),
                    boxes[..., 1].min(),
                    boxes[..., 2].max(),
                    boxes[..., 3].max(),
                ]
            )
            new_bbox["area"].append((boxes[..., 2].max() - boxes[..., 0].min()) * (boxes[..., 3].max() - boxes[..., 1].min()))
        elif fmt == "yolo":
            # (x_center, y_center, w, h)
            # to voc format (xmin, ymin, xmax, ymax)
            boxes[..., [0, 1]] = boxes[..., [0, 1]] - boxes[..., [2, 3]] / 2
            boxes[..., [2, 3]] = boxes[..., [0, 1]] + boxes[..., [2, 3]]
            new_bbox["bbox"].append(
                [
                    (boxes[..., 0].min() + boxes[..., 2].max()) / 2,
                    (boxes[..., 1].min() + boxes[..., 3].max()) / 2,
                    boxes[..., 2].max() - boxes[..., 0].min(),
                    boxes[..., 3].max() - boxes[..., 1].min(),
                ]
            )
            new_bbox["area"].append(
                (boxes[..., 2].max() - boxes[..., 0].min()) * (boxes[..., 3].max() - boxes[..., 1].min()) * area_ratio
            )
        new_bbox["category_id"].append(bbox["category_id"][waveform_bbox_indices[0]])
        new_bbox["category_name"].append(bbox["category_name"][waveform_bbox_indices[0]])

    return new_bbox


def one_hot_encode(dx: List[int], num_classes: int) -> np.ndarray:
    """One-hot encode the diagnostic classes.

    Parameters
    ----------
    dx : List[int]
        The diagnostic classes.
    num_classes : int
        The number of classes.

    Returns
    -------
    np.ndarray
        The one-hot encoded diagnostic classes.

    """
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[dx] = 1
    return one_hot
