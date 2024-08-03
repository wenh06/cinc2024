"""
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Set, Union

import albumentations as A
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import ReprMixin
from torch_ecg.utils.utils_nn import default_collate_fn  # noqa: F401
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

        A_kw = {}
        if self.config.predict_bbox:
            A_kw["bbox_params"] = A.BboxParams(format=self.config.bbox_format, label_fields=["category_id"])
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
        # process the bounding boxes
        self.reader._df_images["bbox_formatted"] = self.reader._df_images.index.map(
            lambda img_id: self.reader.load_bbox(img_id, fmt=self.config.bbox_format, return_dict=True)
        )

        ecg_ids = self._train_test_split(train_ratio=self.config.train_ratio)
        self._df_data = self.reader._df_images[self.reader._df_images.ecg_id.isin(ecg_ids)]
        self._df_data["ecg_path"] = self._df_data["ecg_id"].apply(lambda x: self.reader._df_records.loc[x, "path"])
        self._df_data["dx"] = self._df_data["ecg_id"].apply(
            lambda x: self.reader.load_dx_ann(x, class_map={k: i for i, k in enumerate(self.config.classes)})
        )

        self.fdr = FastDataReader(self.reader, self._df_data, self.config, self.transform)

        if not self.lazy:
            self._load_all_data()

    def __len__(self) -> int:
        if self.cache is None:
            # self._load_all_data()
            return len(self.fdr)
        return self.cache["images"].shape[0]

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
            "digitization", "mask",  # digitization
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
            if self.config.predict_bbox:
                return naive_collate_fn([self[i] for i in range(*index.indices(len(self)))])
            return collate_fn([self[i] for i in range(*index.indices(len(self)))])
        row = self.df_data.loc[self.images[index]]
        # load the image
        image = self.reader.load_image(row.name)  # numpy array, of shape (H, W, C)

        # image_id (of `int` type) required by some object detection models
        # `str` type is not supported by pytorch
        # data = {"image_id": index, "image_name": row.name}
        data = {"image_id": index}

        A_kw = {}
        if self.config.predict_bbox:
            A_kw["bboxes"] = row["bbox_formatted"]["bbox"]
            A_kw["category_id"] = row["bbox_formatted"]["category_id"]
        A_out = self.transform(image=image, **A_kw)
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
            else:
                raise NotImplementedError(f"bbox format {self.config.bbox_format} is not implemented yet")
        if self.config.predict_mask:
            # load the mask
            raise NotImplementedError("mask prediction is not implemented yet")

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
