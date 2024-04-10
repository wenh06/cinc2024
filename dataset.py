"""
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Set, Union

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

        ecg_ids = self._train_test_split(train_ratio=self.config.train_ratio)
        self._df_data = self.reader._df_images[self.reader._df_images.ecg_id.isin(ecg_ids)]
        self._df_data["ecg_path"] = self._df_data["ecg_id"].apply(lambda x: self.reader._df_records.loc[x, "path"])
        self._df_data["dx"] = self._df_data["ecg_id"].apply(
            lambda x: self.reader.load_dx_ann(x, class_map={k: i for i, k in enumerate(self.config.classes)})
        )

        self.fdr = FastDataReader(self.reader, self._df_data, self.config)

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
            "bbox", "category_id", "area"  # object detection
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
    ) -> None:
        self.reader = reader
        self.df_data = df_data
        self.images = self.df_data.index.tolist()
        self.config = config
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
            return collate_fn([self[i] for i in range(*index.indices(len(self)))])
        row = self.df_data.loc[self.images[index]]
        # load the image
        image = self.reader.load_image(row.name)  # numpy array, of shape (H, W, C)
        # image_id (of `int` type) required by some object detection models
        data = {"image": image, "image_id": index}
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
            # keys are "bbox", "category_id", "area"
            raise NotImplementedError("data preparation for bbox prediction is not implemented yet")

        return data

    def extra_repr_keys(self) -> List[str]:
        return [
            "reader",
        ]


def collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, Union[torch.Tensor, List[np.ndarray]]]:
    out_tensors = {}
    for k in batch[0].keys():
        if k == "image":
            continue
        out_tensors[k] = torch.from_numpy(np.concatenate([[b[k]] for b in batch], axis=0))
    out_tensors["image"] = [b["image"] for b in batch]
    return out_tensors
