"""
"""

import gzip
import json
import multiprocessing as mp
import os
import re
import shutil
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import gdown
import numpy as np
import pandas as pd
from bib_lookup.utils import is_notebook
from PIL import Image, ImageDraw, ImageFont
from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.databases.base import DataBaseInfo, PhysioNetDataBase
from torch_ecg.utils.download import _unzip_file, http_get
from torch_ecg.utils.misc import add_docstring, remove_parameters_returns_from_docstring
from tqdm.auto import tqdm

from bbox import BBox, RotatedBBox
from cfg import BaseCfg, ModelCfg
from const import DATA_CACHE_DIR
from helper_code import cast_int_float_unknown, find_records
from prepare_image_data import find_files as find_images
from utils.ecg_image_generator import constants as ecg_img_gen_constants
from utils.ecg_image_generator import run_single_file
from utils.misc import get_record_list_recursive3, url_is_reachable

__all__ = [
    "CINC2024Reader",
]


_CINC2024_INFO = DataBaseInfo(
    title="""
    Digitization and Classification of ECG Images
    """,
    about="""
    1. Objectives: Turn images of 12-lead ECGs (scanned from paper) into waveforms (time series data) representing the same ECGs; Classify the ECGs (either from the image, or from the converted time series data) as normal or abnormal. ref. [1]_.
    2. Data: 12-lead ECG images (scanned or photographed from paper) and synthetic images generated from ECG time series data.
    3. The initial training set uses the waveforms and classes from the PTB-XL datase (ref. [2]_).
    4. The PTB-XL database is a large database of 21799 clinical 12-lead ECGs from 18869 patients of 10 second length collected with devices from Schiller AG over the course of nearly seven years between October 1989 and June 1996.
    5. The raw waveform data of the PTB-XL database was annotated by up to two cardiologists, who assigned potentially multiple ECG statements to each recording which were converted into a standardized set of SCP-ECG statements (scp_codes).
    6. The PTB-XL database contains 71 different ECG statements conforming to the SCP-ECG standard, including diagnostic, form, and rhythm statements.
    7. The waveform files of the PTB-XL database are stored in WaveForm DataBase (WFDB) format with 16 bit precision at a resolution of 1μV/LSB and a sampling frequency of 500Hz. A downsampled versions of the waveform data at a sampling frequency of 100Hz is also provided.
    8. In the metadata file (ptbxl_database.csv), each record of the PTB-XL database is identified by a unique ecg_id. The corresponding patient is encoded via patient_id. The paths to the original record (500 Hz) and a downsampled version of the record (100 Hz) are stored in `filename_hr` and `filename_lr`. The `report` field contains the diagnostic statements assigned to the record by the cardiologists. The `scp_codes` field contains the SCP-ECG statements assigned to the record which are formed as a dictionary with entries of the form `statement: likelihood`, where likelihood is set to 0 if unknown).
    9. The PTB-XL database underwent a 10-fold train-test splits (stored in the `strat_fold` field of the metadata file) obtained via stratified sampling while respecting patient assignments, i.e. all records of a particular patient were assigned to the same fold. Records in fold 9 and 10 underwent at least one human evaluation and are therefore of a particularly high label quality. It is proposed to use folds 1-8 as training set, fold 9 as validation set and fold 10 as test set.
    """,
    usage=[
        "Re-digitization of ECG images",
        "Classification of ECG images",
        "Waveform and lead names detection in ECG images",
    ],
    note="""
    """,
    issues="""
    """,
    references=[
        "https://moody-challenge.physionet.org/2024/",
        "https://physionet.org/content/ptb-xl/",
        "https://physionet.org/content/ptb-xl-plus/",
        "https://github.com/alphanumericslab/ecg-image-kit",
    ],
    doi=[
        "https://doi.org/10.1038/s41597-023-02153-8",  # PTB-XL+ paper
        "https://doi.org/10.1038/s41597-020-0495-6",  # PTB-XL paper
        "https://doi.org/10.13026/nqsf-pc74",  # PTB-XL+ physionet
        "https://doi.org/10.13026/6sec-a640",  # PTB-XL physionet
        "https://doi.org/10.48550/ARXIV.2307.01946",  # ecg-image-kit
    ],
)

_prepare_synthetic_images_docstring = """Prepare synthetic images from the ECG time series data.

        This function is modified from the functions

        - prepare_ptbxl_data.run
        - prepare_image_data.run
        - ecg_image_generator.run_single_file

        Parameters
        ----------
        output_folder : `path-like`, optional
            The output directory to store the synthetic images.
            If not specified, the synthetic images will be stored in the default directory.
        fs : {100, 500}, default 100
            The sampling frequency of the ECG time series data to be used.
            If not specified (``None``), the default (self.fs or 500) sampling frequency will be used.
        force_recompute : bool, default False
            Whether to force recompute the synthetic images regardless of the existence of the output directory.
        parallel : bool, default False
            Whether to use multiprocessing to generate the synthetic images.
        kwargs : dict, optional
            Extra key word arguments passed to the image generator.

        .. note::

            To use configurations other than the default configuration, pass the configurations as key word arguments.
            self.__gen_img_extra_configs__ contains a list of extra configurations stored in the configuration directory.
            It can be used via `**self.__gen_img_extra_configs__[i]`, where `i` is the index of the configuration,
            passed as key word arguments.

        """


@add_docstring(_CINC2024_INFO.format_database_docstring(), mode="prepend")
class CINC2024Reader(PhysioNetDataBase):
    """
    Parameters
    ----------
    db_dir : `path-like`
        Storage path of the database.
        If not specified, data will be fetched from Physionet.
    fs : int, default 500
        (Re-)sampling frequency of the recordings.
    working_dir : `path-like`, optional
        Working directory, to store intermediate files and log files.
    verbose : int, default 1
        Level of logging verbosity.
    kwargs : dict, optional
        Auxilliary key word arguments, including
        - synthetic_images_dir : `path-like`, optional
            The directory to store the synthetic images.
        - bbox_class_names : list, optional
            The class names of the bounding boxes.

    """

    __name__ = "CINC2024Reader"
    __aux_files__ = {
        "metadata_file": "ptbxl_database.csv",
        "scp_statements_file": "scp_statements.csv",
        "12sl_statements_file": "12sl_statements.csv",
        "12sl_mapping_file": "12slv23ToSNOMED.csv",
    }
    __aux_files_url__ = {
        "12sl_statements_file": "https://physionet.org/files/ptb-xl-plus/1.0.1/labels/12sl_statements.csv",
        "12sl_mapping_file": "https://physionet.org/files/ptb-xl-plus/1.0.1/labels/mapping/12slv23ToSNOMED.csv",
    }
    __100Hz_dir__ = "records100"
    __500Hz_dir__ = "records500"
    __synthetic_images_dir__ = "synthetic_images"
    __gen_img_default_config__ = CFG(
        json.loads((ecg_img_gen_constants.CONFIG_DIR / "ecg-image-gen-default-config.json").read_text())
    )
    __gen_img_extra_configs__ = [
        CFG(json.loads((ecg_img_gen_constants.CONFIG_DIR / cfg_file).read_text()))
        for cfg_file in ecg_img_gen_constants.CONFIG_DIR.glob("*.json")
        if cfg_file.name != "ecg-image-gen-default-config.json"
    ]
    __synthetic_images_url__ = {
        "full": None,
        "full-alt": None,
        "subset": "https://drive.google.com/u/0/uc?id=1plrp_G6_1nlpMslaOFEz9cEozYwLovmu",
        "subset-alt": "https://deep-psp.tech/Data/ptb-xl-synthetic-images-subset-tiny.zip",
    }
    __acute_mi_statements__ = set([821, 822, 823, 827, 829, 902, 903, 904, 963, 964, 965, 966, 967, 968])

    def __init__(
        self,
        db_dir: Union[str, bytes, os.PathLike],
        fs: int = 500,
        working_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> None:
        if db_dir is None:
            db_dir = Path(DATA_CACHE_DIR)
        super().__init__(db_name="ptb-xl", db_dir=db_dir, working_dir=working_dir, fs=fs, verbose=verbose, **kwargs)
        self.data_ext = "dat"
        self.header_ext = "hea"
        self.img_ext = ecg_img_gen_constants.save_img_ext.strip(".")
        self.bbox_ext = "npz"
        self.bbox_min_size = 2 * max(kwargs.pop("bbox_min_size", 12) // 2, 1)  # pixels
        self.record_pattern = "[\\d]{5}_[lh]r"

        assert os.access(self.db_dir, os.W_OK) or os.access(
            self.working_dir, os.W_OK
        ), "neither db_dir nor working_dir is writable"

        self.src_datetime_fmt = "%Y-%m-%d %H:%M:%S"
        self.dst_datetime_fmt = "%H:%M:%S %d/%m/%Y"
        self.gen_img_default_fs = 500
        self.gen_img_pattern = "[\\d]{5}_[lh]r-[\\d]+.png"

        self._synthetic_images_dir = kwargs.pop("synthetic_images_dir", None)
        self.__config = CFG(BaseCfg.copy())
        self.__config.update(kwargs)
        self.__bbox_class_names = kwargs.pop("bbox_class_names", ModelCfg.object_detection.class_names)

        self._df_records = None
        self._df_metadata = None
        self._df_scp_statements = None
        self._subclass_to_superclass = {}
        self._df_images = None
        self._df_12sl_statements = None
        self._df_12sl_mapping = None
        self._acute_mi_classes = set()
        self._all_records = None
        self._all_subjects = None
        self._all_images = None
        self._gen_img_config_current = None
        self._ls_rec()

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in a dataframe.
        """
        # locate the true database directory using the metadata file
        try:
            metadata_file = list(self.db_dir.rglob(self.__aux_files__["metadata_file"]))[0]
        except IndexError:
            self.logger.info(
                f"""metadata file {self.__aux_files__["metadata_file"]} not found in {self.db_dir}. """
                "Download the database first using the `download` method."
            )
            self._df_records = pd.DataFrame()
            self._df_metadata = pd.DataFrame()
            self._df_scp_statements = pd.DataFrame()
            self._df_images = pd.DataFrame()
            self._df_12sl_statements = pd.DataFrame()
            self._df_12sl_mapping = pd.DataFrame()
            self._all_records = []
            self._all_subjects = []
            self._all_images = []
            self._create_synthetic_images_dir()
            return
        self.db_dir = metadata_file.parent.resolve()
        assert (
            self.db_dir / self.__aux_files__["scp_statements_file"]
        ).exists(), f"scp_statements file not found in {self.db_dir}"

        if not (self.db_dir / self.__aux_files__["12sl_statements_file"]).exists():
            self.logger.info(
                f"""12sl_statements file {self.__aux_files__["12sl_statements_file"]} not found in {self.db_dir}. """
                "Download the database first using the `download_aux_files` method."
            )
            self._df_12sl_statements = pd.DataFrame()
        else:
            self._df_12sl_statements = pd.read_csv(self.db_dir / self.__aux_files__["12sl_statements_file"])
            self._df_12sl_statements["ecg_id"] = self._df_12sl_statements["ecg_id"].apply(lambda x: f"{x:05d}")
            self._df_12sl_statements.set_index("ecg_id", inplace=True)
        if not (self.db_dir / self.__aux_files__["12sl_mapping_file"]).exists():
            self.logger.info(
                f"""12sl_mapping file {self.__aux_files__["12sl_mapping_file"]} not found in {self.db_dir}. """
                "Download the database first using the `download_aux_files` method."
            )
            self._df_12sl_mapping = pd.DataFrame()
        else:
            self._df_12sl_mapping = pd.read_csv(self.db_dir / self.__aux_files__["12sl_mapping_file"])
            # drop nan rows and columns
            self._df_12sl_mapping.dropna(axis=0, how="all", inplace=True)
            self._df_12sl_mapping.dropna(axis=1, how="all", inplace=True)
            # self._df_12sl_mapping["StatementNumber"] = self._df_12sl_mapping["StatementNumber"].astype(int)
            self._df_12sl_mapping.set_index("StatementNumber", inplace=True)

        self._create_synthetic_images_dir()

        # read metadata file and scp_statements file
        self._df_metadata = pd.read_csv(self.db_dir / self.__aux_files__["metadata_file"])
        self._df_metadata["scp_codes"] = self._df_metadata["scp_codes"].apply(lambda x: literal_eval(x))
        self._df_metadata["ecg_id"] = self._df_metadata["ecg_id"].apply(lambda x: f"{x:05d}")
        self._df_metadata.set_index("ecg_id", inplace=True)
        self._df_metadata["patient_id"] = self._df_metadata["patient_id"].astype(int)
        self._df_scp_statements = pd.read_csv(self.db_dir / self.__aux_files__["scp_statements_file"], index_col=0)

        for row_idx, row in self._df_scp_statements.iterrows():
            if row["diagnostic"] == 1:
                self._subclass_to_superclass[row_idx] = row["diagnostic_class"]
        # # Map the PTB-XL classes to superclasses.
        self._df_metadata["diagnostic_superclass"] = self._df_metadata["scp_codes"].apply(
            lambda x: sorted(
                set([self._subclass_to_superclass[subclass] for subclass in x if subclass in self._subclass_to_superclass])
            )
        )
        self._df_metadata["diagnostic_superclass_augmented"] = self._df_metadata["diagnostic_superclass"].apply(
            lambda x: x if len(x) > 0 else [self.config.default_class]
        )

        # Map the 12SL classes to the PTB-XL classes for the following acute myocardial infarction (MI) classes;
        # PTB-XL does not include a separate acute MI class.

        for statement in self.__acute_mi_statements__:
            if statement in self._df_12sl_mapping.index:
                self._acute_mi_classes.add(self._df_12sl_mapping.loc[statement]["Acronym"])

        # Obtain the labels for the challenge
        self._df_metadata["labels"] = self._df_metadata.apply(
            lambda row: get_record_labels(row, self._df_12sl_statements, self._acute_mi_classes), axis=1
        )
        self._df_metadata["labels_augmented"] = self._df_metadata["labels"].apply(
            lambda x: x if len(x) > 0 else [self.config.default_class]
        )

        # self._df_images = pd.DataFrame({"image": find_images(str(self._synthetic_images_dir), [".png", ".jpg", ".jpeg"])})
        # self._df_images["path"] = self._df_images["image"].apply(lambda x: self._synthetic_images_dir / x)
        self._df_images = pd.DataFrame(
            {
                "path": get_record_list_recursive3(
                    self._synthetic_images_dir,
                    rec_patterns=".+\\.(png|jpg|jpeg)$",
                    relative=False,
                    with_suffix=True,
                ),
            }
        )
        if not self._df_images.empty:
            self._df_images["path"] = self._df_images["path"].apply(lambda x: Path(x))
            self._df_images["image"] = self._df_images["path"].apply(lambda x: x.stem)
            self._df_images["image_header"] = self._df_images.apply(
                lambda row: row["path"].parent / f"""{row["image"].split("-")[0]}.{self.header_ext}""", axis=1
            )
            self._df_images["ecg_id"] = self._df_images["image"].apply(lambda x: x[:5])
            self._df_images["patient_id"] = self._df_images["ecg_id"].apply(lambda x: self._df_metadata.loc[x, "patient_id"])
            self._df_images["strat_fold"] = self._df_images["ecg_id"].apply(lambda x: self._df_metadata.loc[x, "strat_fold"])
            self._df_images["lead_bbox_file"] = self._df_images["path"].apply(
                lambda x: x.parent / ecg_img_gen_constants.lead_bounding_box_dir_name / f"{x.stem}.{self.bbox_ext}"
            )
            self._df_images["lead_bbox_file"] = self._df_images["lead_bbox_file"].apply(lambda x: x if x.exists() else None)
            self._df_images["text_bbox_file"] = self._df_images["path"].apply(
                lambda x: x.parent / ecg_img_gen_constants.text_bounding_box_dir_name / f"{x.stem}.{self.bbox_ext}"
            )
            self._df_images["text_bbox_file"] = self._df_images["text_bbox_file"].apply(lambda x: x if x.exists() else None)
            self._df_images.set_index("image", inplace=True)
        else:
            # fmt: off
            self._df_images = pd.DataFrame(
                columns=[
                    "path", "image", "image_header", "ecg_id", "patient_id", "strat_fold",
                    "lead_bbox_file", "text_bbox_file", "bbox", "ROI",
                ]
            )
            # fmt: on
            self._df_images.set_index("image", inplace=True)
            self.logger.warning(f"no synthetic images found in {self._synthetic_images_dir}")

        self._df_records = self._df_metadata.copy()
        if self.fs == 100:
            self._df_records["path"] = self._df_records["filename_lr"].apply(lambda x: self.db_dir / x)
        else:
            self._df_records["path"] = self._df_records["filename_hr"].apply(lambda x: self.db_dir / x)
        # keep only records that exist
        self._df_records = self._df_records[
            self._df_records["path"].apply(lambda x: x.with_suffix(f".{self.data_ext}").exists())
        ]
        if self._subsample is not None:
            size = min(
                len(self._df_records),
                max(1, int(round(self._subsample * len(self._df_records)))),
            )
            self.logger.debug(f"subsample `{size}` records from `{len(self._df_records)}`")
            self._df_records = self._df_records.sample(n=size, random_state=DEFAULTS.SEED, replace=False)
            self._df_images = self._df_images[self._df_images["ecg_id"].isin(self._df_records.index)]

        self._all_records = self._df_records.index.tolist()
        self._all_subjects = self._df_records["patient_id"].unique().tolist()
        self._all_images = self._df_images.index.tolist()

        # load the bounding boxes into self._df_images
        if not self._df_images.empty:
            self._df_images["bbox"] = None
            self._df_images["ROI"] = None
            # self._df_images["metadata"] = None
            with tqdm(
                total=len(self._df_images), desc="Loading bounding boxes", dynamic_ncols=True, mininterval=1, leave=True
            ) as pbar:
                for img_id in self._all_images:
                    self._df_images.at[img_id, "bbox"] = self._load_bbox(img_id)
                    # self._df_images.at[img_id, "metadata"] = load_gzip_json(self._df_images.loc[img_id, "path"].with_suffix(".json.gz"))
                    # via bounding boxes get the image-level region of interest
                    if len(self._df_images.loc[img_id, "bbox"]) > 0:
                        self._df_images.at[img_id, "ROI"] = CINC2024Reader.get_roi(
                            self._df_images.loc[img_id, "bbox"], padding=0
                        )
                    pbar.update(1)

    def _create_synthetic_images_dir(self) -> None:
        """Create the directory to store the synthetic images."""
        if self._synthetic_images_dir is not None:
            if Path(self._synthetic_images_dir).exists():
                if not os.access(self._synthetic_images_dir, os.W_OK):
                    self.logger.warning(f"synthetic images directory `{self._synthetic_images_dir}` not writable.")
                # else: already exists and writable
            else:
                try:
                    self._synthetic_images_dir.mkdir(parents=True, exist_ok=True)
                except PermissionError:  # No write permission
                    self.logger.warning(f"failed to create synthetic images directory `{self._synthetic_images_dir}`.")
                    self._synthetic_images_dir = None
        if self._synthetic_images_dir is None:
            # by default, use the data directory
            self._synthetic_images_dir = self.db_dir / self.__synthetic_images_dir__
            if os.access(self.db_dir, os.W_OK):
                self._synthetic_images_dir.mkdir(parents=True, exist_ok=True)
            else:  # self.db_dir is not writable
                if (
                    self._synthetic_images_dir.exists()
                    and len(get_record_list_recursive3(self._synthetic_images_dir, rec_patterns=".+\\.(png|jpg|jpeg)$")) > 0
                ):
                    # already exists and not empty, OK
                    pass
                elif not self._synthetic_images_dir.exists():
                    # does not exist, and not writable, switch to working directory
                    # note that at least one of db_dir and working_dir should be writable
                    self._synthetic_images_dir = self.working_dir / self.__synthetic_images_dir__
                else:  # exists but empty, raise warning
                    self.logger.warning(f"synthetic images directory `{self._synthetic_images_dir}` not writable.")
        self._synthetic_images_dir = Path(self._synthetic_images_dir).expanduser().resolve()
        self._synthetic_images_dir.mkdir(parents=True, exist_ok=True)
        if (self._synthetic_images_dir / "ecg_img_gen_config.json").exists():
            self._gen_img_config_current = CFG(json.loads((self._synthetic_images_dir / "ecg_img_gen_config.json").read_text()))
        else:
            self._gen_img_config_current = {}
        self.logger.info(f"Synthetic images directory set to: {self._synthetic_images_dir}")

    def get_record_images(self, rec: Union[str, int]) -> List[str]:
        """Get the image IDs of a record.

        Parameters
        ----------
        rec : str or int
            The record name (ecg_id) or the index of the record.

        Returns
        -------
        list of str
            The images of the record.

        """
        if isinstance(rec, int):
            rec = self._all_records[rec]
        return self._df_images[self._df_images["ecg_id"] == rec].index.tolist()

    def load_image(
        self, img: Union[str, int], fmt: str = "np", roi_only: bool = False, roi_padding: float = 0.1
    ) -> Union[np.ndarray, Image.Image]:
        """Load the image of a record.

        Parameters
        ----------
        img : str or int
            The image name or the index of the image.
        fmt : {"np", "pil"}, default "np"
            The format of the image to be returned, case insensitive.
            If is "np", the image will be returned as a numpy array.
            If is "pil", the image will be returned as a PIL image.

        Returns
        -------
        numpy.ndarray or PIL.Image
            The image of an ECG record, of shape ``(H, W, C)``.

        .. note::

            The image is converted to RGB format if it is not (e.g. RGBA format).
            For using PyTorch models, the image should first be transformed to shape ``(C, H, W)``.
            The image processor classes from the transformers library automatically check the shape of the input image
            and convert it to the correct shape.
            For using other frameworks (e.g. torchvision, timm), the image should be converted to the correct shape manually.

        """
        if isinstance(img, int):
            img_id = self._all_images[img]
        else:
            img_id = img
        img_path = self._df_images.loc[img_id, "path"]
        img = Image.open(img_path).convert("RGB")  # png images are RGBA
        if roi_only:
            xmin, ymin, xmax, ymax = self._df_images.loc[img_id, "ROI"]  # no padding
            if roi_padding > 0:
                width, height = xmax - xmin, ymax - ymin
                pad_x = int(width * roi_padding)
                pad_y = int(height * roi_padding)
                xmin = max(0, xmin - pad_x)
                ymin = max(0, ymin - pad_y)
                xmax = min(img.size[0], xmax + pad_x)
                ymax = min(img.size[1], ymax + pad_y)
            img = img.crop((xmin, ymin, xmax, ymax))
        if fmt.lower() == "np":
            return np.asarray(img)
        elif fmt.lower() == "pil":
            return img
        else:
            raise ValueError(f"Invalid return format `{fmt}`")

    def load_image_metadata(self, img: Union[str, int], items: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """Load the metadata of an image.

        Parameters
        ----------
        img : str or int
            The image name or the index of the image.
        items : str or list of str, optional
            The items to load.

        Returns
        -------
        metadata : dict
            The metadata of the image.

        """
        if isinstance(img, int):
            img = self._all_images[img]
        metadata_file = self._df_images.loc[img, "path"].with_suffix(".json.gz")
        if not metadata_file.exists():
            return {}
        metadata = load_gzip_json(metadata_file)
        if items is None:
            return metadata
        if isinstance(items, str):
            items = [items]
        return {item: metadata[item] for item in items if item in metadata}

    def view_image(
        self,
        img: Union[str, int],
        with_lead_bbox: bool = True,
        with_text_bbox: bool = True,
        with_matched_bbox: bool = True,
        with_plotted_pixels: bool = True,
        roi_only: bool = False,
        roi_padding: float = 0.2,
    ) -> Optional[Image.Image]:
        """View the image of a record.

        Parameters
        ----------
        img : str or int
            The image name or the index of the image.
        with_lead_bbox : bool, default True
            Whether to show the bounding boxes of the waveforms.
        with_text_bbox : bool, default True
            Whether to show the bounding boxes of the text.
        with_matched_bbox : bool, default True
            Whether to show the matched lead names for the waveforms bounding boxes.
        with_plotted_pixels : bool, default False
            Whether to plot the plotted pixels of the ECG waveforms.
        roi_only : bool, default False
            Whether to show only the region of interest.
        roi_padding : float, default 0.1
            The padding ratio of the region of interest.

        Returns
        -------
        PIL.Image
            The image of an ECG record, of shape ``(H, W, C)`` if in notebook.
            Otherwise, the image will be shown in a new window instead of returning,
            typically using the default image viewer of the operating system.

        """
        if isinstance(img, int):
            img = self._all_images[img]
        ecg_image = self.load_image(img, fmt="pil")
        bbox = self.load_bbox(img)
        if with_lead_bbox:
            lead_bbox = [wb["bbox"] for wb in bbox if wb["category_name"] == "waveform"]
            if len(lead_bbox) > 0:
                # plot the bounding boxes in the image
                draw = ImageDraw.Draw(ecg_image)
                for x1, y1, x2, y2 in lead_bbox:
                    x2, y2 = x1 + x2, y1 + y2  # COCO format to PIL format
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            else:
                lead_bbox = None
                # warning already logged in self.load_bbox
                # self.logger.warning(f"no lead bounding box found for image `{img}`.")
        else:
            lead_bbox = None
        if with_text_bbox:
            text_bbox = [wb["bbox"] for wb in bbox if wb["category_name"] != "waveform"]
            if len(text_bbox) > 0:
                # plot the bounding boxes in the image
                draw = ImageDraw.Draw(ecg_image)
                for x1, y1, x2, y2 in text_bbox:
                    x2, y2 = x1 + x2, y1 + y2  # COCO format to PIL format
                    draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            else:
                text_bbox = None
                # warning already logged in self.load_bbox
                # self.logger.warning(f"no text bounding box found for image `{img}`.")
        else:
            text_bbox = None
        if with_matched_bbox:
            for wb in CINC2024Reader.match_bbox(bbox):
                # wb is a dict with keys "bbox" and "lead_name"
                # "bbox" is in COCO format [x, y, width, height]
                font = ImageFont.truetype("arial.ttf", int(min(ecg_image.size) * 0.025))
                draw.text((wb["bbox"][0], wb["bbox"][1]), wb["lead_name"], fill="red", font=font, anchor="lb")
        if with_plotted_pixels:
            image_metadata = self.load_image_metadata(img)
            if image_metadata:  # not empty
                for leads_dict in image_metadata["leads"]:
                    pp_arr = leads_dict["plotted_pixels"]
                    if pp_arr:
                        pp_arr = np.array(pp_arr)
                        draw = ImageDraw.Draw(ecg_image, "RGBA")
                        for y, x in pp_arr:
                            # green, semi-transparent
                            draw.point((x, y), fill=(0, 255, 0, 128))
        if roi_only:
            xmin, ymin, xmax, ymax = self._df_images.loc[img, "ROI"]  # no padding
            if roi_padding > 0:
                width, height = xmax - xmin, ymax - ymin
                pad_x = int(width * roi_padding)
                pad_y = int(height * roi_padding)
                xmin = max(0, xmin - pad_x)
                ymin = max(0, ymin - pad_y)
                xmax = min(ecg_image.size[0], xmax + pad_x)
                ymax = min(ecg_image.size[1], ymax + pad_y)
            ecg_image = ecg_image.crop((xmin, ymin, xmax, ymax))
        # draw the dx annotations in upper right corner
        ecg_id = self._df_images.loc[img, "ecg_id"]
        dx_ann = self.load_dx_ann(ecg_id, class_map=False)
        dx_ann_str = ", ".join(dx_ann)
        draw = ImageDraw.Draw(ecg_image)
        font = ImageFont.truetype("arial.ttf", int(min(ecg_image.size) * 0.05))
        draw.text((ecg_image.size[0] - 20, 20), dx_ann_str, align="right", anchor="rt", fill="red", font=font)
        # if is jupyter notebook, show the image inline
        if is_notebook():
            return ecg_image
        ecg_image.show()

    def load_metadata(self, rec: Union[str, int], items: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """Load the metadata of a record.

        Parameters
        ----------
        rec : str or int
            The record name (ecg_id) or the index of the record.
        items : str or list of str, optional
            The items to load.

        Returns
        -------
        metadata : dict
            The metadata of the record.

        """
        if isinstance(rec, int):
            rec = self._all_records[rec]
        if items is None:
            return self._df_metadata.loc[rec].to_dict()
        if isinstance(items, str):
            items = [items]
        return self._df_metadata.loc[rec, items].to_dict()

    def load_ann(self, rec: Union[str, int], with_interpretation: bool = False) -> Dict[str, Union[float, Dict[str, Any]]]:
        """Load the annotation (the "scp_codes" field) of a record.

        Parameters
        ----------
        rec : str or int
            The record name (ecg_id) or the index of the record.
        with_interpretation : bool, default False
            Whether to include the interpretation of the statement.

        Returns
        -------
        ann : dict
            The annotation of the record, of the form ``{statement: likelihood}``.
            If ``with_interpretation`` is ``True``, the form is
            ``{statement: {"likelihood": likelihood, ...}}``,
            where ``...`` are other information of the statement.

        """
        ann = self.load_metadata(rec)["scp_codes"]
        if with_interpretation:
            for statement, likelihood in ann.items():
                ann[statement] = {"likelihood": likelihood}
                ann[statement].update(self._df_scp_statements.loc[statement].to_dict())
        return ann

    def load_dx_ann(
        self, rec: Union[str, int], class_map: Optional[Union[bool, Dict[str, int]]] = None, augmented: bool = True
    ) -> Union[List[str], List[int]]:
        """Load the challenge labels (Dx classification labels) of a record.

        Parameters
        ----------
        rec : str or int
            The record name (ecg_id) or the index of the record.
        class_map : dict or bool, optional
            The mapping from the statement to the binary class.
            If is ``None``, the default mapping will be used.
            If is ``False``, the statement will be returned.
        augmented : bool, default True
            Whether to use the augmented annotations ("OTHER" included).

        Returns
        -------
        list of str or list of int
            The Dx annotations of the record.

        """
        key = "labels_augmented" if augmented else "labels"
        dx = self.load_metadata(rec)[key]
        if class_map is None:
            dx = [self.config.classes.index(dx) for dx in dx]
        elif class_map is not False:
            dx = [class_map[dx] for dx in dx]
        return dx

    def load_diagnostic_superclasses(
        self, rec: Union[str, int], class_map: Optional[Dict[str, int]] = None, augmented: bool = True
    ) -> Union[List[str], List[int]]:
        """Load the diagnostic superclasses annotations of a record.

        Parameters
        ----------
        rec : str or int
            The record name (ecg_id) or the index of the record.
        class_map : dict or bool, optional
            The mapping from the statement to the binary class.
            If is ``None``, the statements will be returned.
        augmented : bool, default True
            Whether to use the augmented diagnostic superclass annotations ("OTHER" included).

        Returns
        -------
        list of str or list of int
            The diagnostic superclasses annotations of the record.

        """
        key = "diagnostic_superclass_augmented" if augmented else "diagnostic_superclass"
        diagnostic_superclasses = self.load_metadata(rec)[key]  # list of superclasses names (str)
        if class_map is not None:
            diagnostic_superclasses = [class_map[diag_sc] for diag_sc in diagnostic_superclasses]
        return diagnostic_superclasses

    def load_bbox(
        self,
        img: Union[str, int],
        bbox_type: Optional[str] = None,
        fmt: str = "coco",
        return_dict: bool = False,
        clip: bool = True,
        min_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load the bounding boxes of the image.

        Parameters
        ----------
        img : str or int
            The image name or the index of the image.
        bbox_type : {"lead", "text"}, optional
            The type of the bounding box.
            If is ``None``, both types of bounding boxes will be loaded.
        fmt : {"coco", "pascal_voc", "yolo", "albumentations"}, default "coco"
            The format of the bounding boxes to be returned.
            - If is "coco", the bounding boxes will be in COCO format:
              ``[x, y, width, height]``.
            - If is "pascal_voc", the bounding boxes will be in Pascal VOC format:
              ``[xmin, ymin, xmax, ymax]``.
            - If is "yolo", the bounding boxes will be in YOLO format:
              ``[x_center, y_center, width, height]`` (normalized to [0, 1]).
            - If is "albumentations", the bounding boxes will be in Albumentations format:
                ``[x_min, y_min, x_max, y_max]`` (normalized to [0, 1]).
        return_dict: bool, default False
            If False, return the bounding boxes as a list of dictionaries.
            If True, return the bounding boxes (and category_id, etc.) as a dictionary of lists.
            This is useful for feeding data into augmentation libraries (e.g. `albumentations`).
        clip: bool, default True
            Whether to clip the bounding boxes to the image boundaries.
        min_size: int, optional
            The minimum size of the bounding box.
            If is None, the default value of the reader will be used.

        Returns
        -------
        bbox : list of dict
            The bounding boxes of the ECG waveforms and (or) lead names in the image.

        .. note::

        Official phase:

            The bounding boxes generated by the `ecg-image-generator` module are obtained
            using :func:`matplotlib.axes.Axes.get_window_extent`, and converted to the format of
            `[[y0, x0], [y1, x1], [y2, x2], [y3, x3]]`, where the origin is at the upper left
            corner of the axes, which is the same as the PIL coordinate system, and all the
            common image processing libraries (e.g. OpenCV, torchvision).

            This format has the advantage of representing the rotated bounding boxes other than
            the normal axis-aligned bounding boxes. This can be useful for the rotated object
            detection models.

        Unofficial phase:

            The bounding boxes generated by the `ecg-image-generator` module are obtained
            using :func:`matplotlib.axes.Axes.get_window_extent`, which is in the coordinate system of the axes,
            namely the origin is at the lower left corner of the axes.
            While the PIL coordinate system, and all the common image processing libraries (e.g. OpenCV, torchvision),
            and also the COCO, Pascal VOC, YOLO formats, have the origin at the upper left corner of the image.

        """
        if isinstance(img, int):
            img = self._all_images[img]
        if min_size is None:
            min_size = self.bbox_min_size
        if clip and min_size == self.bbox_min_size:
            bbox = self._df_images.loc[img, "bbox"]
        else:
            bbox = self._load_bbox(img, clip=clip, min_size=min_size)
        if bbox_type == "lead":
            bbox = [wb for wb in bbox if wb.category_name == "waveform"]
        elif bbox_type == "text":
            bbox = [wb for wb in bbox if wb.category_name != "waveform"]
        if len(bbox) == 0:
            self.logger.warning(f"no bounding box found for image `{img}`.")
        if fmt.lower() == "coco":
            bbox = [wb.to_coco_format() for wb in bbox]
        elif fmt.lower() == "pascal_voc":
            bbox = [wb.to_pascal_voc_format() for wb in bbox]
        elif fmt.lower() == "yolo":
            bbox = [wb.to_yolo_format() for wb in bbox]
        elif fmt.lower() == "albumentations":
            bbox = [wb.to_albumentations_format() for wb in bbox]
        else:
            raise ValueError(f"Invalid format `{fmt}`")
        if return_dict:
            keys = bbox[0].keys()
            bbox = {key: [b[key] for b in bbox] for key in keys}
        return bbox

    @add_docstring(remove_parameters_returns_from_docstring(load_bbox.__doc__, parameters=["bbox_type", "fmt", "return_dict"]))
    def _load_bbox(
        self, img: Union[str, int], clip: bool = True, min_size: Optional[int] = None
    ) -> List[Union[BBox, RotatedBBox]]:
        if isinstance(img, int):
            img = self._all_images[img]
        img_path = self._df_images.loc[img, "path"]
        if min_size is None:
            min_size = self.bbox_min_size

        # get image width and height without loading the image
        pil_img = Image.open(img_path)
        img_width, img_height = pil_img.size
        pil_img.close()

        bbox = []
        # lead_bbox has the format [[y0, x0], [y1, x1], [y2, x2], [y3, x3]]
        lead_bbox_file = self._df_images.loc[img, "lead_bbox_file"]
        if lead_bbox_file is not None:
            # lead_bbox = np.loadtxt(lead_bbox_file, delimiter=",", dtype=int)
            lead_bbox = np.load(lead_bbox_file)["lead_bbox"].astype(int)
            for l_bbox in lead_bbox:
                # convert to the PIL coordinate system
                x1, y1, x2, y2 = l_bbox[:, 1].min(), l_bbox[:, 0].min(), l_bbox[:, 1].max(), l_bbox[:, 0].max()
                if clip:
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)
                if (x2 - x1) < min_size:
                    if x1 < min_size / 2:
                        x1, x2 = 0, min_size
                    elif x2 > img_width - min_size / 2:
                        x1, x2 = img_width - min_size, img_width
                    else:
                        pad_left = (min_size - (x2 - x1)) // 2
                        pad_right = min_size - (x2 - x1) - pad_left
                        x1, x2 = x1 - pad_left, x2 + pad_right
                if (y2 - y1) < min_size:
                    if y1 < min_size / 2:
                        y1, y2 = 0, min_size
                    elif y2 > img_height - min_size / 2:
                        y1, y2 = img_height - min_size, img_height
                    else:
                        pad_top = (min_size - (y2 - y1)) // 2
                        pad_bottom = min_size - (y2 - y1) - pad_top
                        y1, y2 = y1 - pad_top, y2 + pad_bottom
                bbox.append(
                    BBox(
                        xmin=x1,
                        ymin=y1,
                        width=x2 - x1,
                        height=y2 - y1,
                        category_name="waveform",
                        category_id=self.__bbox_class_names.index("waveform"),
                        img_height=img_height,
                        img_width=img_width,
                    )
                )
        # text_bbox has the format [[y0, x0], [y1, x1], [y2, x2], [y3, x3]]
        text_bbox_file = self._df_images.loc[img, "text_bbox_file"]
        if text_bbox_file is not None:
            text_bbox = np.load(text_bbox_file)
            text_bbox, lead_name = text_bbox["text_bbox"].astype(int), text_bbox["lead_name"]
            for t_bbox, ln in zip(text_bbox, lead_name):
                # convert to the PIL coordinate system
                x1, y1, x2, y2 = t_bbox[:, 1].min(), t_bbox[:, 0].min(), t_bbox[:, 1].max(), t_bbox[:, 0].max()
                if clip:
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)
                bbox.append(
                    BBox(
                        xmin=x1,
                        ymin=y1,
                        width=x2 - x1,
                        height=y2 - y1,
                        category_name=ln,
                        category_id=self.__bbox_class_names.index(ln),
                        img_height=img_height,
                        img_width=img_width,
                    )
                )
        return bbox

    def load_header(self, rec_or_img: Union[str, int], source: Optional[str] = "image") -> str:
        """Load the header of a record or an image.

        Parameters
        ----------
        rec_or_img : str or int
            The record name (ecg_id) or the index of the record.
        source : {"record", "image"}, optional
            The source of the header.
            If is ``record``, the header of the record will be loaded.
            If is ``image``, the header of the image will be loaded.

        Returns
        -------
        header : str
            The header of the record or the image.

        """
        if source == "image":
            if isinstance(rec_or_img, int):
                rec_or_img = self._all_images[rec_or_img]
            return self._df_images.loc[rec_or_img, "image_header"].read_text()
        else:
            if isinstance(rec_or_img, int):
                rec_or_img = self._all_records[rec_or_img]
            return self.load_metadata(rec_or_img)["header"]

    @add_docstring(remove_parameters_returns_from_docstring(_prepare_synthetic_images_docstring, parameters=["parallel"]))
    def _prepare_synthetic_images(
        self,
        output_folder: Optional[Union[str, bytes, os.PathLike]] = None,
        fs: int = 100,
        force_recompute: bool = False,
        **kwargs: Any,
    ) -> None:
        if output_folder is None:
            output_folder = self._synthetic_images_dir
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        if fs is None:
            if self.fs in [100, 500]:
                fs = self.fs
            else:
                self.logger.warning(f"invalid fs `{fs}`, use default fs {self.gen_img_default_fs} instead.")
                fs = self.gen_img_default_fs
        input_folder = self.db_dir / self.__500Hz_dir__ if fs == 500 else self.db_dir / self.__100Hz_dir__

        ecg_img_gen_config = CFG(self.__gen_img_default_config__.copy())
        ecg_img_gen_config.update(kwargs)
        # NOTE: to fill "input_file", "header_file" and "output_directory" in the config for generating for each record

        records = find_records(str(input_folder))
        num_ecg_img_gen = 0

        # Update the header files and create the synthetic images.
        # TODO: find out why the signal files are copied to the output directory, which is NOT the expected behavior.
        with tqdm(
            records,
            total=len(records),
            dynamic_ncols=True,
            mininterval=1.0,
            desc="Generating synthetic images",
        ) as pbar:
            for record in pbar:
                # Extract the demographics data.
                record_dir, record_basename = os.path.split(record)
                ecg_id = record_basename.split("_")[0]
                row = self._df_metadata.loc[ecg_id]

                recording_date_string = row["recording_date"]
                recording_date = datetime.strptime(recording_date_string, self.src_datetime_fmt)
                recording_date_string = recording_date.strftime(self.dst_datetime_fmt)

                age = row["age"]
                age = cast_int_float_unknown(age)

                sex = row["sex"]
                if sex == 0:
                    sex = "Male"
                elif sex == 1:
                    sex = "Female"
                else:
                    sex = "Unknown"

                height = row["height"]
                height = cast_int_float_unknown(height)

                weight = row["weight"]
                weight = cast_int_float_unknown(weight)

                # Extract the diagnostic superclasses.
                scp_codes = [scp_code for scp_code, likelihood in row["scp_codes"].items() if likelihood > 0]
                superclasses = row["diagnostic_superclass"]

                if ecg_id in self._df_12sl_statements.index:
                    sl_codes = self._df_12sl_statements.loc[ecg_id]["statements"]
                else:
                    sl_codes = list()

                labels = list()
                if "NORM" in superclasses:
                    labels.append("NORM")
                if any(c in sl_codes for c in self._acute_mi_classes):
                    labels.append("Acute MI")
                if "MI" in superclasses and not any(c in sl_codes for c in self._acute_mi_classes):
                    labels.append("Old MI")
                if "STTC" in superclasses:
                    labels.append("STTC")
                if "CD" in superclasses:
                    labels.append("CD")
                if "HYP" in superclasses:
                    labels.append("HYP")
                if "PAC" in scp_codes:
                    labels.append("PAC")
                if "PVC" in scp_codes:
                    labels.append("PVC")
                if "AFIB" in scp_codes or "AFLT" in scp_codes:
                    labels.append("AFIB/AFL")
                if "STACH" in scp_codes or "SVTAC" in scp_codes or "PSVT" in scp_codes:
                    labels.append("TACHY")
                if "SBRAD" in scp_codes:
                    labels.append("BRADY")
                labels = ", ".join(labels)

                input_dir = input_folder / record_dir
                output_dir = output_folder / record_dir
                output_dir.mkdir(parents=True, exist_ok=True)

                # skip if output_dir contains images and not force_recompute
                if not force_recompute:
                    existing_images = find_images(str(output_dir), [".png", ".jpg", ".jpeg"])
                    existing_images = [img for img in existing_images if img.startswith(record_basename)]
                    if len(existing_images) > 0:
                        continue

                input_header_file = (input_dir / record_basename).with_suffix(f".{self.header_ext}")
                input_signal_files = (input_dir / record_basename).with_suffix(f".{self.data_ext}")
                output_header_file = (output_dir / record_basename).with_suffix(f".{self.header_ext}")

                # generate the synthetic images
                ecg_img_gen_config.input_file = str(input_signal_files)
                ecg_img_gen_config.header_file = str(input_header_file)
                ecg_img_gen_config.output_directory = str(output_dir)
                ecg_img_gen_config.start_index = -1

                ecg_img_gen_config.debug_msg = {
                    "input_dir": str(input_dir),
                    "output_dir": str(output_dir),
                    "record": record,
                }

                num_gen, out_image_files, metadata_array = run_single_file(ecg_img_gen_config)
                num_ecg_img_gen += num_gen
                pbar.set_postfix(num_ecg_img_gen=num_ecg_img_gen)

                record_images = []
                for image in find_images(str(output_dir), [".png", ".jpg", ".jpeg"]):
                    if image.startswith(record_basename):
                        record_images.append(image)

                # generate new header file
                input_header = input_header_file.read_text()
                lines = input_header.splitlines()
                record_line = " ".join(lines[0].strip().split(" ")[:4]) + "\n"
                signal_lines = "\n".join(ln.strip() for ln in lines[1:] if ln.strip() and not ln.startswith("#")) + "\n"
                comment_lines = (
                    "\n".join(
                        ln.strip()
                        for ln in lines[1:]
                        if ln.startswith("#")
                        and not any(
                            (ln.startswith(x) for x in ("# Age:", "# Sex:", "# Height:", "# Weight:", "# Labels:", "# Image:"))
                        )
                    )
                    + "\n"
                )

                record_line = record_line.strip() + f" {recording_date_string}\n"
                signal_lines = signal_lines.strip() + "\n"
                comment_lines = (
                    comment_lines.strip()
                    + f"# Age: {age}\n# Sex: {sex}\n# Height: {height}\n# Weight: {weight}\n# Labels: {labels}\n"
                )
                record_image_string = ", ".join(record_images)
                comment_lines += f"# Image: {record_image_string}\n"

                output_header = record_line + signal_lines + comment_lines

                output_header_file.write_text(output_header)

    @add_docstring(_prepare_synthetic_images_docstring)
    def prepare_synthetic_images(
        self,
        output_folder: Optional[Union[str, bytes, os.PathLike]] = None,
        fs: int = 500,
        force_recompute: bool = False,
        parallel: bool = False,
        **kwargs: Any,
    ) -> None:
        random_resolution = kwargs.get("random_resolution", self.__gen_img_default_config__["random_resolution"])
        if random_resolution:
            self.logger.warning(
                "In order to make the training process easier, it is recommended to generate the synthetic images "
                "with fixed resolution so that all images have the same size. In this way, the images can be "
                "processed in batches without resizing."
            )
        try:
            if parallel:
                if kwargs.get("hw_text", self.__gen_img_default_config__["hw_text"]) is True:
                    from utils.ecg_image_generator import en_core_sci_sm_model_dir

                    if en_core_sci_sm_model_dir is None:
                        self.logger.warning(
                            "The spaCy model en_core_sci_sm is not cached locally. Call the function `download_en_core_sci_sm` "
                            "from `utils.ecg_image_generator` to download the model. "
                            "Otherwise, it would be downloaded multiple times in parallel."
                        )
                        return
                if output_folder is None:
                    output_folder = self._synthetic_images_dir
                output_folder = Path(output_folder).expanduser().resolve()
                output_folder.mkdir(parents=True, exist_ok=True)
                if fs is None:
                    if self.fs in [100, 500]:
                        fs = self.fs
                    else:
                        self.logger.warning(f"invalid fs `{fs}`, use default fs {self.gen_img_default_fs} instead.")
                        fs = self.gen_img_default_fs
                input_folder = self.db_dir / self.__500Hz_dir__ if fs == 500 else self.db_dir / self.__100Hz_dir__

                ecg_img_gen_config = CFG(self.__gen_img_default_config__.copy())
                ecg_img_gen_config.update(kwargs)

                records = find_records(str(input_folder))
                args_list = []
                for record in records:
                    record_dir, record_basename = os.path.split(record)
                    ecg_id = record_basename.split("_")[0]
                    row = self._df_metadata.loc[ecg_id]
                    args_list.append(
                        {
                            "input_folder": input_folder,
                            "output_folder": output_folder,
                            "record": record,
                            "row": row,
                            "src_datetime_fmt": self.src_datetime_fmt,
                            "dst_datetime_fmt": self.dst_datetime_fmt,
                            "header_ext": self.header_ext,
                            "data_ext": self.data_ext,
                            "force_recompute": force_recompute,
                            "ecg_img_gen_config": ecg_img_gen_config,
                            "df_12sl_statements": self._df_12sl_statements,
                            "acute_mi_classes": self._acute_mi_classes,
                        }
                    )
                pool = mp.Pool(processes=max(1, mp.cpu_count() - 3), maxtasksperchild=10)
                # use tqdm to show progress
                for _ in tqdm(pool.imap_unordered(_generate_synthetic_image, args_list), total=len(args_list)):
                    pass
            else:
                self._prepare_synthetic_images(output_folder=output_folder, fs=fs, force_recompute=force_recompute, **kwargs)
        except KeyboardInterrupt:
            self.logger.info("Cancelled by user.")

        # save `ecg_img_gen_config` to the output folder
        Path(output_folder).joinpath("ecg_img_gen_config.json").write_text(json.dumps(ecg_img_gen_config, indent=4))
        if output_folder == self._synthetic_images_dir:
            self._gen_img_config_current = ecg_img_gen_config.copy()

    def _prepare_synthetic_image_from_record(
        self,
        record: Union[str, int],
        output_folder: Union[str, bytes, os.PathLike],
        fs: int = 500,
        force_recompute: bool = False,
        **kwargs: Any,
    ) -> None:
        """Prepare the synthetic images from a record.

        This function is mainly for testing and debugging purposes
        for generating synthetic images from a single record.

        Parameters
        ----------
        record : str or int
            The record name (ecg_id) or the index of the record.
        output_folder : str or bytes or os.PathLike
            The output folder to save the synthetic images.
        fs : int, default 500
            The sampling frequency of the record to generate the images.
        force_recompute : bool, default False
            Whether to force recompute the synthetic images.
        **kwargs : dict
            The keyword arguments for the image generator.

        """
        if isinstance(record, int):
            record = self._all_records[record]
        if output_folder is None:
            output_folder = self._synthetic_images_dir
        output_folder = Path(output_folder).expanduser().resolve()
        output_folder.mkdir(parents=True, exist_ok=True)
        if fs is None:
            if self.fs in [100, 500]:
                fs = self.fs
            else:
                self.logger.warning(f"invalid fs `{fs}`, use default fs {self.gen_img_default_fs} instead.")
                fs = self.gen_img_default_fs

        ecg_img_gen_config = CFG(self.__gen_img_default_config__.copy())
        ecg_img_gen_config.update(kwargs)

        ecg_id = record
        row = self._df_metadata.loc[record]
        if fs == 500:
            input_folder = self.db_dir / self.__500Hz_dir__
            record_rel_path = (self.db_dir / row.filename_hr).relative_to(input_folder)
        else:
            input_folder = self.db_dir / self.__100Hz_dir__
            record_rel_path = (self.db_dir / row.filename_lr).relative_to(input_folder)
        gen_img_args = {
            "input_folder": input_folder,
            "output_folder": output_folder,
            "record": record_rel_path,
            "row": row,
            "src_datetime_fmt": self.src_datetime_fmt,
            "dst_datetime_fmt": self.dst_datetime_fmt,
            "header_ext": self.header_ext,
            "data_ext": self.data_ext,
            "force_recompute": force_recompute,
            "ecg_img_gen_config": ecg_img_gen_config,
            "df_12sl_statements": self._df_12sl_statements,
            "acute_mi_classes": self._acute_mi_classes,
        }
        _generate_synthetic_image(gen_img_args)

    @property
    def database_info(self) -> DataBaseInfo:
        return _CINC2024_INFO

    @property
    def config(self) -> CFG:
        return self.__config

    @property
    def default_train_val_test_split(self) -> Dict[str, List[str]]:
        return {
            "train": self._df_records[self._df_records["strat_fold"] < 9].index.tolist(),
            "val": self._df_records[self._df_records["strat_fold"] == 9].index.tolist(),
            "test": self._df_records[self._df_records["strat_fold"] == 10].index.tolist(),
        }

    @property
    def default_train_val_split(self) -> Dict[str, List[str]]:
        return {
            "train": self._df_records[self._df_records["strat_fold"] < 10].index.tolist(),
            "val": self._df_records[self._df_records["strat_fold"] == 10].index.tolist(),
        }

    def download_synthetic_images(self, set_name: str = "subset") -> None:
        """Download the synthetic files generated offline from Google Drive."""
        if url_is_reachable("https://drive.google.com/"):
            source = "gdrive"
            url = self.__synthetic_images_url__[set_name]
        elif url_is_reachable("https://deep-psp.tech"):
            source = "deep-psp"
            url = self.__synthetic_images_url__[f"{set_name}-alt"]
        else:
            self.logger.warn("Can reach neither Google Drive nor deep-psp.tech. The synthetic images will not be downloaded.")
            return
        if url is None:
            self.logger.warn("The download URL is not available yet. The synthetic images will not be downloaded.")
            return
        if not os.access(self._synthetic_images_dir, os.W_OK):
            self.logger.warn("No write access. The synthetic images will not be downloaded.")
            return
        dl_file = str(self._synthetic_images_dir.parent / "ptb-xl-synthetic-images.zip")
        if source == "gdrive":
            gdown.download(url, dl_file, quiet=False)
            _unzip_file(dl_file, self._synthetic_images_dir)
        elif source == "deep-psp":
            http_get(url, self._synthetic_images_dir, extract=True)

        # reload the records
        self._ls_rec()

        # fix the datetime format of the header files if necessary
        self.logger.info("Fixing the datetime format of the header files in the synthetic images directory.")
        self.fix_datetime_format(self._synthetic_images_dir)

    def download_subset(self) -> None:
        """Download the subset of the database."""
        if url_is_reachable("https://drive.google.com/"):
            source = "gdrive"
            # url = "https://drive.google.com/u/0/uc?id=1ZsIPg-K9AUXq1LgI0DRLFLgviPfvx5P3"
            # url = "https://drive.google.com/u/0/uc?id=1tTEsgq5HNvB-Qy9cSk_S1koeDHE7M-d2"
            url = "https://drive.google.com/u/0/uc?id=1KM8ZFb5yMjaFxa0WoRthCe3YQMTqIMpy"
            dl_file = str(self.db_dir / "ptb-xl-subset.zip")
            gdown.download(url, dl_file, quiet=False)
            _unzip_file(dl_file, self.db_dir)
        elif url_is_reachable("https://deep-psp.tech"):
            source = "deep-psp"
            url = "https://deep-psp.tech/Data/ptb-xl-subset-tiny.zip"
            http_get(url, self.db_dir, extract=True)
        else:
            self.logger.warn("Can reach neither Google Drive nor deep-psp.tech. The synthetic images will not be downloaded.")

        # reload the records
        self._ls_rec()

    def download_aux_files(self) -> None:
        """Download the auxiliary files."""
        for filename, url in self.__aux_files__.items():
            if (self.db_dir / filename).exists():
                continue
            http_get(url, self.db_dir, filename=filename, extract=False)

        # reload the records
        self._ls_rec()

    @staticmethod
    def fix_datetime_format(folder: Union[str, bytes, os.PathLike]) -> None:
        """Fix the datetime format of the header files in the folder.

        In the first time of synthetic image generation, the datetime format of the header files is incorrect.
        It was mistakenly set as "%d/%m/%Y %H:%M:%S" instead of the correct form "%H:%M:%S %d/%m/%Y".

        Parameters
        ----------
        folder : `path-like`
            The folder containing the header files.

        Returns
        -------
        None

        """
        folder = Path(folder)
        wrong_pattern = "\\d{2}/\\d{2}/\\d{4} \\d{2}:\\d{2}:\\d{2}$"
        header_files = list(folder.rglob("*.hea"))
        for header_file in tqdm(header_files, desc="Fixing datetime format", dynamic_ncols=True, mininterval=1.0):
            lines = header_file.read_text().splitlines()
            # check the first line, where the recording date is located
            correction_flag = False
            if re.search(wrong_pattern, lines[0]):
                wrong_datetime_str = re.findall(wrong_pattern, lines[0])[0]
                correct_datetime_str = datetime.strptime(wrong_datetime_str, "%d/%m/%Y %H:%M:%S").strftime("%H:%M:%S %d/%m/%Y")
                lines[0] = lines[0].replace(wrong_datetime_str, correct_datetime_str)
                correction_flag = True
            if correction_flag:
                header_file.write_text("\n".join(lines))

    def get_img_size(self, img: Union[str, int]) -> Tuple[int, int]:
        """Get the size of the image.

        Parameters
        ----------
        img : str or int
            The image name or the index of the image.

        Returns
        -------
        tuple
            The size of the image in the form ``(width, height)``.

        """
        if isinstance(img, int):
            img = self._all_images[img]
        with Image.open(self._df_images.loc[img, "path"]) as img:
            return img.size

    @staticmethod
    def match_bbox(bbox: List[Dict[str, Any]], fmt: str = "coco") -> List[Dict[str, Any]]:
        """Match the waveform boxes with the lead name boxes.

        The matching is done by finding the minimum distance between
        the upper left corner of the lead name box and the lower left corner of the waveform box.

        Parameters
        ----------
        bbox : list of dict
            The bounding boxes. Each dict has the keys "bbox", "category_id", "category_name", etc.
        fmt : {"coco", "pascal_voc", "yolo", "albumentations"}, default "coco"
            The format of the bounding boxes, case insensitive.
            If is "coco", the bounding boxes will be in COCO format: ``[xmin, ymin, width, height]``.
            If is "pascal_voc", the bounding boxes will be in Pascal VOC format: ``[xmin, ymin, xmax, ymax]``.
            If is "yolo", the bounding boxes will be in YOLO format: ``[x_center, y_center, width, height]`` (normalized to [0, 1]).
            If is "albumentations", the bounding boxes will be in Albumentations format: ``[x_min, y_min, x_max, y_max]`` (normalized to [0, 1]).
            The coordinates have the origin at the upper left corner of the image.

        Returns
        -------
        list of dict
            The matched waveform bounding boxes,
            an item with key "lead_name" will be added to each dict.

        """
        waveform_boxes = [box for box in bbox if box["category_name"] == "waveform"]
        lead_name_boxes = [box for box in bbox if box["category_name"] != "waveform"]
        for wb in waveform_boxes:
            # find the minimum distance between the upper left corner of the lead name box
            # and the lower left corner of the waveform box.
            if fmt.lower() == "coco":
                dist_x = [np.abs(wb["bbox"][0] - lb["bbox"][0]) for lb in lead_name_boxes]
                dist_y = [np.abs(wb["bbox"][1] + wb["bbox"][3] - lb["bbox"][1]) for lb in lead_name_boxes]
            elif fmt.lower() == "pascal_voc":
                dist_x = [np.abs(wb["bbox"][0] - lb["bbox"][0]) for lb in lead_name_boxes]
                dist_y = [np.abs(wb["bbox"][3] - lb["bbox"][1]) for lb in lead_name_boxes]
            elif fmt.lower() == "yolo":
                dist_x = [np.abs(wb["bbox"][0] - lb["bbox"][0]) for lb in lead_name_boxes]
                dist_y = [np.abs(wb["bbox"][1] + wb["bbox"][3] / 2 - lb["bbox"][1]) for lb in lead_name_boxes]
            elif fmt.lower() == "albumentations":
                dist_x = [np.abs(wb["bbox"][0] - lb["bbox"][0]) for lb in lead_name_boxes]
                dist_y = [np.abs(wb["bbox"][3] - lb["bbox"][1]) for lb in lead_name_boxes]
            else:
                raise ValueError(f"Invalid bbox format `{fmt}`")
            min_dist_idx = np.argmin(np.array(dist_x) ** 2 + np.array(dist_y) ** 2)
            wb["lead_name"] = lead_name_boxes[min_dist_idx]["category_name"]
        return waveform_boxes

    def _convert_bbox_ann(self, image_dir: Optional[Union[str, bytes, os.PathLike]] = None) -> None:
        """Convert the bounding box annotations from the unofficial phase format
        to the official phase format.

        Unofficial phase format (txt files, actually csv files):

        - lead_bounding_box: x0, y0, x1, y1, is_full
        - text_bounding_box: x0, y0, x1, y1, lead_name

        The coordinates have the origin at the lower left corner of the image (matplotlib coordinate system).

        Official phase format (npz files):

        - lead_bounding_box:

            - lead_bbox: [[[y0, x0], [y1, x1], [y2, x2], [y3, x3]], ...]

        - text_bounding_box:

            - text_bbox: [[[y0, x0], [y1, x1], [y2, x2], [y3, x3]], ...]
            - lead_name: [lead_name, ...]

        Parameters
        ----------
        image_dir : str or bytes or os.PathLike, optional
            The directory containing the images.
            If is ``None``, self._synthetic_images_dir will be used.

        Returns
        -------
        None

        """
        if image_dir is None:
            image_dir = self._synthetic_images_dir
        image_dir = Path(image_dir).resolve()

        for folder in tqdm(list(image_dir.iterdir()), desc="Converting bbox annotations", dynamic_ncols=True, mininterval=1.0):
            for image_file in tqdm(
                list(Path(folder).glob(f"*.{self.img_ext}")),
                desc=f"Converting bbox annotations in {folder.name}",
                dynamic_ncols=True,
                mininterval=1.0,
                leave=False,
            ):
                image_name = image_file.stem
                with Image.open(image_file) as img:
                    img_width, img_height = img.size
                lead_bbox_file = Path(folder) / ecg_img_gen_constants.lead_bounding_box_dir_name / f"{image_name}.txt"
                if lead_bbox_file.exists():
                    lead_bbox = np.loadtxt(lead_bbox_file, delimiter=",", dtype=int)
                    new_lead_bbox = []
                    for x0, y0, x1, y1, is_full in lead_bbox:
                        new_lead_bbox.append(
                            [[img_height - y1, x0], [img_height - y1, x1], [img_height - y0, x1], [img_height - y0, x0]]
                        )
                    np.savez_compressed(
                        Path(folder) / ecg_img_gen_constants.lead_bounding_box_dir_name / f"{image_name}.{self.bbox_ext}",
                        lead_bbox=new_lead_bbox,
                    )
                text_bbox_file = Path(folder) / ecg_img_gen_constants.text_bounding_box_dir_name / f"{image_name}.txt"
                if text_bbox_file.exists():
                    text_bbox = pd.read_csv(text_bbox_file, header=None, names=["x0", "y0", "x1", "y1", "lead_name"])
                    new_text_bbox = []
                    new_lead_name = []
                    for _, row in text_bbox.iterrows():
                        x0, y0, x1, y1, lead_name = row
                        new_text_bbox.append(
                            [[img_height - y1, x0], [img_height - y1, x1], [img_height - y0, x1], [img_height - y0, x0]]
                        )
                        new_lead_name.append(lead_name)
                    np.savez_compressed(
                        Path(folder) / ecg_img_gen_constants.text_bounding_box_dir_name / f"{image_name}.{self.bbox_ext}",
                        text_bbox=new_text_bbox,
                        lead_name=new_lead_name,
                    )

    @staticmethod
    def get_roi(bbox: Sequence[BBox], padding: Union[int, float] = 0.0) -> Tuple[int, int, int, int]:
        """Get the region of interest (ROI) from the bounding boxes.

        Parameters
        ----------
        bbox : sequence of BBox
            The bounding boxes.
        padding : int or float, default 0.0
            The padding ratio to the bounding boxes.

        Returns
        -------
        tuple
            The region of interest in the form ``(x0, y0, x1, y1)``.

        """
        xmin = min([box.left for box in bbox])
        ymin = min([box.top for box in bbox])
        xmax = max([box.right for box in bbox])
        ymax = max([box.bottom for box in bbox])
        width = xmax - xmin
        height = ymax - ymin
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        img_width = bbox[0].img_width
        img_height = bbox[0].img_height
        return max(0, xmin - pad_x), max(0, ymin - pad_y), min(img_width, xmax + pad_x), min(img_height, ymax + pad_y)

    def select_image_subset(self, dst_dir: Union[str, bytes, os.PathLike], ratio: float = 0.01) -> None:
        """Select a subset of the synthetic images.

        Parameters
        ----------
        dst_dir : str or bytes or os.PathLike
            The destination directory to save the selected images.
        ratio : float, default 0.01
            The ratio of the selected images.

        Returns
        -------
        None

        """
        dst_dir = Path(dst_dir).expanduser().resolve()
        dst_dir.mkdir(parents=True, exist_ok=True)
        selected_images = {
            "train": self._df_images[self._df_images["strat_fold"] < 9].sample(frac=ratio).index.tolist(),
            "val": self._df_images[self._df_images["strat_fold"] == 9].sample(frac=ratio).index.tolist(),
            "test": self._df_images[self._df_images["strat_fold"] == 10].sample(frac=ratio).index.tolist(),
        }
        selected_images = self._df_images.loc[selected_images["train"] + selected_images["val"] + selected_images["test"]]
        for _, row in tqdm(
            selected_images.iterrows(),
            total=len(selected_images),
            desc="Copying ECG images",
            dynamic_ncols=True,
            mininterval=1.0,
        ):
            image_file = row["path"]
            (dst_dir / image_file.parent.relative_to(self._synthetic_images_dir)).mkdir(parents=True, exist_ok=True)
            shutil.copy(image_file, dst_dir / image_file.parent.relative_to(self._synthetic_images_dir))
            header_file = row["image_header"]
            (dst_dir / header_file.parent.relative_to(self._synthetic_images_dir)).mkdir(parents=True, exist_ok=True)
            shutil.copy(header_file, dst_dir / header_file.parent.relative_to(self._synthetic_images_dir))
            image_metadata_file = image_file.with_suffix(".json.gz")
            (dst_dir / image_metadata_file.parent.relative_to(self._synthetic_images_dir)).mkdir(parents=True, exist_ok=True)
            shutil.copy(image_metadata_file, dst_dir / image_metadata_file.parent.relative_to(self._synthetic_images_dir))
            lead_bbox_file = row["lead_bbox_file"]
            (dst_dir / lead_bbox_file.parent.relative_to(self._synthetic_images_dir)).mkdir(parents=True, exist_ok=True)
            shutil.copy(lead_bbox_file, dst_dir / lead_bbox_file.parent.relative_to(self._synthetic_images_dir))
            text_bbox_file = row["text_bbox_file"]
            (dst_dir / text_bbox_file.parent.relative_to(self._synthetic_images_dir)).mkdir(parents=True, exist_ok=True)
            shutil.copy(text_bbox_file, dst_dir / text_bbox_file.parent.relative_to(self._synthetic_images_dir))


def get_record_labels(row: pd.Series, df_12sl_statements: pd.DataFrame, acute_mi_classes: Set[str]) -> List[str]:
    """Get the diagnostic labels of the record.

    Parameters
    ----------
    row : pd.Series
        The metadata of the record.
    df_12sl_statements : pd.DataFrame
        The 12SL statements.
    acute_mi_classes : set of str
        Additional 12SL statements for acute myocardial infarction.

    Returns
    -------
    list of str
        The diagnostic labels of the record.

    """
    scp_codes = [scp_code for scp_code, likelihood in row["scp_codes"].items() if likelihood > 0]
    superclasses = row["diagnostic_superclass"]
    if row.name in df_12sl_statements.index:
        sl_codes = df_12sl_statements.loc[row.name]["statements"]
    else:
        sl_codes = list()
    return scp_codes_to_labels(scp_codes, superclasses, sl_codes, acute_mi_classes)


def scp_codes_to_labels(
    scp_codes: Sequence[str], superclasses: Sequence[str], sl_codes: Sequence[str], acute_mi_classes: Set[str]
) -> List[str]:
    """Map the SCP codes to the diagnostic labels.

    Parameters
    ----------
    scp_codes : Sequence[str]
        The SCP codes.
    superclasses : Sequence[str]
        The diagnostic superclasses obtained from the SCP codes.
    sl_codes : Sequence[str]
        Codes from the 12SL statements.
    acute_mi_classes : set of str
        Additional 12SL statements for acute myocardial infarction.

    Returns
    -------
    list of str
        The diagnostic labels.

    """
    labels = list()
    if "NORM" in superclasses:
        labels.append("NORM")
    if any(c in sl_codes for c in acute_mi_classes):
        labels.append("Acute MI")
    if "MI" in superclasses and not any(c in sl_codes for c in acute_mi_classes):
        labels.append("Old MI")
    if "STTC" in superclasses:
        labels.append("STTC")
    if "CD" in superclasses:
        labels.append("CD")
    if "HYP" in superclasses:
        labels.append("HYP")
    if "PAC" in scp_codes:
        labels.append("PAC")
    if "PVC" in scp_codes:
        labels.append("PVC")
    if "AFIB" in scp_codes or "AFLT" in scp_codes:
        labels.append("AFIB/AFL")
    if "STACH" in scp_codes or "SVTAC" in scp_codes or "PSVT" in scp_codes:
        labels.append("TACHY")
    if "SBRAD" in scp_codes:
        labels.append("BRADY")
    return labels


def _generate_synthetic_image(args: Dict[str, Any]) -> None:
    """Generate the synthetic images from the ECG time series data.

    This function is used for multiprocessing.

    Parameters
    ----------
    args : dict
        The arguments for generating the synthetic images.

    Returns
    -------
    None

    """
    input_folder = Path(args["input_folder"])
    output_folder = Path(args["output_folder"])
    record = args["record"]  # str
    row = args["row"]  # pd.Series
    src_datetime_fmt = args["src_datetime_fmt"]  # str
    dst_datetime_fmt = args["dst_datetime_fmt"]  # str
    header_ext = args["header_ext"]  # str
    data_ext = args["data_ext"]  # str
    force_recompute = args["force_recompute"]  # bool
    ecg_img_gen_config = args["ecg_img_gen_config"]  # CFG
    ecg_img_gen_config["link"] = ""  # set to empty to avoid internet errors
    df_12sl_statements = args["df_12sl_statements"]  # pd.DataFrame
    acute_mi_classes = args["acute_mi_classes"]  # set

    # Extract the demographics data.
    record_dir, record_basename = os.path.split(record)
    ecg_id = record_basename.split("_")[0]

    recording_date_string = row["recording_date"]
    recording_date = datetime.strptime(recording_date_string, src_datetime_fmt)
    recording_date_string = recording_date.strftime(dst_datetime_fmt)

    age = row["age"]
    age = cast_int_float_unknown(age)

    sex = row["sex"]
    if sex == 0:
        sex = "Male"
    elif sex == 1:
        sex = "Female"
    else:
        sex = "Unknown"

    height = row["height"]
    height = cast_int_float_unknown(height)

    weight = row["weight"]
    weight = cast_int_float_unknown(weight)

    # Extract the diagnostic superclasses.
    scp_codes = [scp_code for scp_code, likelihood in row["scp_codes"].items() if likelihood > 0]
    superclasses = row["diagnostic_superclass"]

    if ecg_id in df_12sl_statements.index:
        sl_codes = df_12sl_statements.loc[ecg_id]["statements"]
    else:
        sl_codes = list()

    labels = ", ".join(scp_codes_to_labels(scp_codes, superclasses, sl_codes, acute_mi_classes))

    input_dir = input_folder / record_dir
    output_dir = output_folder / record_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # skip if output_dir contains images and not force_recompute
    if not force_recompute:
        existing_images = find_images(str(output_dir), [".png", ".jpg", ".jpeg"])
        existing_images = [img for img in existing_images if img.startswith(record_basename)]
        if len(existing_images) > 0:
            return

    input_header_file = (input_dir / record_basename).with_suffix(f".{header_ext}")
    input_signal_files = (input_dir / record_basename).with_suffix(f".{data_ext}")
    output_header_file = (output_dir / record_basename).with_suffix(f".{header_ext}")

    # generate the synthetic images
    ecg_img_gen_config.input_file = str(input_signal_files)
    ecg_img_gen_config.header_file = str(input_header_file)
    ecg_img_gen_config.output_directory = str(output_dir)
    ecg_img_gen_config.start_index = -1

    _, out_image_files, metadata_array = run_single_file(ecg_img_gen_config)

    record_images = []
    for image in find_images(str(output_dir), [".png", ".jpg", ".jpeg"]):
        if image.startswith(record_basename):
            record_images.append(image)

    # generate new header file
    input_header = input_header_file.read_text()
    lines = input_header.splitlines()
    record_line = " ".join(lines[0].strip().split(" ")[:4]) + "\n"
    signal_lines = "\n".join(ln.strip() for ln in lines[1:] if ln.strip() and not ln.startswith("#")) + "\n"
    comment_lines = (
        "\n".join(
            ln.strip()
            for ln in lines[1:]
            if ln.startswith("#")
            and not any((ln.startswith(x) for x in ("# Age:", "# Sex:", "# Height:", "# Weight:", "# Labels:", "# Image:")))
        )
        + "\n"
    )

    record_line = record_line.strip() + f" {recording_date_string}\n"
    signal_lines = signal_lines.strip() + "\n"
    comment_lines = (
        comment_lines.strip() + f"# Age: {age}\n# Sex: {sex}\n# Height: {height}\n# Weight: {weight}\n# Labels: {labels}\n"
    )
    record_image_string = ", ".join(record_images)
    comment_lines += f"# Image: {record_image_string}\n"

    output_header = record_line + signal_lines + comment_lines

    output_header_file.write_text(output_header)

    # write the metadata (mainly bounding boxes) to files
    for idx, metadata in enumerate(metadata_array):
        image_file = Path(out_image_files[idx])
        text_bbox = np.array([list(item["text_bounding_box"].values()) for item in metadata["leads"]])
        lead_name = [item["lead_name"] for item in metadata["leads"]]
        metadata_file = (
            image_file.parent / ecg_img_gen_constants.text_bounding_box_dir_name / image_file.with_suffix(".npz").name
        )
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(metadata_file, text_bbox=text_bbox, lead_name=lead_name)
        lead_bbox = np.array([list(item["lead_bounding_box"].values()) for item in metadata["leads"]])
        metadata_file = (
            image_file.parent / ecg_img_gen_constants.lead_bounding_box_dir_name / image_file.with_suffix(".npz").name
        )
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(metadata_file, lead_bbox=lead_bbox)

        # save the full metadata a gzip-compressed JSON file
        if not ecg_img_gen_config["store_config"]:  # not stored by `ecg_image_generator.run_single_file`
            metadata_file = image_file.with_suffix(".json.gz")
            save_gzip_json(metadata, metadata_file)

    del metadata_array, lead_bbox, text_bbox, lead_name


def save_gzip_json(data: Any, file_path: Union[str, bytes, os.PathLike], **kwargs: Any) -> None:
    """Save the data to a gzip-compressed JSON file.

    Parameters
    ----------
    data : any
        The data to save.
    file_path : str or bytes or os.PathLike
        The file path.
    kwargs : dict
        The keyword arguments for `json.dump`.

    Returns
    -------
    None

    """
    with gzip.open(file_path, "wt") as f:
        json.dump(data, f, **kwargs)


def load_gzip_json(file_path: Union[str, bytes, os.PathLike], **kwargs: Any) -> Any:
    """Load the data from a gzip-compressed JSON file.

    Parameters
    ----------
    file_path : str or bytes or os.PathLike
        The file path.
    kwargs : dict
        The keyword arguments for `json.load`.

    Returns
    -------
    any
        The loaded data.

    """
    with gzip.open(file_path, "rt") as f:
        return json.load(f, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process CINC2024 database.")
    parser.add_argument(
        "operations",
        nargs=argparse.ONE_OR_MORE,
        type=str,
        choices=["download", "download_subset", "download_synthetic_images", "prepare_synthetic_images"],
    )
    parser.add_argument(
        "-d",
        "--db-dir",
        type=str,
        help="The directory to (store) the database.",
        dest="db_dir",
    )
    parser.add_argument(
        "-w",
        "--working-dir",
        type=str,
        default=None,
        help="The working directory to store the intermediate results.",
        dest="working_dir",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        help="The output directory to store the generated synthetic images, used when `operations` contain `prepare_synthetic_images`.",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Whether to use multiprocessing for generating synthetic images, used when `operations` contain `prepare_synthetic_images`.",
        default=False,
    )
    parser.add_argument(
        "--gen-img-config",
        type=str,
        help="Path to the configuration file for generating synthetic images, used when `operations` contain `prepare_synthetic_images`.",
        default=None,
        dest="gen_img_config",
    )
    parser.add_argument(
        "--download-image-set",
        type=str,
        default="subset",
        help="The image set to download",
        dest="download_image_set",
    )

    args = parser.parse_args()

    dr = CINC2024Reader(db_dir=args.db_dir)

    operations = args.operations
    if "download" in operations:
        dr.download()

    if "download_subset" in operations:
        dr.download_subset()

    if "download_synthetic_images" in operations:
        dr.download_synthetic_images(set_name=args.download_image_set)

    if "prepare_synthetic_images" in operations:
        if args.gen_img_config is not None:
            gen_img_config = json.loads(Path(args.gen_img_config).read_text())
        if args.parallel:
            from utils.ecg_image_generator import download_en_core_sci_sm, en_core_sci_sm_model_dir

            if (
                en_core_sci_sm_model_dir is None
                and gen_img_config.get("hw_text", dr.__gen_img_default_config__["hw_text"]) is True
            ):
                download_en_core_sci_sm()
        dr.prepare_synthetic_images(output_folder=args.output_folder, parallel=args.parallel, **gen_img_config)

    print("Done.")

    # usage examples:
    # python data_reader.py download -d /path/to/db_dir
    # python data_reader.py download download_synthetic_images -d /path/to/db_dir
    # python data_reader.py prepare_synthetic_images -d /path/to/db_dir [-o /path/to/output_folder] [--parallel] [--gen-img-config /path/to/gen_img_config.json]
