"""
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from torch_ecg.cfg import CFG
from torch_ecg.databases.base import DataBaseInfo, PhysioNetDataBase
from torch_ecg.utils.misc import add_docstring, dict_to_str  # noqa: F401
from tqdm.auto import tqdm

from add_image_filenames import find_images
from helper_code import cast_int_float_unknown, find_records
from utils.ecg_image_generator.gen_ecg_image_from_data import run_single_file

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
    ],
    note="""
    """,
    issues="""
    """,
    references=[
        "https://moody-challenge.physionet.org/2024/",
        "https://physionet.org/content/ptb-xl/",
        "https://github.com/alphanumericslab/ecg-image-kit",
    ],
    doi=[
        "https://doi.org/10.13026/kfzx-aw45",
        "https://doi.org/10.1038/s41597-020-0495-6",
        "https://doi.org/10.48550/ARXIV.2307.01946",
    ],
)


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
        Auxilliary key word arguments.

    """

    __name__ = "CINC2024Reader"
    __metadata_file__ = "ptbxl_database.csv"
    __scp_statements_file__ = "scp_statements.csv"
    __100Hz_dir__ = "records100"
    __500Hz_dir__ = "records500"
    __synthetic_images_dir__ = "synthetic_images"
    __gen_img_default_config__ = CFG(
        json.loads((Path(__file__).resolve().parent / "utils" / "ecg-image-gen-default-config.json").read_text())
    )

    def __init__(
        self,
        db_dir: Union[str, bytes, os.PathLike],
        fs: int = 500,
        working_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(db_name="ptb-xl", db_dir=db_dir, working_dir=working_dir, fs=fs, verbose=verbose, **kwargs)
        self.data_ext = "dat"
        self.header_ext = "hea"

        assert os.access(self.db_dir, os.W_OK) or os.access(
            self.working_dir, os.W_OK
        ), "neither db_dir nor working_dir is writable"

        self.src_datetime_fmt = "%Y-%m-%d %H:%M:%S"
        self.dst_datetime_fmt = "%d/%m/%Y %H:%M:%S"
        self.gen_img_default_fs = 100

        self._synthetic_images_dir = kwargs.get("synthetic_images_dir", None)

        self._df_records = None
        self._df_metadata = None
        self._df_scp_statements = None
        self._all_records = None
        self._all_subjects = None
        self._ls_rec()

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in a dataframe.
        """
        # locate the true database directory using the metadata file
        try:
            metadata_file = list(self.db_dir.rglob(self.__metadata_file__))[0]
        except IndexError:
            raise FileNotFoundError(f"metadata file {self.__metadata_file__} not found in {self.db_dir}")
        self.db_dir = metadata_file.parent.resolve()
        assert (self.db_dir / self.__scp_statements_file__).exists(), f"scp_statements file not found in {self.db_dir}"

        # create synthetic images directory
        if self._synthetic_images_dir is not None and not os.access(self._synthetic_images_dir, os.W_OK):
            self.logger.warning(f"synthetic images directory `{self._synthetic_images_dir}` not writable.")
            self._synthetic_images_dir = None
        if self._synthetic_images_dir is None:
            if os.access(self.db_dir, os.W_OK):
                self._synthetic_images_dir = self.db_dir / self.__synthetic_images_dir__
            else:
                self._synthetic_images_dir = self.working_dir / self.__synthetic_images_dir__
        os.makedirs(self._synthetic_images_dir, exist_ok=True)

        # read metadata file and scp_statements file
        self._df_metadata = pd.read_csv(self.db_dir / self.__metadata_file__)
        self._df_metadata["ecg_id"] = self._df_metadata["ecg_id"].apply(lambda x: f"{x:05d}")
        self._df_metadata.set_index("ecg_id", inplace=True)
        self._df_metadata["patient_id"] = self._df_metadata["patient_id"].astype(int)
        self._df_scp_statements = pd.read_csv(self.db_dir / self.__scp_statements_file__, index_col=0)

        # TODO: search for the synthetic images and add them to the metadata dataframe

        if self._subsample is not None:
            size = min(
                len(self._df_metadata),
                max(1, int(round(self._subsample * len(self._df_metadata)))),
            )
            self.logger.debug(f"subsample `{size}` records from `{len(self._df_records)}`")
            self._df_records = self._df_metadata.sample(n=size, random_state=self._random_state)
        else:
            self._df_records = self._df_metadata.copy()

        if self.fs == 100:
            self._df_records["path"] = self._df_records["filename_lr"].apply(lambda x: self.db_dir / x)
        else:
            self._df_records["path"] = self._df_records["filename_hr"].apply(lambda x: self.db_dir / x)

        self._all_records = self._df_records.index.tolist()
        self._all_subjects = self._df_records["patient_id"].unique().tolist()

    def load_image(self, rec: Union[str, int]) -> Any:
        """Load the image of a record.

        Parameters
        ----------
        rec : str
            The record name (ecg_id).

        Returns
        -------
        img : ndarray
            The image of the record.

        """
        raise NotImplementedError

    def load_ann(self, rec: Union[str, int]) -> Any:
        """Load the annotation of a record.

        Parameters
        ----------
        rec : str
            The record name (ecg_id).

        Returns
        -------
        ann : dict
            The annotation of the record.

        """
        raise NotImplementedError

    def prepare_synthetic_images(
        self,
        output_folder: Optional[Union[str, bytes, os.PathLike]] = None,
        fs: int = 100,
        force_recompute: bool = False,
        **kwargs: Any,
    ) -> None:
        """Prepare synthetic images from the ECG time series data.

        This function is modified from the functions

        - prepare_ptbxl_data.run
        - add_image_filenames.run
        - ecg_image_generator.gen_ecg_image_from_data.run_single_file

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
        kwargs : dict, optional
            Extra key word arguments passed to the image generator.

        """
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
                scp_codes = row["scp_codes"]
                if "NORM" in scp_codes:
                    dx = "Normal"
                else:
                    dx = "Abnormal"

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
                inpout_signal_files = (input_dir / record_basename).with_suffix(f".{self.data_ext}")
                output_header_file = (output_dir / record_basename).with_suffix(f".{self.header_ext}")

                # generate the synthetic images
                ecg_img_gen_config.input_file = str(inpout_signal_files)
                ecg_img_gen_config.header_file = str(input_header_file)
                ecg_img_gen_config.output_directory = str(output_dir)
                ecg_img_gen_config.start_index = -1

                ecg_img_gen_config.debug_msg = {
                    "input_dir": str(input_dir),
                    "output_dir": str(output_dir),
                    "record": record,
                }

                # print(dict_to_str(ecg_img_gen_config))

                num_ecg_img_gen += run_single_file(ecg_img_gen_config)

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
                        and not any((ln.startswith(x) for x in ("#Age:", "#Sex:", "#Height:", "#Weight:", "#Dx:", "#Image:")))
                    )
                    + "\n"
                )

                record_line = record_line.strip() + f" {recording_date_string}\n"
                signal_lines = signal_lines.strip() + "\n"
                comment_lines = (
                    comment_lines.strip() + f"#Age: {age}\n#Sex: {sex}\n#Height: {height}\n#Weight: {weight}\n#Dx: {dx}\n"
                )
                record_image_string = ", ".join(record_images)
                comment_lines += f"#Image: {record_image_string}\n"

                output_header = record_line + signal_lines + comment_lines

                output_header_file.write_text(output_header)

    @property
    def database_info(self) -> DataBaseInfo:
        return _CINC2024_INFO
