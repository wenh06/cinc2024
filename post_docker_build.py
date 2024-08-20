import os
import time
from pathlib import Path

import albumentations as A
import numpy as np
import transformers
from torch_ecg.utils.download import http_get, url_is_reachable

from const import (
    DATA_CACHE_DIR,
    FULL_DATA_CACHE_DIR,
    MODEL_CACHE_DIR,
    PROJECT_DIR,
    REMOTE_MODELS,
    SUBSET_DATA_CACHE_DIR,
    TEST_DATA_CACHE_DIR,
)
from data_reader import CINC2024Reader
from models import ECGWaveformDetector, ECGWaveformDigitizer, MultiHead_CINC2024
from team_code import SubmissionCfg
from utils.ecg_image_generator import download_en_core_sci_sm

# workaround for using huggingface hub in China
if os.environ.get("HF_ENDPOINT", None) is not None and (not url_is_reachable(os.environ["HF_ENDPOINT"])):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
elif os.environ.get("HF_ENDPOINT", None) is None and (not url_is_reachable("https://huggingface.co")):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR)
os.environ["HF_HOME"] = str(Path(MODEL_CACHE_DIR).parent)


def check_env():
    print("Checking the environment variables...")
    print(f"MODEL_CACHE_DIR: {MODEL_CACHE_DIR}")
    print(f"DATA_CACHE_DIR: {DATA_CACHE_DIR}")

    for env in [
        "HF_ENDPOINT",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_CACHE",
        "HF_HOME",
        "NO_ALBUMENTATIONS_UPDATE",
        "ALBUMENTATIONS_DISABLE_VERSION_CHECK",
    ]:
        print(f"{env}: {str(os.environ.get(env, None))}")

    print("Checking the environment variables done.")


def cache_pretrained_models():
    """Cache the pretrained models."""
    print("Caching the pretrained models...")
    if url_is_reachable("https://drive.google.com/"):
        remote_model_source = "google-drive"
    else:
        remote_model_source = "deep-psp"
    if SubmissionCfg.digitizer is not None:
        http_get(
            url=REMOTE_MODELS[SubmissionCfg.digitizer]["url"][remote_model_source],
            dst_dir=MODEL_CACHE_DIR,
            filename=REMOTE_MODELS[SubmissionCfg.digitizer]["filename"],
            extract=False,
        )
        model, train_config = ECGWaveformDigitizer.from_checkpoint(
            Path(MODEL_CACHE_DIR) / REMOTE_MODELS[SubmissionCfg.digitizer]["filename"]
        )
        print("digitizer loaded")
        print(f"digitizer: {model}")
        print(f"digitizer train config: {train_config}")
        del model, train_config

    if SubmissionCfg.classifier is not None:
        http_get(
            url=REMOTE_MODELS[SubmissionCfg.classifier]["url"][remote_model_source],
            dst_dir=MODEL_CACHE_DIR,
            filename=REMOTE_MODELS[SubmissionCfg.classifier]["filename"],
            extract=False,
        )
        model, train_config = MultiHead_CINC2024.from_checkpoint(
            Path(MODEL_CACHE_DIR) / REMOTE_MODELS[SubmissionCfg.classifier]["filename"]
        )
        print("classifier loaded")
        print(f"classifier: {model}")
        print(f"classifier train config: {train_config}")
        del model, train_config

    if SubmissionCfg.detector is not None:
        http_get(
            url=REMOTE_MODELS[SubmissionCfg.detector]["url"][remote_model_source],
            dst_dir=MODEL_CACHE_DIR,
            filename=REMOTE_MODELS[SubmissionCfg.detector]["filename"],
            extract=False,
        )
        model, train_config = ECGWaveformDetector.from_checkpoint(
            Path(MODEL_CACHE_DIR) / REMOTE_MODELS[SubmissionCfg.detector]["filename"]
        )
        print("detector loaded")
        print(f"detector: {model}")
        print(f"detector train config: {train_config}")
        del model, train_config

    # Download the spacy model
    download_en_core_sci_sm()


def cache_data():
    """Cache the synthetic image data and the subset."""
    print("Caching the synthetic image data...")
    reader_kwargs = {
        "db_dir": Path(SUBSET_DATA_CACHE_DIR),
        "synthetic_images_dir": Path(SUBSET_DATA_CACHE_DIR) / "synthetic_images",
        "aux_files_dir": Path(PROJECT_DIR) / "aux_files",
    }
    dr = CINC2024Reader(**reader_kwargs)
    # dr.download_synthetic_images(set_name="subset")  # "full" is too large, not uploaded to any cloud storage
    dr.download(name="subset")

    print("   Checking the subset data   ".center(80, "#"))
    print(f"{len(dr._df_metadata) = }")
    print(f"{len(dr._df_records) = }")
    print(f"{len(dr._all_records) = }")
    print(f"{len(dr._df_images) = }")
    default_train_val_split = dr.default_train_val_split
    print("For the default 2 part split:")
    print(f"train samples: {len(default_train_val_split['train'])}")
    print(f"val samples: {len(default_train_val_split['val'])}")
    default_train_val_test_split = dr.default_train_val_test_split
    print("For the default 3 part split:")
    print(f"test samples: {len(default_train_val_test_split['test'])}")
    print(f"train samples: {len(default_train_val_test_split['train'])}")
    print(f"val samples: {len(default_train_val_test_split['val'])}")

    print("   Test data checking complete.   ".center(80, "#"))

    print("Caching the action test data...")
    reader_kwargs = {
        "db_dir": Path(TEST_DATA_CACHE_DIR),
        "synthetic_images_dir": Path(TEST_DATA_CACHE_DIR) / "synthetic_images",
        "aux_files_dir": Path(PROJECT_DIR) / "aux_files",
    }
    dr = CINC2024Reader(**reader_kwargs)
    dr.download(name="subset-tiny")

    print("   Checking the action test data   ".center(80, "#"))
    print(f"{len(dr._df_metadata) = }")
    print(f"{len(dr._df_records) = }")
    print(f"{len(dr._all_records) = }")
    print(f"{len(dr._df_images) = }")
    default_train_val_split = dr.default_train_val_split
    print("For the default 2 part split:")
    print(f"train samples: {len(default_train_val_split['train'])}")
    print(f"val samples: {len(default_train_val_split['val'])}")
    default_train_val_test_split = dr.default_train_val_test_split
    print("For the default 3 part split:")
    print(f"test samples: {len(default_train_val_test_split['test'])}")
    print(f"train samples: {len(default_train_val_test_split['train'])}")
    print(f"val samples: {len(default_train_val_test_split['val'])}")

    print("   GitHub Action test data checking complete.   ".center(80, "#"))

    print("Caching the full data...")
    reader_kwargs = {
        "db_dir": Path(FULL_DATA_CACHE_DIR),
        "synthetic_images_dir": Path(FULL_DATA_CACHE_DIR) / "synthetic_images",
        "aux_files_dir": Path(PROJECT_DIR) / "aux_files",
    }
    dr = CINC2024Reader(**reader_kwargs)
    # dr.download_aux_files()
    dr.download(name="full")

    print("   Checking the full data   ".center(80, "#"))
    print(f"{len(dr._df_metadata) = }")
    print(f"{len(dr._df_records) = }")
    print(f"{len(dr._all_records) = }")
    print(f"{len(dr._df_images) = }")
    default_train_val_split = dr.default_train_val_split
    print("For the default 2 part split:")
    print(f"train samples: {len(default_train_val_split['train'])}")
    print(f"val samples: {len(default_train_val_split['val'])}")
    default_train_val_test_split = dr.default_train_val_test_split
    print("For the default 3 part split:")
    print(f"test samples: {len(default_train_val_test_split['test'])}")
    print(f"train samples: {len(default_train_val_test_split['train'])}")
    print(f"val samples: {len(default_train_val_test_split['val'])}")

    print("   Full data checking complete.   ".center(80, "#"))

    # download the aux files

    # dr.download_aux_files(dst_dir=Path(DATA_CACHE_DIR) / "aux_files")


def test_albumentations():
    transform = A.Compose(
        [
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
    )
    transform(image=np.random.randint(0, 256, (256, 256, 3)).astype(np.uint8))
    transform = A.Compose(
        [
            A.NoOp(),
        ],
    )
    transform(image=np.random.randint(0, 256, (256, 256, 3)).astype(np.uint8))


def prepare_synthetic_images():
    """Prepare the synthetic images."""
    print("Preparing the synthetic images...")
    reader_kwargs = {
        "db_dir": Path(DATA_CACHE_DIR),
        "synthetic_images_dir": Path(DATA_CACHE_DIR) / "synthetic_images",
        "aux_files_dir": Path(PROJECT_DIR) / "aux_files",
    }
    dr = CINC2024Reader(**reader_kwargs)
    config_index = 0  # -1 for default config, taking no action
    if config_index >= 0:
        config = dict(**dr.__gen_img_extra_configs__[config_index])
    else:
        config = dict()
    dr.prepare_synthetic_images(parallel=True, **config)


if __name__ == "__main__":
    transformers.logging.set_verbosity_info()
    check_env()
    time.sleep(2)
    cache_pretrained_models()
    time.sleep(2)
    cache_data()  # or prepare_synthetic_images(), prepare_synthetic_images NOT tested yet.
    time.sleep(2)
    test_albumentations()
    print("Done.")
