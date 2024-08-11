import os
from pathlib import Path

import albumentations as A
import numpy as np
import transformers
from deprecated import deprecated
from torch_ecg.utils.download import http_get, url_is_reachable

from cfg import ModelCfg
from const import DATA_CACHE_DIR, MODEL_CACHE_DIR, REMOTE_HEADS_URLS, REMOTE_MODELS
from data_reader import CINC2024Reader
from models import ECGWaveformDetector, MultiHead_CINC2024
from team_code import SubmissionCfg
from utils.ecg_image_generator import download_en_core_sci_sm

# workaround for using huggingface hub in China
if os.environ.get("HF_ENDPOINT", None) is not None and (not url_is_reachable(os.environ["HF_ENDPOINT"])):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
elif os.environ.get("HF_ENDPOINT", None) is None and (not url_is_reachable("https://huggingface.co")):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)


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
        print(model)
        print(train_config)
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
        print(model)
        print(train_config)
        del model, train_config

    # Download the spacy model
    download_en_core_sci_sm()


@deprecated(reason="Use cache_pretrained_models instead.", action="error")
def cache_pretrained_models_bak():
    """Cache the pretrained models."""
    print("Caching the pretrained models...")
    key = f"{ModelCfg.backbone_source}--{ModelCfg.backbone_name}"
    if url_is_reachable("https://www.dropbox.com/"):
        remote_heads_url = REMOTE_HEADS_URLS[key]["dropbox"]
    else:
        remote_heads_url = REMOTE_HEADS_URLS[key]["deep-psp"]
    model_dir = Path(MODEL_CACHE_DIR) / key.replace("/", "--")
    model = MultiHead_CINC2024.from_remote_heads(
        url=remote_heads_url,
        model_dir=model_dir,
    )

    # Download the spacy model
    download_en_core_sci_sm()


def cache_data():
    """Cache the synthetic image data and the subset."""
    print("Caching the synthetic image data...")
    reader_kwargs = {
        "db_dir": Path(DATA_CACHE_DIR),
        "synthetic_images_dir": Path(DATA_CACHE_DIR) / "synthetic_images",
    }
    dr = CINC2024Reader(**reader_kwargs)
    dr.download_synthetic_images(set_name="subset")  # "full" is too large, not uploaded to any cloud storage
    dr.download_subset()


def test_albumentations():
    transform = A.Compose(
        [
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
    )
    transform(image=np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8))
    transform = A.Compose(
        [
            A.NoOp(),
        ],
    )
    transform(image=np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8))


def prepare_synthetic_images():
    """Prepare the synthetic images."""
    print("Preparing the synthetic images...")
    reader_kwargs = {
        "db_dir": Path(DATA_CACHE_DIR),
        "synthetic_images_dir": Path(DATA_CACHE_DIR) / "synthetic_images",
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
    cache_pretrained_models()
    cache_data()  # or prepare_synthetic_images(), prepare_synthetic_images NOT tested yet.
    test_albumentations()
    print("Done.")
