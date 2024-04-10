import os
from pathlib import Path

import transformers

from cfg import ModelCfg
from const import DATA_CACHE_DIR, MODEL_CACHE_DIR, REMOTE_HEADS_URLS
from data_reader import CINC2024Reader
from models import MultiHead_CINC2024
from utils.ecg_image_generator.HandwrittenText import download_en_core_sci_sm
from utils.misc import url_is_reachable

if os.environ.get("HF_ENDPOINT", None) is not None and (not url_is_reachable("https://huggingface.co")):
    # workaround for using huggingface hub in China
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)


def cache_pretrained_models():
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
    print("Done.")
