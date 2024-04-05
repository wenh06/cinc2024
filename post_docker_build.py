import os
from pathlib import Path

import transformers

from cfg import ModelCfg
from const import DATA_CACHE_DIR, MODEL_CACHE_DIR, REMOTE_HEADS_URLS
from data_reader import CINC2024Reader
from models import MultiHead_CINC2024
from utils.misc import url_is_reachable

os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)


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


if __name__ == "__main__":
    transformers.logging.set_verbosity_info()
    cache_data()
    cache_pretrained_models()
    print("Done.")
