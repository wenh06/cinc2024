from pathlib import Path

import spacy
from torch_ecg.utils.download import http_get

from ..constants import CACHE_DIR, en_core_sci_sm_url

__all__ = ["download_en_core_sci_sm", "en_core_sci_sm_model_dir", "load_en_core_sci_sm"]


en_core_sci_sm_model_dir = None
# en_core_sci_sm_model = None

(CACHE_DIR / "en_core_sci_sm").mkdir(exist_ok=True, parents=True)
if len(list((CACHE_DIR / "en_core_sci_sm").rglob("config.cfg"))) == 1:
    en_core_sci_sm_model_dir = str(list((CACHE_DIR / "en_core_sci_sm").rglob("config.cfg"))[0].parent)
    # en_core_sci_sm_model = spacy.load(en_core_sci_sm_model_dir)


def download_en_core_sci_sm():
    global en_core_sci_sm_model_dir
    # global en_core_sci_sm_model
    if en_core_sci_sm_model_dir is not None:
        return
    model_dir = http_get(en_core_sci_sm_url, dst_dir=CACHE_DIR / "en_core_sci_sm", extract=True)
    # locate the model directory
    en_core_sci_sm_model_dir = str(list(Path(model_dir).rglob("config.cfg"))[0].parent)
    # en_core_sci_sm_model = spacy.load(en_core_sci_sm_model_dir)


def load_en_core_sci_sm():
    if en_core_sci_sm_model_dir is None:
        return None
    return spacy.load(en_core_sci_sm_model_dir)
