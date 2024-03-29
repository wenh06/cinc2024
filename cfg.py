"""
Configurations for models, training, etc., as well as some constants.
"""

import pathlib

from torch_ecg.cfg import CFG

__all__ = [
    "BaseCfg",
]


_BASE_DIR = pathlib.Path(__file__).absolute().parent


###############################################################################
# Base Configs,
# including path, data type, classes, etc.
###############################################################################

BaseCfg = CFG()
BaseCfg.db_dir = None
BaseCfg.working_dir = None
BaseCfg.project_dir = _BASE_DIR
BaseCfg.log_dir = _BASE_DIR / "log"
BaseCfg.model_dir = _BASE_DIR / "saved_models"
BaseCfg.log_dir.mkdir(exist_ok=True)
BaseCfg.model_dir.mkdir(exist_ok=True)

BaseCfg.normal_class = "Normal"
BaseCfg.abnormal_class = "Abnormal"
BaseCfg.classes = [BaseCfg.normal_class, BaseCfg.abnormal_class]
