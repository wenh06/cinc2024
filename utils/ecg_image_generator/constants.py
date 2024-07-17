from pathlib import Path

MODULE_DIR = Path(__file__).parent
CACHE_DIR = MODULE_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
CONFIG_DIR = MODULE_DIR / "Config"

save_img_ext = ".png"
lead_bounding_box_dir_name = "lead_bounding_box"
text_bounding_box_dir_name = "text_bounding_box"

format_4_by_3 = [["I", "II", "III"], ["aVR", "aVL", "aVF", "AVR", "AVL", "AVF"], ["V1", "V2", "V3"], ["V4", "V5", "V6"]]

en_core_sci_sm_url = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
