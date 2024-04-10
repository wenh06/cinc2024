from pathlib import Path

MODULE_DIR = Path(__file__).parent
CACHE_DIR = MODULE_DIR / ".cache"

save_img_ext = ".png"
lead_bounding_box_dir_name = "lead_bounding_box"
text_bounding_box_dir_name = "text_bounding_box"

format_4_by_3 = [["I", "II", "III"], ["aVR", "aVL", "aVF", "AVR", "AVL", "AVF"], ["V1", "V2", "V3"], ["V4", "V5", "V6"]]
