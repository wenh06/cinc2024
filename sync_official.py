"""
synchronize with the official repo
"""

from pathlib import Path

project_dir = Path(__file__).resolve().parent
official_dir = {
    "baseline": project_dir / "official_baseline",
    "scoring": project_dir / "official_scoring_metric",
}

files = {
    "baseline": [
        "helper_code.py",
        "remove_hidden_data.py",
        "prepare_image_data.py",
        "run_model.py",
        "train_model.py",
        "prepare_ptbxl_data.py",
    ],
    "scoring": [
        "evaluate_model.py",
    ],
}


def main():
    updated = False
    for repo, file_list in files.items():
        for filename in file_list:
            src = official_dir[repo] / filename
            dst = project_dir / filename
            if not dst.exists():
                dst.touch()
            if src.read_text() == dst.read_text():
                continue
            print(f"Copying **{src.relative_to(project_dir)}** " f"to **{dst.relative_to(project_dir)}**")
            dst.write_text(src.read_text())
            updated = True
    if not updated:
        print("Submodules of the official repositories are up-to-date.")


if __name__ == "__main__":
    main()
