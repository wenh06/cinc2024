"""
synchronize with the official repo
"""

from pathlib import Path

project_dir = Path(__file__).resolve().parent
official_dir = {
    "baseline": project_dir / "official_baseline_classifier",
    "scoring": project_dir / "official_scoring_metric",
}

files = {
    "baseline": [
        "helper_code.py",
        "remove_data.py",
        "remove_labels.py",
        "run_model.py",
        "train_model.py",
        "truncate_data.py",
    ],
    "scoring": [
        "evaluate_model.py",
    ],
}


def main():
    for repo, file_list in files.items():
        for filename in file_list:
            src = official_dir[repo] / filename
            dst = project_dir / filename
            if src.read_text() == dst.read_text():
                continue
            print(f"Copying **{src.relative_to(project_dir)}** " f"to **{dst.relative_to(project_dir)}**")
            dst.write_text(src.read_text())


if __name__ == "__main__":
    main()
