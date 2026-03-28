import shutil
from pathlib import Path

from src.aesthetic.train import train_aesthetic_baseline


CONFIG_PATH = "Training/configs/aesthetic/linear_head_baseline.yaml"
CHECKPOINT_ROOT = Path("Training/artifacts/checkpoints/aesthetic/linear_head/aesthetic-linear-v1")
ARTIFACT_DIR = Path("Training/artifacts/aesthetic/linear_head_baseline")


def clean_previous_outputs():
    if CHECKPOINT_ROOT.exists():
        shutil.rmtree(CHECKPOINT_ROOT)

    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)


def main():
    print("=== Fresh run ===")
    clean_previous_outputs()
    summary_1 = train_aesthetic_baseline(CONFIG_PATH)

    for key, value in summary_1.items():
        print(f"{key}: {value}")

    print("\n=== Resume run ===")
    # Temporarily flip resume in config manually before second run if you want real resume behavior.
    print("For a true resume test, set training.resume: true in the config and rerun this script.")
    

if __name__ == "__main__":
    main()