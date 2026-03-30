# from pathlib import Path

# TRAINING_ROOT= Path(__file__).resolve().parents[2]
# CONFIGS_DIR= TRAINING_ROOT/"configs"
# DATA_DIR= TRAINING_ROOT/"data"
# ARTIFACTS_DIR= TRAINING_ROOT/"artifacts"

from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = TRAINING_ROOT / "configs"
DATA_DIR = TRAINING_ROOT / "data"
ARTIFACTS_DIR = TRAINING_ROOT / "artifacts"


def resolve_training_path(path_str: str) -> str:
    path = Path(path_str)

    if path.is_absolute():
        return str(path)

    return str(TRAINING_ROOT / path)