from pathlib import Path

TRAINING_ROOT= Path(__file__).resolve().parents[2]
CONFIGS_DIR= TRAINING_ROOT/"configs"
DATA_DIR= TRAINING_ROOT/"data"
ARTIFACTS_DIR= TRAINING_ROOT/"artifacts"