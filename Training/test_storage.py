from src.common.config import load_config
from src.storage.artifact_io import (
    save_history_artifact,
    save_metrics_artifact,
    save_summary_artifact,
)
from src.storage.resolver import resolve_storage_backend


def main():
    config = load_config("Training/configs/semantic/openclip_vitb32_baseline.yaml")

    backend = resolve_storage_backend(config)
    print("Resolved backend:", backend.__class__.__name__)

    metrics_path = save_metrics_artifact(
        config,
        metrics={"recall_at_1": 0.5, "mAP": 0.75},
    )
    print("Metrics saved to:", metrics_path)

    history_path = save_history_artifact(
        config,
        history=[
            {"epoch": 1, "loss": 0.8},
            {"epoch": 2, "loss": 0.6},
        ],
    )
    print("History saved to:", history_path)

    summary_path = save_summary_artifact(
        config,
        summary={"status": "ok", "model_version": "demo-v1"},
    )
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()