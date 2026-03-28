from pprint import pprint

import torch

from src.common.checkpointing import (
    build_checkpoint_dir,
    checkpoint_exists,
    load_latest_checkpoint,
    save_checkpoint,
)


def main():
    checkpoint_dir = build_checkpoint_dir(
        checkpoint_root="Training/artifacts/checkpoints",
        task="aesthetic",
        model_family="linear_head",
        model_version="aesthetic-linear-v1",
    )

    dummy_state = {
        "model_state_dict": {
            "weight": torch.randn(4, 4),
            "bias": torch.randn(4),
        },
        "optimizer_state_dict": {
            "lr": 1e-3,
            "step": 10,
        },
        "epoch": 1,
        "global_step": 10,
    }

    saved_paths = save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        state=dummy_state,
        task="aesthetic",
        model_family="linear_head",
        model_version="aesthetic-linear-v1",
        epoch=1,
        global_step=10,
        metric_name="val_loss",
        metric_value=0.42,
        dataset_version="ava-subset-v1",
        config_path="Training/configs/aesthetic/linear_head.yaml",
        is_best=True,
        save_epoch_copy=True,
    )

    print("Checkpoint directory:")
    print(checkpoint_dir)

    print("\nSaved checkpoint files:")
    pprint(saved_paths)

    print("\nCheckpoint exists:", checkpoint_exists(checkpoint_dir))

    loaded_state, loaded_metadata = load_latest_checkpoint(checkpoint_dir)

    print("\nLoaded state keys:")
    print(list(loaded_state.keys()))

    print("\nLoaded metadata:")
    pprint(loaded_metadata)


if __name__ == "__main__":
    main()