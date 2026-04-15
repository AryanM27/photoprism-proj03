from pathlib import Path
from typing import Dict, Optional
import mlflow
import torch

from src.common.checkpointing import build_checkpoint_dir
from src.common.config import load_config
from src.mlflow.logger import (
    configure_mlflow,
    log_artifact_if_exists,
    log_config_params,
    start_run,
)
from src.semantic.infer import generate_semantic_embeddings
from src.storage.artifact_io import save_summary_artifact
from src.storage.checkpoint_sync import sync_checkpoint_dir_from_remote

def _compute_retrieval_metrics(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> Dict[str, float]:
    sims = image_embeddings @ text_embeddings.T
    n = sims.size(0)
    targets = torch.arange(n)

    # image to text ranks
    i2t_ranks = []
    for i in range(n):
        order = torch.argsort(sims[i], descending=True)
        rank = (order == targets[i]).nonzero(as_tuple=False).item() + 1
        i2t_ranks.append(rank)

    # text to image ranks
    t2i_ranks = []
    for i in range(n):
        order = torch.argsort(sims[:, i], descending=True)
        rank = (order == targets[i]).nonzero(as_tuple=False).item() + 1
        t2i_ranks.append(rank)

    def recall_at_k(ranks, k):
        return sum(1 for r in ranks if r <= k) / len(ranks) if ranks else 0.0

    metrics = {
        "i2t_recall_at_1": recall_at_k(i2t_ranks, 1),
        "i2t_recall_at_5": recall_at_k(i2t_ranks, 5),
        "t2i_recall_at_1": recall_at_k(t2i_ranks, 1),
        "t2i_recall_at_5": recall_at_k(t2i_ranks, 5),
        "mean_i2t_rank": sum(i2t_ranks) / len(i2t_ranks) if i2t_ranks else 0.0,
        "mean_t2i_rank": sum(t2i_ranks) / len(t2i_ranks) if t2i_ranks else 0.0,
    }
    return metrics

def _prefix_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}

def _resolve_remote_checkpoint_prefix(config: dict) -> Optional[str]:
    if config.get("storage", {}).get("backend") != "object_store":
        return None

    artifact_uri = config["output"]["artifact_dir"]  # sample=swift://container/artifacts/semantic/...
    container = artifact_uri.replace("swift://", "", 1).split("/", 1)[0]

    return (
        f"swift://{container}/checkpoints/"
        f"{config['task']}/{config['model']['version']}"
    )

def _resolve_checkpoint_path(config: dict) -> Optional[str]:
    checkpoint_dir = build_checkpoint_dir(
        checkpoint_root=config["checkpoint"]["root_dir"],
        task=config["task"],
        model_family=config["model"]["family"],
        model_version=config["model"]["version"],
    )

    remote_checkpoint_prefix = _resolve_remote_checkpoint_prefix(config)
    if remote_checkpoint_prefix is not None:
        sync_checkpoint_dir_from_remote(config, remote_checkpoint_prefix, checkpoint_dir)

    best_path = Path(checkpoint_dir) / "best.pt"
    latest_path = Path(checkpoint_dir) / "latest.pt"

    if best_path.exists():
        print("Using best checkpoint")
        return str(best_path)
    if latest_path.exists():
        print("Using latest checkpoint")
        return str(latest_path)

    return None

def run_semantic_evaluation_for_split(
    config: dict,
    split: str = "val",
    checkpoint_path: Optional[str] = None,
    log_to_mlflow: bool = False,
) -> Dict:
    if checkpoint_path is None:
        checkpoint_path = _resolve_checkpoint_path(config)

    outputs = generate_semantic_embeddings(
        config=config,
        split=split,
        checkpoint_path=checkpoint_path,
        batch_size=config.get("evaluation", {}).get("batch_size"),
    )

    metrics = _compute_retrieval_metrics(
        image_embeddings=outputs["image_embeddings"],
        text_embeddings=outputs["text_embeddings"],
    )

    metrics["recall_at_1_mean"] = (
        metrics["i2t_recall_at_1"] + metrics["t2i_recall_at_1"]
    ) / 2.0

    prefixed_metrics = _prefix_metrics(metrics, split)

    summary = {
        "candidate_name": config.get("candidate_name", "unknown_candidate"),
        "model_type": config["model"]["type"],
        "model_version": config["model"]["version"],
        "dataset_version": config["dataset"]["dataset_version"],
        "checkpoint_path": outputs["checkpoint_path"],
        "device": outputs["device"],
        "num_images": len(outputs["image_ids"]),
        "num_texts": len(outputs["texts"]),
        "evaluation_split": split,
        "manifest_ref": outputs.get("manifest_ref"),
        "manifest_path": outputs.get("manifest_path"),
        **prefixed_metrics,
    }

    if log_to_mlflow:
        mlflow.log_param("evaluation_split", split)
        mlflow.set_tag("checkpoint_path", outputs["checkpoint_path"])
        for metric_name, metric_value in prefixed_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

    return summary

def _run_semantic_evaluation_impl(config: dict, config_path: str, tracking_uri: str)-> Dict:
    checkpoint_path = _resolve_checkpoint_path(config)

    # outputs = generate_semantic_embeddings(
    #     config=config,
    #     split="val",
    #     checkpoint_path=checkpoint_path,
    # )

    # metrics = _compute_retrieval_metrics(
    #     image_embeddings=outputs["image_embeddings"],
    #     text_embeddings=outputs["text_embeddings"],
    # )

    # summary = {
    #     "candidate_name": config.get("candidate_name", "unknown_candidate"),
    #     "model_type": config["model"]["type"],
    #     "model_version": config["model"]["version"],
    #     "dataset_version": config["dataset"]["dataset_version"],
    #     "checkpoint_path": checkpoint_path,
    #     "device": outputs["device"],
    #     "num_images": len(outputs["image_ids"]),
    #     "num_texts": len(outputs["texts"]),
    #     "mlflow_tracking_uri": tracking_uri,
    #     **metrics,
    # }

    # log_config_params(config)

    # mlflow.log_param("candidate_name", config.get("candidate_name", "unknown_candidate"))
    # mlflow.log_param("model_type", config["model"]["type"])
    # mlflow.log_param("model_version", config["model"]["version"])
    # mlflow.log_param("dataset_version", config["dataset"]["dataset_version"])
    # mlflow.log_param("evaluation_split", "val")
    # mlflow.log_param("device", outputs["device"])
    # mlflow.set_tag("checkpoint_path", checkpoint_path or "none")

    # for metric_name, metric_value in metrics.items():
    #     mlflow.log_metric(metric_name, metric_value)
    
    summary = run_semantic_evaluation_for_split(
        config=config,
        split="val",
        checkpoint_path=checkpoint_path,
        log_to_mlflow=True,
    )

    summary["mlflow_tracking_uri"] = tracking_uri

    summary_file = save_summary_artifact(config, summary)
    log_artifact_if_exists(str(summary_file))

    return summary

def run_semantic_evaluation(config_path: str) -> Dict:
    config = load_config(config_path)
    tracking_uri = configure_mlflow(config)
    experiment_name = config["experiment_name"]

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{config['candidate_name']}_semantic_eval"):
        log_config_params(config)

        summary = run_semantic_evaluation_for_split(
            config=config,
            split="val",
            checkpoint_path=None,
            log_to_mlflow=True,
        )

        summary["mlflow_tracking_uri"] = tracking_uri

        summary_file = save_summary_artifact(config, summary)
        log_artifact_if_exists(str(summary_file))

        return summary

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate a semantic model candidate.")
    parser.add_argument("--config", required=True, help="Path to semantic config YAML")
    args = parser.parse_args()

    summary = run_semantic_evaluation(args.config)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()