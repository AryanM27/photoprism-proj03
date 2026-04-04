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

    # image -> text ranks
    i2t_ranks = []
    for i in range(n):
        order = torch.argsort(sims[i], descending=True)
        rank = (order == targets[i]).nonzero(as_tuple=False).item() + 1
        i2t_ranks.append(rank)

    # text -> image ranks
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


def _resolve_remote_checkpoint_prefix(config: dict) -> Optional[str]:
    if config.get("storage", {}).get("backend") != "object_store":
        return None

    artifact_uri = config["output"]["artifact_dir"]  # e.g. swift://container/artifacts/semantic/...
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

def _run_semantic_evaluation_impl(config: dict, config_path: str, tracking_uri: str)-> Dict:
    checkpoint_path = _resolve_checkpoint_path(config)

    outputs = generate_semantic_embeddings(
        config=config,
        split="val",
        checkpoint_path=checkpoint_path,
    )

    metrics = _compute_retrieval_metrics(
        image_embeddings=outputs["image_embeddings"],
        text_embeddings=outputs["text_embeddings"],
    )

    summary = {
        "candidate_name": config.get("candidate_name", "unknown_candidate"),
        "model_type": config["model"]["type"],
        "model_version": config["model"]["version"],
        "dataset_version": config["dataset"]["dataset_version"],
        "checkpoint_path": checkpoint_path,
        "device": outputs["device"],
        "num_images": len(outputs["image_ids"]),
        "num_texts": len(outputs["texts"]),
        "mlflow_tracking_uri": tracking_uri,
        **metrics,
    }

    log_config_params(config)

    mlflow.log_param("candidate_name", config.get("candidate_name", "unknown_candidate"))
    mlflow.log_param("model_type", config["model"]["type"])
    mlflow.log_param("model_version", config["model"]["version"])
    mlflow.log_param("dataset_version", config["dataset"]["dataset_version"])
    mlflow.log_param("evaluation_split", "val")
    mlflow.log_param("device", outputs["device"])
    mlflow.set_tag("checkpoint_path", checkpoint_path or "none")

    # checkpoint_value = checkpoint_path or "none"

    # active_run = mlflow.active_run()
    # already_logged_checkpoint = None
    # if active_run is not None:
    #     already_logged_checkpoint = active_run.data.params.get("checkpoint_path")

    # if already_logged_checkpoint is None:
    #     mlflow.log_param("checkpoint_path", checkpoint_value)
    # elif already_logged_checkpoint != checkpoint_value:
    #     mlflow.set_tag("latest_checkpoint_path", checkpoint_value)

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    summary_file = save_summary_artifact(config, summary)
    log_artifact_if_exists(str(summary_file))

    return summary



def run_semantic_evaluation(config_path: str) -> Dict:
    config = load_config(config_path)
    tracking_uri = configure_mlflow(config)

    # checkpoint_path = _resolve_checkpoint_path(config)

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

    active_run = mlflow.active_run()
    if active_run is None:
        print("No active MLFlow run found; starting standalone evaluation")
        with start_run(experiment_name=config["experiment_name"]):
            return _run_semantic_evaluation_impl(config, config_path, tracking_uri)
    
    print("Active MLFlow run detected; logging evaluation into the current run")
    return _run_semantic_evaluation_impl(config, config_path, tracking_uri)

        # log_config_params(config)

        # mlflow.log_param("candidate_name", config.get("candidate_name", "unknown_candidate"))
        # mlflow.log_param("model_type", config["model"]["type"])
        # mlflow.log_param("model_version", config["model"]["version"])
        # mlflow.log_param("dataset_version", config["dataset"]["dataset_version"])
        # mlflow.log_param("evaluation_split", "val")
        # mlflow.log_param("device", outputs["device"])
        # mlflow.log_param("checkpoint_path", checkpoint_path or "none")

        # for metric_name, metric_value in metrics.items():
        #     mlflow.log_metric(metric_name, metric_value)

        # summary_file = save_summary_artifact(config, summary)
        # log_artifact_if_exists(str(summary_file))

    # return summary


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



# import json
# from pathlib import Path

# from src.common.config import load_config
# from src.eval.retrieval_metrics import (
#     compute_retrieval_metrics,
#     compute_similarity_matrix,
# )
# from src.semantic.infer import generate_embeddings


# def save_metrics(metrics: dict, output_dir: str) -> None:
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)

#     metrics_file = output_path / "metrics.json"
#     with metrics_file.open("w", encoding="utf-8") as f:
#         json.dump(metrics, f, indent=2)


# def run_semantic_evaluation(config_path: str) -> dict:
#     config = load_config(config_path)

#     outputs = generate_embeddings(
#         manifest_path=config["dataset"]["manifest_path"],
#         model_name=config["model"]["variant"],
#         device_str=config["runtime"]["device"],
#         use_mock_inference=config["model"].get("use_mock_inference", False),
#         embedding_dim=config["model"].get("embedding_dim", 512),
#     )

#     similarity_matrix = compute_similarity_matrix(
#         text_embeddings=outputs["text_embeddings"],
#         image_embeddings=outputs["image_embeddings"],
#     )

#     # aligned evaluation assumption:
#     # text i corresponds to image i
#     text_ids = outputs["image_ids"]
#     image_ids = outputs["image_ids"]

#     metrics = compute_retrieval_metrics(
#         similarity_matrix=similarity_matrix,
#         text_ids=text_ids,
#         image_ids=image_ids,
#         top_k=config["evaluation"]["top_k"],
#     )

#     metrics["device"] = outputs["device"]
#     metrics["num_samples"] = len(outputs["image_ids"])

#     save_metrics(metrics, config["output"]["artifact_dir"])

#     return metrics

#NEW IMPL

# import json
# from pathlib import Path

# import mlflow

# from src.common.config import load_config
# from src.eval.retrieval_metrics import (
#     compute_retrieval_metrics,
#     compute_similarity_matrix,
# )
# from src.mlflow.logger import (
#     configure_mlflow,
#     log_artifact_if_exists,
#     log_config_params,
#     log_metrics,
#     start_run,
# )
# from src.semantic.infer import generate_embeddings


# def save_metrics(metrics: dict, output_dir: str) -> Path:
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)

#     metrics_file = output_path / "metrics.json"
#     with metrics_file.open("w", encoding="utf-8") as f:
#         json.dump(metrics, f, indent=2)

#     return metrics_file


# def run_semantic_evaluation(config_path: str) -> dict:
#     config = load_config(config_path)

#     outputs = generate_embeddings(
#         manifest_path=config["dataset"]["manifest_path"],
#         model_name=config["model"]["variant"],
#         device_str=config["runtime"]["device"],
#         use_mock_inference=config["model"].get("use_mock_inference", False),
#         embedding_dim=config["model"].get("embedding_dim", 512),
#     )

#     similarity_matrix = compute_similarity_matrix(
#         text_embeddings=outputs["text_embeddings"],
#         image_embeddings=outputs["image_embeddings"],
#     )

#     text_ids = outputs["image_ids"]
#     image_ids = outputs["image_ids"]

#     metrics = compute_retrieval_metrics(
#         similarity_matrix=similarity_matrix,
#         text_ids=text_ids,
#         image_ids=image_ids,
#         top_k=config["evaluation"]["top_k"],
#     )

#     metrics["device"] = outputs["device"]
#     metrics["num_samples"] = len(outputs["image_ids"])

#     metrics_file = save_metrics(metrics, config["output"]["artifact_dir"])

#     tracking_uri = configure_mlflow()
#     experiment_name = config["experiment_name"]

#     with start_run(experiment_name=experiment_name):
#         log_config_params(config)

#         mlflow_safe_metrics = {
#             key: value
#             for key, value in metrics.items()
#             if isinstance(value, (int, float))
#         }
#         log_metrics(mlflow_safe_metrics)

#         log_artifact_if_exists(str(metrics_file))

#     metrics["mlflow_tracking_uri"] = tracking_uri
#     return metrics


#NEW IMPL

# import sys
# from pathlib import Path
# from src.datasets.uri_resolver import cache_manifest_from_uri

# # Ensure Training directory is in PYTHONPATH
# CURRENT_FILE = Path(__file__).resolve()
# PROJECT_ROOT = CURRENT_FILE.parents[2]  # points to Training/
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# import argparse
# import json
# from pathlib import Path
# from src.storage.artifact_io import save_metrics_artifact

# import mlflow

# from src.common.config import load_config
# from src.eval.retrieval_metrics import (
#     compute_retrieval_metrics,
#     compute_similarity_matrix,
# )
# from src.mlflow.logger import (
#     configure_mlflow,
#     log_artifact_if_exists,
#     log_config_params,
#     log_metrics,
#     start_run,
# )
# from src.semantic.infer import generate_embeddings

# #Commented to integrate storage
# # def save_metrics(metrics: dict, output_dir: str) -> Path:
# #     output_path = Path(output_dir)
# #     output_path.mkdir(parents=True, exist_ok=True)

# #     metrics_file = output_path / "metrics.json"
# #     with metrics_file.open("w", encoding="utf-8") as f:
# #         json.dump(metrics, f, indent=2)

# #     return metrics_file


# def run_semantic_evaluation(config_path: str) -> dict:
#     config = load_config(config_path)

#     # outputs = generate_embeddings(
#     #     manifest_path=config["dataset"]["manifest_path"],
#     #     model_name=config["model"]["variant"],
#     #     device_str=config["runtime"]["device"],
#     #     use_mock_inference=config["model"].get("use_mock_inference", False),
#     #     embedding_dim=config["model"].get("embedding_dim", 512),
#     # )

#     manifest_ref = config["dataset"].get("manifest_uri") or config["dataset"]["manifest_path"]
#     manifest_path = cache_manifest_from_uri(config, manifest_ref)

#     outputs = generate_embeddings(
#         manifest_path=manifest_path,
#         config=config,
#         model_name=config["model"]["variant"],
#         device_str=config["runtime"]["device"],
#         use_mock_inference=config["model"].get("use_mock_inference", False),
#         embedding_dim=config["model"].get("embedding_dim", 512),
#     )

#     similarity_matrix = compute_similarity_matrix(
#         text_embeddings=outputs["text_embeddings"],
#         image_embeddings=outputs["image_embeddings"],
#     )

#     text_ids = outputs["image_ids"]
#     image_ids = outputs["image_ids"]

#     metrics = compute_retrieval_metrics(
#         similarity_matrix=similarity_matrix,
#         text_ids=text_ids,
#         image_ids=image_ids,
#         top_k=config["evaluation"]["top_k"],
#     )

#     metrics["device"] = outputs["device"]
#     metrics["num_samples"] = len(outputs["image_ids"])

#     # metrics_file = save_metrics(metrics, config["output"]["artifact_dir"])
#     metrics_file = save_metrics_artifact(config, metrics)

#     # tracking_uri = configure_mlflow()
#     tracking_uri = configure_mlflow(config)

#     experiment_name = config["experiment_name"]

#     with start_run(experiment_name=experiment_name):
#         log_config_params(config)

#         mlflow_safe_metrics = {
#             key: value
#             for key, value in metrics.items()
#             if isinstance(value, (int, float))
#         }
#         log_metrics(mlflow_safe_metrics)

#         log_artifact_if_exists(str(metrics_file))

#     metrics["mlflow_tracking_uri"] = tracking_uri
#     return metrics


# def parse_args():
#     parser = argparse.ArgumentParser(description="Run semantic retrieval evaluation.")
#     parser.add_argument(
#         "--config",
#         required=True,
#         help="Path to the semantic evaluation YAML config.",
#     )
#     return parser.parse_args()


# def main():
#     args = parse_args()
#     metrics = run_semantic_evaluation(args.config)

#     print("Semantic evaluation metrics:")
#     for key, value in metrics.items():
#         print(f"{key}: {value}")


# if __name__ == "__main__":
#     main()