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

import sys
from pathlib import Path
from src.datasets.uri_resolver import cache_manifest_from_uri

# Ensure Training directory is in PYTHONPATH
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]  # points to Training/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from pathlib import Path
from src.storage.artifact_io import save_metrics_artifact

import mlflow

from src.common.config import load_config
from src.eval.retrieval_metrics import (
    compute_retrieval_metrics,
    compute_similarity_matrix,
)
from src.mlflow.logger import (
    configure_mlflow,
    log_artifact_if_exists,
    log_config_params,
    log_metrics,
    start_run,
)
from src.semantic.infer import generate_embeddings

#Commented to integrate storage
# def save_metrics(metrics: dict, output_dir: str) -> Path:
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)

#     metrics_file = output_path / "metrics.json"
#     with metrics_file.open("w", encoding="utf-8") as f:
#         json.dump(metrics, f, indent=2)

#     return metrics_file


def run_semantic_evaluation(config_path: str) -> dict:
    config = load_config(config_path)

    # outputs = generate_embeddings(
    #     manifest_path=config["dataset"]["manifest_path"],
    #     model_name=config["model"]["variant"],
    #     device_str=config["runtime"]["device"],
    #     use_mock_inference=config["model"].get("use_mock_inference", False),
    #     embedding_dim=config["model"].get("embedding_dim", 512),
    # )

    manifest_ref = config["dataset"].get("manifest_uri") or config["dataset"]["manifest_path"]
    manifest_path = cache_manifest_from_uri(config, manifest_ref)

    outputs = generate_embeddings(
        manifest_path=manifest_path,
        config=config,
        model_name=config["model"]["variant"],
        device_str=config["runtime"]["device"],
        use_mock_inference=config["model"].get("use_mock_inference", False),
        embedding_dim=config["model"].get("embedding_dim", 512),
    )

    similarity_matrix = compute_similarity_matrix(
        text_embeddings=outputs["text_embeddings"],
        image_embeddings=outputs["image_embeddings"],
    )

    text_ids = outputs["image_ids"]
    image_ids = outputs["image_ids"]

    metrics = compute_retrieval_metrics(
        similarity_matrix=similarity_matrix,
        text_ids=text_ids,
        image_ids=image_ids,
        top_k=config["evaluation"]["top_k"],
    )

    metrics["device"] = outputs["device"]
    metrics["num_samples"] = len(outputs["image_ids"])

    # metrics_file = save_metrics(metrics, config["output"]["artifact_dir"])
    metrics_file = save_metrics_artifact(config, metrics)

    # tracking_uri = configure_mlflow()
    tracking_uri = configure_mlflow(config)

    experiment_name = config["experiment_name"]

    with start_run(experiment_name=experiment_name):
        log_config_params(config)

        mlflow_safe_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        log_metrics(mlflow_safe_metrics)

        log_artifact_if_exists(str(metrics_file))

    metrics["mlflow_tracking_uri"] = tracking_uri
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run semantic retrieval evaluation.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the semantic evaluation YAML config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = run_semantic_evaluation(args.config)

    print("Semantic evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()