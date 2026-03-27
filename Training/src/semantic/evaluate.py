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


import json
from pathlib import Path

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


def save_metrics(metrics: dict, output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_file = output_path / "metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics_file


def run_semantic_evaluation(config_path: str) -> dict:
    config = load_config(config_path)

    outputs = generate_embeddings(
        manifest_path=config["dataset"]["manifest_path"],
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

    metrics_file = save_metrics(metrics, config["output"]["artifact_dir"])

    tracking_uri = configure_mlflow()
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