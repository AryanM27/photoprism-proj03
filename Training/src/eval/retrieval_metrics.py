from typing import Dict, List

import torch


def compute_similarity_matrix(
    text_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
) -> torch.Tensor:

    return text_embeddings @ image_embeddings.T


def recall_at_k(
    similarity_matrix: torch.Tensor,
    text_ids: List[str],
    image_ids: List[str],
    k: int,
) -> float:
    """
    Assumes text_ids[i] should match image_ids[i].
    """
    num_queries = similarity_matrix.shape[0]
    hits = 0

    for i in range(num_queries):
        topk_indices = torch.topk(similarity_matrix[i], k=k).indices.tolist()
        topk_image_ids = [image_ids[idx] for idx in topk_indices]

        if text_ids[i] in topk_image_ids:
            hits += 1

    return hits / num_queries if num_queries > 0 else 0.0


def average_precision_for_query(
    ranked_image_ids: List[str],
    correct_image_id: str,
) -> float:
    """
    Single-positive AP for aligned image-text pair evaluation.
    """
    for rank, image_id in enumerate(ranked_image_ids, start=1):
        if image_id == correct_image_id:
            return 1.0 / rank
    return 0.0


def mean_average_precision(
    similarity_matrix: torch.Tensor,
    text_ids: List[str],
    image_ids: List[str],
) -> float:
    num_queries = similarity_matrix.shape[0]
    ap_scores = []

    for i in range(num_queries):
        ranked_indices = torch.argsort(similarity_matrix[i], descending=True).tolist()
        ranked_image_ids = [image_ids[idx] for idx in ranked_indices]
        ap = average_precision_for_query(ranked_image_ids, text_ids[i])
        ap_scores.append(ap)

    return sum(ap_scores) / num_queries if num_queries > 0 else 0.0


def compute_retrieval_metrics(
    similarity_matrix: torch.Tensor,
    text_ids: List[str],
    image_ids: List[str],
    top_k: List[int],
) -> Dict[str, float]:
    metrics = {}

    for k in top_k:
        metrics[f"recall_at_{k}"] = recall_at_k(
            similarity_matrix=similarity_matrix,
            text_ids=text_ids,
            image_ids=image_ids,
            k=k,
        )

    metrics["mAP"] = mean_average_precision(
        similarity_matrix=similarity_matrix,
        text_ids=text_ids,
        image_ids=image_ids,
    )

    return metrics