"""
metrics.py — Custom Prometheus metrics for the Serving search API.

Module-level definitions are automatically exposed via the /metrics endpoint
registered by prometheus_fastapi_instrumentator in main.py.
"""
from prometheus_client import Counter, Histogram

INFERENCE_SEARCH_TOTAL = Counter(
    "inference_search_total",
    "Total search requests by outcome",
    ["result"],  # ok | empty | error
)

INFERENCE_EMPTY_RESULTS_TOTAL = Counter(
    "inference_empty_results_total",
    "Total search requests that returned zero results",
)

INFERENCE_QUERY_LENGTH_CHARS = Histogram(
    "inference_query_length_chars",
    "Length of search queries in characters",
    buckets=[5, 10, 20, 40, 80, 160, 320],
)

INFERENCE_QUERY_EMBEDDING_NORM = Histogram(
    "inference_query_embedding_norm",
    "L2 norm of the query embedding vector",
    buckets=[0.5, 0.8, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1, 1.5],
)

INFERENCE_TOP_K_SCORE = Histogram(
    "inference_top_k_score",
    "Distribution of top-k similarity scores by pipeline stage",
    ["stage"],  # vector | rerank
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

INFERENCE_SCORE_SPREAD = Histogram(
    "inference_score_spread",
    "Difference between top-1 and top-k scores (retrieval quality signal)",
    buckets=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
)
