"""
Prometheus metric definitions shared across all Celery workers.
Each worker calls start_metrics_server(port) once at startup.
"""
import os
import logging
from prometheus_client import Counter, Histogram, start_http_server

logger = logging.getLogger(__name__)

# Generic Celery task metrics
CELERY_TASK_TOTAL = Counter(
    "celery_task_total",
    "Total Celery tasks by worker, task name, and status",
    ["worker", "task", "status"],
)
CELERY_TASK_DURATION = Histogram(
    "celery_task_duration_seconds",
    "Celery task duration in seconds",
    ["worker", "task"],
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120, 300],
)

# Validation-specific metrics
VALIDATION_CHECK_TOTAL = Counter(
    "validation_check_total",
    "Validation check outcomes by check name, result (pass/fail), and reason",
    ["check", "result", "reason"],
)
INGESTION_FILES_SEEN = Counter(
    "ingestion_files_seen_total",
    "Total files seen by the ingestion worker per source dataset",
    ["source_dataset"],
)


def start_metrics_server(port: int) -> None:
    """Start the Prometheus HTTP /metrics server on the given port."""
    try:
        start_http_server(port)
        logger.info("Prometheus metrics server started on port %d", port)
    except OSError as exc:
        logger.warning("Could not start metrics server on port %d: %s", port, exc)
