"""
Prometheus metric definitions shared across all Celery workers.

Uses prometheus_client multiprocess mode so metrics updated in Celery
child processes (ForkPoolWorkers) are visible to the main process HTTP server.

Requires PROMETHEUS_MULTIPROC_DIR env var to point to a writable directory.
Each worker calls start_metrics_server(port) once at startup.
"""
import os
import logging
import threading
from wsgiref.simple_server import make_server

from prometheus_client import Counter, Histogram, CollectorRegistry, multiprocess, make_wsgi_app

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


def _multiprocess_app():
    def app(environ, start_response):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return make_wsgi_app(registry)(environ, start_response)
    return app


def start_metrics_server(port: int) -> None:
    """Start the Prometheus /metrics server.

    Uses MultiProcessCollector when PROMETHEUS_MULTIPROC_DIR is set so that
    observations made in Celery child processes are visible to Prometheus.
    """
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "")
    if multiproc_dir:
        os.makedirs(multiproc_dir, exist_ok=True)
        try:
            httpd = make_server("", port, _multiprocess_app())
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            logger.info("Prometheus metrics server (multiprocess) started on port %d", port)
        except OSError as exc:
            logger.warning("Could not start metrics server on port %d: %s", port, exc)
    else:
        from prometheus_client import start_http_server
        try:
            start_http_server(port)
            logger.info("Prometheus metrics server started on port %d", port)
        except OSError as exc:
            logger.warning("Could not start metrics server on port %d: %s", port, exc)
