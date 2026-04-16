"""
Celery signal handlers for generic task-level Prometheus metrics.

Import and call register_signals(app, worker_name, metrics_port) once per worker
entrypoint. The worker_init signal starts the HTTP metrics server; task_prerun /
task_postrun / task_failure update the counters and histograms.
"""
import logging
import time

from celery.signals import worker_init, task_prerun, task_postrun, task_failure

from src.data_pipeline.observability.metrics import (
    CELERY_TASK_TOTAL,
    CELERY_TASK_DURATION,
    start_metrics_server,
)

logger = logging.getLogger(__name__)

# Module-level storage for in-flight task start times: {task_id: (worker, task, start_time)}
_task_start: dict[str, tuple[str, str, float]] = {}


def register_signals(worker_name: str, metrics_port: int) -> None:
    """Wire up Celery signals for `worker_name`, exposing /metrics on `metrics_port`."""

    @worker_init.connect(weak=False)
    def on_worker_init(**kwargs):
        start_metrics_server(metrics_port)
        logger.info("[%s] worker_init: metrics server on :%d", worker_name, metrics_port)

    @task_prerun.connect(weak=False)
    def on_task_prerun(task_id, task, **kwargs):
        _task_start[task_id] = (worker_name, task.name, time.monotonic())

    @task_postrun.connect(weak=False)
    def on_task_postrun(task_id, task, retval, state, **kwargs):
        entry = _task_start.pop(task_id, None)
        status = "success" if state == "SUCCESS" else "failure"
        CELERY_TASK_TOTAL.labels(worker=worker_name, task=task.name, status=status).inc()
        if entry:
            _, _, start = entry
            CELERY_TASK_DURATION.labels(worker=worker_name, task=task.name).observe(
                time.monotonic() - start
            )

    @task_failure.connect(weak=False)
    def on_task_failure(task_id, exception, task, **kwargs):
        # task_postrun also fires on failure and increments the counter there;
        # only clean up the start-time entry here to avoid double-counting.
        _task_start.pop(task_id, None)
        logger.debug("[%s] task_failure: %s — %s", worker_name, task.name, exception)
