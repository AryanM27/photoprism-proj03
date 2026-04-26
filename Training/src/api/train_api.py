from datetime import datetime
from threading import Thread
from uuid import uuid4
from typing import Dict, Any

from fastapi import FastAPI, HTTPException

from src.api.schemas import TrainRequest, TrainResponse, JobStatusResponse
from src.common.config import load_config, deep_update
from src.aesthetic.train import train_aesthetic_from_config
from src.semantic.train import train_semantic_from_config

from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="Training Trigger API")

JOBS: Dict[str, Dict[str, Any]] = {}

TRAINING_JOBS_TOTAL = Counter("training_jobs_total", "Total training jobs", ["task"])
TRAINING_JOBS_FAILED = Counter("training_jobs_failed_total", "Failed jobs", ["task"])
TRAINING_JOB_RUNNING = Gauge("training_job_running", "Training job running state")


def _apply_request_overrides(base_config: dict, req: TrainRequest) -> dict:
    overrides = {}

    if req.manifest_uri is not None:
        overrides.setdefault("dataset", {})["manifest_uri"] = req.manifest_uri
    if req.dataset_version is not None:
        overrides.setdefault("dataset", {})["dataset_version"] = req.dataset_version
    if req.start_index is not None:
        overrides.setdefault("dataset", {})["start_index"] = req.start_index
    if req.max_records is not None:
        overrides.setdefault("dataset", {})["max_records"] = req.max_records
    if req.subset_seed is not None:
        overrides.setdefault("dataset", {})["subset_seed"] = req.subset_seed

    if req.experiment_name is not None:
        overrides["experiment_name"] = req.experiment_name
    if req.candidate_name is not None:
        overrides["candidate_name"] = req.candidate_name
    if req.model_version is not None:
        overrides.setdefault("model", {})["version"] = req.model_version
    if req.artifact_dir is not None:
        overrides.setdefault("output", {})["artifact_dir"] = req.artifact_dir

    if req.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = req.epochs
    if req.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = req.batch_size
    if req.learning_rate is not None:
        overrides.setdefault("training", {})["learning_rate"] = req.learning_rate
    if req.weight_decay is not None:
        overrides.setdefault("training", {})["weight_decay"] = req.weight_decay

    if req.run_test_after_training is not None:
        overrides.setdefault("evaluation", {})["run_test_after_training"] = req.run_test_after_training
    if req.evaluation_manifest_uri is not None:
        overrides.setdefault("evaluation", {})["test_manifest_uri"] = req.evaluation_manifest_uri
    if req.evaluation_max_records is not None:
        overrides.setdefault("evaluation", {})["max_records"] = req.evaluation_max_records

    if req.click_weight_alpha is not None:
        overrides.setdefault("training", {})["click_weight_alpha"] = req.click_weight_alpha
    if req.favourite_weight_alpha is not None:
        overrides.setdefault("training", {})["favourite_weight_alpha"] = req.favourite_weight_alpha
    if req.max_feedback_weight is not None:
        overrides.setdefault("training", {})["max_feedback_weight"] = req.max_feedback_weight

    config = deep_update(base_config, overrides)

    # small metadata tags for observability
    if req.trigger_reason is not None:
        config["_trigger_reason"] = req.trigger_reason
    config["_triggered_via_api"] = True

    return config


def _run_job(job_id: str, req: TrainRequest):
    JOBS[job_id]["status"] = "running"
    TRAINING_JOB_RUNNING.set(1)
    JOBS[job_id]["started_at"] = datetime.utcnow().isoformat()

    try:
        base_config = load_config(req.base_config_path)
        resolved_config = _apply_request_overrides(base_config, req)

        if req.task == "aesthetic":
            summary = train_aesthetic_from_config(resolved_config)
        elif req.task == "semantic":
            summary = train_semantic_from_config(resolved_config)
        else:
            raise ValueError(f"Unsupported task: {req.task}")

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["summary"] = summary
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        TRAINING_JOBS_FAILED.labels(task=req.task).inc()
    finally:
        TRAINING_JOB_RUNNING.set(0)
        JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat()


@app.post("/train", response_model=TrainResponse)
def trigger_train(req: TrainRequest):
    TRAINING_JOBS_TOTAL.labels(task=req.task).inc()
    job_id = str(uuid4())

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "task": req.task,
        "base_config_path": req.base_config_path,
        "trigger_reason": req.trigger_reason,
        "started_at": None,
        "finished_at": None,
        "error": None,
        "summary": None,
    }

    thread = Thread(target=_run_job, args=(job_id, req), daemon=True)
    thread.start()

    return TrainResponse(job_id=job_id, status="queued")


@app.get("/train/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)