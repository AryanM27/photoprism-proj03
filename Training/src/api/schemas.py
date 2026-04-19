from typing import Literal, Optional
from pydantic import BaseModel


class TrainRequest(BaseModel):
    task: Literal["semantic", "aesthetic"]
    base_config_path: str

    manifest_uri: Optional[str] = None
    dataset_version: Optional[str] = None
    start_index: Optional[int] = None
    max_records: Optional[int] = None
    subset_seed: Optional[int] = None

    experiment_name: Optional[str] = None
    candidate_name: Optional[str] = None
    model_version: Optional[str] = None
    artifact_dir: Optional[str] = None

    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None

    trigger_reason: Optional[str] = None
    run_test_after_training: Optional[bool] = None
    evaluation_manifest_uri: Optional[str] = None
    evaluation_max_records: Optional[int] = None


class TrainResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    task: str
    base_config_path: str
    trigger_reason: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    summary: Optional[dict] = None