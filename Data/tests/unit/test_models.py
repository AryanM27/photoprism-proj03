from src.data_pipeline.db.models import (
    Image, ImageMetadata, ProcessingJob, FeedbackEvent, DatasetSnapshot
)


def test_image_model_has_required_fields():
    img = Image(image_id="img_001", storage_path="raw/img_001.jpg", status="pending")
    assert img.image_id == "img_001"
    assert img.status == "pending"


def test_feedback_event_has_required_fields():
    event = FeedbackEvent(
        event_id="e1", user_id="u1", query_id="q1", image_id="img_001",
        shown_rank=1, clicked=True, favorited=False,
        semantic_score=0.8, aesthetic_score=0.6, model_version="v1"
    )
    assert event.clicked is True
    assert event.model_version == "v1"


def test_processing_job_defaults_to_queued():
    job = ProcessingJob(job_id="j1", image_id="img_001", job_type="validation")
    assert job.status == "queued"


def test_dataset_snapshot_fields():
    snap = DatasetSnapshot(
        snapshot_id="snap_001", version_tag="v1.0",
        manifest_path="manifests/v1.0.json", record_count=5000
    )
    assert snap.version_tag == "v1.0"
    assert snap.record_count == 5000
