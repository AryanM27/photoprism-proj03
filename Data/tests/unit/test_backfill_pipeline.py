"""Unit tests for backfill pipeline — no live DB required."""
from unittest.mock import patch, MagicMock, call
from src.data_pipeline.backfill.pipeline import trigger_backfill


def _make_fake_images(n: int):
    images = []
    for i in range(n):
        img = MagicMock()
        img.image_id = f"img_{i:03d}"
        images.append(img)
    return images


def test_trigger_backfill_dry_run_returns_jobs():
    fake_session = MagicMock()
    fake_session.query.return_value.filter_by.return_value.all.return_value = _make_fake_images(5)

    jobs = trigger_backfill(fake_session, model_version="v2", dry_run=True)

    assert len(jobs) == 5
    assert all(j["job_type"] == "backfill" for j in jobs)
    assert all(j["model_version"] == "v2" for j in jobs)
    assert all("image_id" in j for j in jobs)
    assert all("job_id" in j for j in jobs)


def test_trigger_backfill_dry_run_does_not_write_db():
    fake_session = MagicMock()
    fake_session.query.return_value.filter_by.return_value.all.return_value = _make_fake_images(3)

    trigger_backfill(fake_session, model_version="v2", dry_run=True)

    fake_session.add.assert_not_called()
    fake_session.commit.assert_not_called()


def test_trigger_backfill_wet_run_writes_db_and_publishes():
    fake_session = MagicMock()
    fake_session.query.return_value.filter_by.return_value.all.return_value = _make_fake_images(3)

    with patch("src.data_pipeline.workers.backfill_worker.reprocess_image") as mock_task:
        jobs = trigger_backfill(fake_session, model_version="v2", dry_run=False)

    assert fake_session.add.call_count == 3
    fake_session.commit.assert_called()
    assert mock_task.apply_async.call_count == 3


def test_trigger_backfill_returns_empty_for_no_validated_images():
    fake_session = MagicMock()
    fake_session.query.return_value.filter_by.return_value.all.return_value = []

    jobs = trigger_backfill(fake_session, model_version="v2", dry_run=True)

    assert jobs == []
