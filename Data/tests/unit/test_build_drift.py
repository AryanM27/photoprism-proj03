import json
import os
import pytest
from unittest.mock import patch, MagicMock


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_drift_report_creates_html(tmp_path):
    """Happy path: two manifests with shared numeric column -> HTML file created."""
    ref = tmp_path / "ref.jsonl"
    cur = tmp_path / "cur.jsonl"
    out = tmp_path / "reports" / "drift.html"
    _write_jsonl(ref, [{"aesthetic_score": 0.5, "image_id": f"img_{i}"} for i in range(5)])
    _write_jsonl(cur, [{"aesthetic_score": 0.6, "image_id": f"img_{i}"} for i in range(5)])

    mock_report = MagicMock()
    with patch("evidently.report.Report", return_value=mock_report), \
         patch("evidently.metric_preset.DataDriftPreset"):
        from src.data_pipeline.manifests.build import generate_drift_report
        generate_drift_report(str(ref), str(cur), str(out))

    mock_report.run.assert_called_once()
    mock_report.save_html.assert_called_once_with(str(out))


def test_drift_report_raises_on_empty_reference(tmp_path):
    """Empty reference manifest raises ValueError before any Evidently call."""
    ref = tmp_path / "ref.jsonl"
    cur = tmp_path / "cur.jsonl"
    ref.write_text("")
    _write_jsonl(cur, [{"aesthetic_score": 0.5}])

    from src.data_pipeline.manifests.build import generate_drift_report
    with pytest.raises(ValueError, match="empty"):
        generate_drift_report(str(ref), str(cur), str(tmp_path / "out.html"))


def test_drift_report_raises_on_no_shared_numeric_columns(tmp_path):
    """No shared numeric columns raises ValueError."""
    ref = tmp_path / "ref.jsonl"
    cur = tmp_path / "cur.jsonl"
    _write_jsonl(ref, [{"image_id": "a", "tags": "nature"}])
    _write_jsonl(cur, [{"image_id": "b", "tags": "urban"}])

    from src.data_pipeline.manifests.build import generate_drift_report
    with pytest.raises(ValueError, match="numeric"):
        generate_drift_report(str(ref), str(cur), str(tmp_path / "out.html"))


def test_drift_report_bare_filename_no_makedirs_error(tmp_path):
    """Bare output filename (no directory component) does not raise FileNotFoundError."""
    ref = tmp_path / "ref.jsonl"
    cur = tmp_path / "cur.jsonl"
    _write_jsonl(ref, [{"pair_label": 1, "image_id": f"img_{i}"} for i in range(5)])
    _write_jsonl(cur, [{"pair_label": 0, "image_id": f"img_{i}"} for i in range(5)])

    # Change to tmp_path so bare filename resolves to a writable location
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        mock_report = MagicMock()
        with patch("evidently.report.Report", return_value=mock_report), \
             patch("evidently.metric_preset.DataDriftPreset"):
            from src.data_pipeline.manifests.build import generate_drift_report
            generate_drift_report(str(ref), str(cur), "drift.html")
        mock_report.save_html.assert_called_once_with("drift.html")
    finally:
        os.chdir(original_dir)
