.PHONY: ingest build-manifests generate-feedback backfill test test-integration drift-report

ingest:
	python -m src.data_pipeline.ingestion.scanner --source-dir $(DATA_DIR) --source-dataset yfcc

build-manifests:
	dvc repro

generate-feedback:
	python -m src.data_generator.generator \
	  --duration $(or $(DURATION),300) \
	  --rate $(or $(RATE),1)

backfill:
	python -c "\
from src.data_pipeline.db.session import SessionLocal; \
from src.data_pipeline.backfill import trigger_backfill; \
db = SessionLocal(); trigger_backfill(db, model_version='$(VERSION)'); db.close()"

drift-report:
	mkdir -p reports
	python -c "\
from src.data_pipeline.manifests.build import generate_drift_report; \
generate_drift_report( \
  'data/manifests/v2/semantic_train.jsonl', \
  'data/manifests/v2/semantic_val.jsonl', \
  'reports/drift.html')"

test:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration
