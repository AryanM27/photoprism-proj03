# Data Design Document
**Author:** Aryan Mamidwar

## 1. Data Repositories

### 1.1 Object Storage (MinIO / Chameleon Swift)
**Bucket:** `photoprism-proj03`

| Path prefix        | Contents                          | Written by              | When                        |
|--------------------|-----------------------------------|-------------------------|-----------------------------|
| `raw/<id>.<ext>`   | Original images (YFCC100M subset) | ingestion_worker        | On ingestion event          |
| `dvc/`             | DVC-tracked dataset artifacts     | DVC CLI                 | On `dvc push`               |
| `mlflow-artifacts/`| MLflow model artifacts            | MLflow                  | On model registration       |

**Schema:** No strict schema — binary blobs. Identified by `image_id` (MD5 hash of original path).

**Versioning:** DVC tracks `data/raw/`, `data/processed/`, `data/manifests/`, `data/feedback/` as named dataset versions. Each DVC snapshot has a git tag.

---

### 1.2 PostgreSQL (Metadata + Operational State)
**Database:** `photoprism`

#### Table: `images`
| Column         | Type      | Notes                                                              |
|----------------|-----------|--------------------------------------------------------------------|
| image_id       | VARCHAR PK| MD5 hash of original file path                                     |
| image_uri      | VARCHAR   | S3-compatible URI (`s3://photoprism-proj03/raw/<id>.<ext>`)        |
| storage_path   | VARCHAR   | Object storage key (`raw/<id>.<ext>`)                              |
| source_dataset | VARCHAR   | `yfcc` or `ava_subset`                                             |
| split             | VARCHAR   | `train` / `val` / `test` — deterministic: hex[-1] 0-2→val, 3-5→test, 6-f→train |
| status            | VARCHAR   | pending → validated / failed                                            |
| embedding_status  | VARCHAR   | pending / running / done / failed (default: pending)                    |
| embedded_at       | TIMESTAMP | When embedding completed (nullable)                                     |
| model_version     | VARCHAR   | Embedding model version used (nullable)                                 |
| created_at        | TIMESTAMP |                                                                         |
| updated_at        | TIMESTAMP |                                                                         |

**Written by:** `ingestion_worker` (insert), `validation_worker` (update status)

#### Table: `image_metadata`
| Column         | Type      | Notes                                          |
|----------------|-----------|------------------------------------------------|
| image_id       | VARCHAR PK| FK → images                                    |
| text           | TEXT      | Caption / description for semantic search      |
| source_dataset | VARCHAR   | `yfcc` or `ava_subset`                         |
| width          | INT       |                                                |
| height         | INT       |                                                |
| format         | VARCHAR   | JPEG / PNG / WEBP                              |
| exif_json       | TEXT      | Raw EXIF as JSON string                        |
| tags            | TEXT      | Comma-separated normalized                     |
| captured_at     | TIMESTAMP | From EXIF DateTimeOriginal                     |
| normalized_at   | TIMESTAMP |                                                |
| aesthetic_score | FLOAT     | 0–1 raw score from aesthetic model (nullable)  |

**Written by:** `validation_worker` after normalization

#### Table: `processing_jobs`
| Column     | Type      | Notes                                  |
|------------|-----------|----------------------------------------|
| job_id     | VARCHAR PK| UUID                                   |
| image_id   | VARCHAR   | FK → images                            |
| job_type   | VARCHAR   | ingestion / validation / backfill      |
| status     | VARCHAR   | queued / running / done / failed       |
| created_at    | TIMESTAMP |                                            |
| updated_at    | TIMESTAMP |                                            |
| error_message | TEXT      | Failure reason if status=failed (nullable) |
| retry_count   | INT       | Number of reprocessing attempts (default 0)|

**Written by:** `ingestion_worker`, `validation_worker`, `backfill_worker`

#### Table: `feedback_events`
| Column          | Type      | Notes                      |
|-----------------|-----------|----------------------------|
| event_id        | VARCHAR PK| UUID                       |
| user_id         | VARCHAR   |                            |
| query_id        | VARCHAR   |                            |
| image_id        | VARCHAR   |                            |
| shown_rank      | INT       | Position in result list    |
| clicked         | BOOLEAN   |                            |
| favorited       | BOOLEAN   |                            |
| semantic_score  | FLOAT     | Model score at query time  |
| aesthetic_score | FLOAT     | Model score at query time  |
| model_version   | VARCHAR   |                            |
| timestamp       | TIMESTAMP |                            |

**Written by:** `data_generator` (synthetic) / future UI integration

#### Table: `dataset_snapshots`
| Column        | Type      | Notes                          |
|---------------|-----------|--------------------------------|
| snapshot_id   | VARCHAR PK| UUID                           |
| version_tag   | VARCHAR   | DVC git tag (e.g. `v1.0`)      |
| manifest_path | VARCHAR   | S3 key to JSONL manifest       |
| record_count   | INT       |                                    |
| split_strategy | VARCHAR   | e.g. `hash_hex_62_19_19` (hex[-1] 0-2→val, 3-5→test, 6-f→train) |
| created_at     | TIMESTAMP |                                    |

**Written by:** `manifest_builder` after each DVC snapshot

**Versioning:** Alembic manages schema migrations. Each schema change gets a migration file.

---

### 1.3 RabbitMQ (Event Queue)
Three durable queues:

| Queue        | Producer               | Consumer                | Message schema                                                                      |
|--------------|------------------------|-------------------------|-------------------------------------------------------------------------------------|
| `ingestion`  | scanner CLI            | `ingestion_worker`      | `{"message_id": str, "timestamp": ISO8601, "image_id": str, "file_path": str}`      |
| `validation` | `ingestion_worker`     | `validation_worker`     | `{"message_id": str, "timestamp": ISO8601, "image_id": str, "storage_path": str}`   |
| `embedding`  | `validation_worker`    | `embedding_worker`      | `{"message_id": str, "timestamp": ISO8601, "image_id": str}`                        |
| `backfill`   | `backfill/pipeline.py` | `backfill_worker`       | `{"message_id": str, "timestamp": ISO8601, "image_id": str, "model_version": str}`  |

---

### 1.4 Training Manifest Contract

Manifest files are JSONL, versioned with DVC, stored at `s3://photoprism-proj03/manifests/`.

**Semantic manifest** (`manifests/<version>/semantic_<split>.jsonl`):
| Field           | Type   | Notes                                           |
|-----------------|--------|-------------------------------------------------|
| record_id       | string | UUID4 — unique per manifest row                 |
| image_id        | string | MD5 hash                                        |
| image_uri       | string | `s3://photoprism-proj03/raw/...`                |
| split           | string | `train` / `val` / `test`                        |
| dataset_version | string | Version tag (e.g. `v2`)                         |
| source_dataset  | string | `yfcc` or `ava_subset`                          |
| text_id         | string | UUID5 derived from text content (stable)        |
| text            | string | Caption / description                           |
| pair_label      | int    | `1` (positive pair)                             |

**Aesthetic manifest** (`manifests/<version>/aesthetic_<split>.jsonl`):
| Field           | Type   | Notes                                           |
|-----------------|--------|-------------------------------------------------|
| record_id       | string | UUID4 — unique per manifest row                 |
| image_id        | string | MD5 hash                                        |
| image_uri       | string | `s3://photoprism-proj03/raw/...`                |
| split           | string | `train` / `val` / `test`                        |
| dataset_version | string | Version tag (e.g. `v2`)                         |
| source_dataset  | string | `yfcc` or `ava_subset`                          |
| aesthetic_score | float  | 0–10 scale (raw 0–1 score × 10)                 |

**Version metadata** (`manifests/<version>/metadata.json`):
| Field            | Type   | Notes                                          |
|------------------|--------|------------------------------------------------|
| version          | string | Version tag                                    |
| built_at         | string | ISO8601 timestamp                              |
| split_strategy   | string | `hash_hex_62_19_19`                            |
| source_datasets  | list   | Dataset names included                         |
| manifest_uris    | object | S3 URIs keyed by type+split                    |
| counts           | object | Record counts per type+split + total           |

---

### 1.5 Qdrant (Vector Store)
**Collection:** `photoprism_images`

| Field          | Type        | Notes                                               |
|----------------|-------------|-----------------------------------------------------|
| id             | uint64      | SHA-256 of `image_id`, truncated to 64 bits         |
| vector         | float[512]  | CLIP `clip-ViT-B-32` embedding                      |
| image_id       | string      | MD5 hash (payload, for lookup)                      |
| source_dataset | string      | `yfcc` or `ava_subset` (payload)                    |
| split          | string      | `train` / `val` / `test` (payload)                  |
| aesthetic_score| float       | 0–1 raw score (payload, nullable)                   |
| model_version  | string      | Embedding model version (payload)                   |

**Written by:** `embedding_worker` after validation passes
**Read by:** `/search` endpoint in Features API (ANN nearest-neighbour queries)

---

## 2. Data Flow Diagram

```
YFCC100M subset on disk
        │
        ▼
[scanner.py] ──────────────────────────────────────────► RabbitMQ: ingestion queue
                                                                    │
                                                                    ▼
                                                         [ingestion_worker]
                                                          │         │          │
                                                          ▼         ▼          ▼
                                                       MinIO: raw/      Postgres: images    Postgres:
                                                       object store      (status=pending)   processing_jobs
                                                            │                │
                                                            └────────────────┘
                                                                    │
                                                                    ▼
                                                         RabbitMQ: validation queue
                                                                    │
                                                                    ▼
                                                          [validation_worker]
                                                           │           │            │           │
                                                           ▼           ▼            ▼           ▼
                                                  Postgres:      Postgres:        Postgres:  RabbitMQ:
                                                  image_metadata  images           processing  embedding
                                                  (EXIF, dims)   (status=validated)_jobs       queue
                                                           │                                   │
                                                           │                                   ▼
                                                           │                          [embedding_worker]
                                                           │                           │         │
                                                           │                           ▼         ▼
                                                           │                        Qdrant   Postgres:
                                                           │                        (vectors) images
                                                           │                                 (embedding_status)
                                                           ▼
                                                  [manifest_builder]
                                                           │
                                                           ▼
                                                      data/manifests/*.jsonl
                                                      (semantic_train, aesthetic_train, etc.)
                                                           │
                                                           ▼
                                                        [DVC snapshot]
                                                        (versioned, pushed to MinIO)
                                                           │
                                                           ▼
                                                   Postgres: dataset_snapshots


Postgres: images (status=validated)
                   │
                   ▼
[data_generator] ──► Postgres: feedback_events
                                       │
                                       ▼
                              [manifest_builder]
                                       │
                                       ▼
                               data/feedback/feedback_dataset.jsonl


[Model update trigger]
        │
        ▼
[backfill/pipeline.py] ──► RabbitMQ: backfill queue
                                         │
                                         ▼
                               [backfill_worker]
                                         │
                                         ▼
                              Re-embed images with new model
                              Update model_version in Postgres
```

---

## 3. Data Lineage & Versioning

- Every image carries `image_id` (MD5 of original path) through the entire pipeline
- DVC tracks all derived artifacts with the originating git commit as anchor
- `dataset_snapshots` table records which DVC version corresponds to which training run
- `feedback_events.model_version` tracks which model produced the scores a user reacted to
- Alembic migration history tracks schema evolution

---

## 4. Training Data Candidate Selection & Anti-Leakage

- Train/val/test split is deterministic: last hex digit of `image_id` → 0-2=val (~19%), 3-5=test (~19%), 6-f=train (~62%). Stable across pipeline re-runs.
- Feedback events are split so no feedback from validation or test images appears in train feedback
- Temporal split: feedback collected after a snapshot date is held out for next version's eval
