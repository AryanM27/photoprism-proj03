# PhotoPrism MLOps Project — Full System Overview

**Course:** NYU MLOps  
**Cloud:** Chameleon CHI@TACC (m1.xlarge — 8 vCPU, 16 GB RAM)  
**Team:** Aryan (Data), Milind (Training), Adarsh (Serving & Deployment)

---

## What This System Does

A production-grade MLOps system that adds semantic image search and aesthetic quality scoring on top of PhotoPrism (an open-source photo manager). Users can search photos by natural language query ("sunset over mountains") and get results ranked by both semantic relevance and visual quality.

The system is fully automated: it ingests images, trains models, monitors live performance, detects data drift, retrains on new data, runs a promotion gate to prevent regressions, deploys with zero downtime, monitors the deployment with a canary window, and rolls back automatically if something goes wrong.

---

## System Architecture

```
[Images on disk]
       ↓
[Data Pipeline — Aryan]
  Ingest → Validate → Embed with CLIP → Store vectors in Qdrant
       ↓
[Serving Layer — Adarsh]
  FastAPI (3 replicas) behind nginx
  POST /search → CLIP embed → Qdrant ANN search → Aesthetic rerank → Results
       ↓
[Monitoring]
  Prometheus + Grafana → detect drift → fire webhook
       ↓
[Training — Milind]
  Retrain aesthetic model → upload artifact to S3
       ↓
[Deployment — Adarsh]
  Promotion gate → commit updated config → redeploy → canary → backfill
```

---

## Module 1: Data Pipeline

**Owner:** Aryan  
**Location:** `/Data`

### What it does
Multi-stage async pipeline that processes images from raw files to searchable vectors.

**Stages:**
1. **Ingestion** — scans source directories, uploads images to S3, inserts metadata into PostgreSQL
2. **Validation** — extracts EXIF data, checks image quality, normalizes metadata
3. **Embedding** — runs CLIP inference on each image, stores 512-dim vectors in Qdrant
4. **Backfill** — re-scores all images when a new aesthetic model is promoted

**Tech:** Celery workers + RabbitMQ, PostgreSQL, Qdrant, DVC, Evidently

### Database (PostgreSQL)
| Table | Purpose |
|---|---|
| `images` | Metadata, status, S3 path, embedding status |
| `image_metadata` | EXIF, captions, aesthetic scores |
| `processing_jobs` | Async job tracking |
| `feedback_events` | User clicks, favorites, search interactions |
| `dataset_snapshots` | DVC versioning snapshots |

### Drift Detection
Evidently monitors the `feedback_events` table for data drift. When drift is detected, a Grafana alert fires a webhook to the serving layer, which triggers the full retraining pipeline automatically.

### Dataset Versioning (DVC)
Training manifests (JSONL files listing train/val/test images) are versioned with DVC. Before every training run, the pipeline SSHes into the VM, rebuilds manifests from the current database state, and pushes to DVC remote.

---

## Module 2: Serving

**Owner:** Adarsh  
**Location:** `/Serving`

### What it does
FastAPI inference server with 3 replicas behind an nginx load balancer. Handles all search, scoring, feedback, and deployment.

### API Endpoints
| Endpoint | Purpose |
|---|---|
| `POST /search` | Text-to-image semantic search with aesthetic reranking |
| `POST /score/aesthetic` | Score a single image's aesthetic quality |
| `POST /feedback` | Log user interaction (click, favorite, search) |
| `POST /webhook/drift` | Trigger retraining when drift is detected |
| `GET /health` | Liveness probe |
| `GET /metrics` | Prometheus metrics scrape |

### Search Flow
```
User query: "golden hour beach"
  → CLIP text embedding (Open-CLIP ViT-B/32)
  → Qdrant ANN search (top-50 candidates)
  → MobileNetV3 aesthetic ranker scores each image
  → Return top-10 ranked by combined score
  → Log feedback event to PostgreSQL
```

### Two Models
1. **Semantic model (CLIP):** Frozen — `semantic-openclip-enhanced-vit-b32-v2`. Handles text ↔ image embedding for search.
2. **Aesthetic model (MobileNetV3):** Retrained in the pipeline — `aesthetic-mobilenet-v3-large-fusion-v3`. Scores visual quality on a 0-1 scale.

Both checkpoints are downloaded from S3 at container startup by `checkpoint_resolver.py`.

### Promotion Gate (`scripts/promote_model.py`)
Before deploying a newly trained model, metrics are compared:
- New aesthetic MAE must be at least **0.02 lower** than current production
- New MAE must be **under 0.6 absolute**
- If it passes: `current_production.yaml` is updated and committed to the repo
- If it fails: no deployment, current model stays

### `current_production.yaml`
Single source of truth for active model versions:
```yaml
semantic:
  model_version: semantic-openclip-enhanced-vit-b32-v2
  s3_artifact_path: artifacts/semantic/openclip_enhanced_real_v1/training_summary.txt
aesthetic:
  model_version: aesthetic-mobilenet-v3-large-fusion-v3
  mae: 0.4761
  s3_artifact_path: artifacts/aesthetic/mobilenet_v3_large_fusion_real_v1/training_summary.txt
```
Every promotion is a traceable git commit with `[skip ci]` tag.

### Monitoring (Prometheus + Grafana)
Custom metrics tracked in real time:
- Search latency (p50 / p95 / p99)
- Error rate per endpoint
- Embedding norm distribution
- Aesthetic score distribution
- Results count per query
- Active replica count

---

## Module 3: Training

**Owner:** Milind  
**Location:** `/Training`

### What it does
Trains the aesthetic scoring model on image quality data. Exposes a REST API (`/train`, `/train/status/:id`) so GitHub Actions can trigger and poll training jobs remotely.

### Active Model: Aesthetic (MobileNetV3 Large Fusion)
- **Config:** `configs/aesthetic/mobilenet_v3_large_fusion_real_v3.yaml`
- **Data:** up to 250 training images, 50 evaluation images (from DVC manifests)
- **Output:** `best.pt` checkpoint + `training_summary.txt` uploaded to S3
- **Tracked with:** MLflow (metrics, params, artifacts)

### Semantic Model (Frozen)
CLIP fine-tuning code exists but is not triggered in the production pipeline. The semantic model checkpoint is fixed.

### Evaluation Metrics
- **MAE / MSE / RMSE** — primary metrics for aesthetic model promotion
- **mAP, Recall@K, nDCG** — retrieval metrics for semantic evaluation

---

## Full Automated Pipeline (End-to-End)

### Trigger: Scheduled or Drift-Detected
The `retrain-redeploy` workflow runs automatically on the 1st of every month, or when Grafana detects data drift and fires a webhook.

### Step-by-Step

```
1. BUILD MANIFESTS
   GitHub Actions SSHes into Chameleon VM
   Runs: dvc repro build_manifests
   Pushes updated manifests to DVC remote
   ↓

2. TRIGGER AESTHETIC TRAINING
   POST /train to Milind's training API
   {task: "aesthetic", config: "mobilenet_v3_large_fusion_real_v3.yaml", max_records: 250}
   Poll /train/status every 60 seconds (up to 3 hours)
   ↓

3. PROMOTION GATE
   Download training_summary.txt from S3
   Compare new MAE vs current_production.yaml
   If new MAE not 0.02 better → STOP, keep current model
   If passes → update current_production.yaml, commit to repo
   ↓

4. ZERO-DOWNTIME DEPLOY
   SSH to VM: git pull
   nginx: --no-recreate (stays running, no downtime)
   serving-api x3: --force-recreate (restart, loads new checkpoint from S3)
   embedding-worker: restart
   ↓

5. CANARY WINDOW (3 minutes)
   Query Prometheus: error rate over last 5 minutes
   If error rate > 5%:
     → git revert HEAD, git push
     → redeploy old model
     → if rollback fails: open GitHub Issue (critical)
   If stable: continue
   ↓

6. BACKFILL
   SSH to VM
   Trigger backfill-worker to re-score all images with new aesthetic model
   (Long-running — can be cancelled; prod is already stable at this point)
```

---

## Infrastructure & Deployment

**Cloud:** Chameleon CHI@TACC  
**VM:** m1.xlarge — 8 vCPU, 16 GB RAM  
**Deployment:** Docker Compose (single VM, not Kubernetes)

### Running Containers
| Container | Purpose |
|---|---|
| `nginx` | Load balancer (never recreated during deploys) |
| `serving-api` x3 | FastAPI replicas (force-recreated on model update) |
| `embedding-worker` | CLIP inference for new image uploads |
| `backfill-worker` | Re-score images after model promotion |
| `postgres` | Metadata database |
| `qdrant` | Vector store (never touched during serving deploys) |
| `rabbitmq` | Celery message broker |
| `prometheus` | Metrics collection |
| `grafana` | Dashboards |
| `data-generator` | Synthetic traffic simulator |

---

## Technology Stack

| Category | Technology |
|---|---|
| ML Framework | PyTorch, Open-CLIP, Torchvision |
| Vector Search | Qdrant v1.13.6 |
| Task Queue | Celery 5.3.6 + RabbitMQ |
| Databases | PostgreSQL 15 + Qdrant |
| Object Storage | S3-compatible (Chameleon) |
| Dataset Versioning | DVC |
| Experiment Tracking | MLflow 3.9.0 |
| API Framework | FastAPI 0.111.0 |
| Monitoring | Prometheus + Grafana |
| Drift Detection | Evidently 0.4.33 |
| Container Orchestration | Docker Compose |
| CI/CD | GitHub Actions |
| Load Testing | Locust |
| Testing | pytest 8.0.0 |

---

## Key MLOps Properties

| Property | How it's implemented |
|---|---|
| **Reproducibility** | DVC manifests version training data; MLflow tracks every experiment |
| **Automated retraining** | Cron schedule + drift-triggered webhook |
| **Model versioning** | `current_production.yaml` committed on every promotion |
| **Promotion gate** | Metric comparison before any deployment |
| **Zero-downtime deploy** | nginx untouched; only serving-api containers recreated |
| **Canary monitoring** | 3-min post-deploy error rate check |
| **Automatic rollback** | git revert + redeploy if canary fails |
| **Escalation** | GitHub Issue opened if rollback itself fails |
| **Feedback loop** | User interactions logged → drift detection → retraining |
| **Observability** | Prometheus metrics + Grafana dashboards + alert rules |
