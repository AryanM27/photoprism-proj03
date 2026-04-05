# Infrastructure Resource Sizing

Evidence collected via `docker stats --no-stream` after ingestion load on Chameleon KVM@TACC (m1.xlarge: 8 vCPU, 16 GB RAM).

| Service           | Observed CPU | Observed Mem | Notes                                      |
|-------------------|-------------|-------------|---------------------------------------------|
| postgres          | 0.01%       | 58.13MiB    | Metadata inserts under ingestion load       |
| rabbitmq          | 0.58%       | 182.5MiB    | 4 queues (ingestion/validation/embedding/backfill) |
| qdrant            | 5.76%       | 81.32MiB    | Vector upserts during embedding             |
| ingestion-worker  | 0.30%       | 725.4MiB    | Scanner + uploader, PIL resize              |
| validation-worker | 0.31%       | 798MiB      | Validation checks + EXIF extraction         |
| embedding-worker  | 0.31%       | 899MiB      | CLIP model inference (clip-ViT-B-32)        |
| backfill-worker   | 0.20%       | 574.4MiB    | Task dispatch only, low CPU                 |
| features-api      | 0.11%       | 101.3MiB    | FastAPI + stub encoder, idle                |
| prometheus        | 0.07%       | 100.1MiB    | Scraping all exporters                      |
| grafana           | 0.11%       | 163MiB      | Dashboard only                              |
| adminer           | 0.00%       | 39.59MiB    | DB UI, idle                                 |

**Disk usage per volume (block storage):**

| Volume                  | Size   |
|-------------------------|--------|
| /mnt/block/postgres     | 70M    |
| /mnt/block/rabbitmq     | 3.3M   |
| /mnt/block/qdrant       | 2.0M   |
| /mnt/block/prometheus   | 11M    |
| /mnt/block/grafana      | 1004K  |

> Collected via `docker stats --no-stream` and `du -sh` on Chameleon KVM@TACC (m1.xlarge: 8 vCPU, 16 GB RAM), 2026-04-05.
