from sqlalchemy import Column, String, Boolean, Float, Integer, DateTime, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class Image(Base):
    __tablename__ = "images"
    image_id         = Column(String, primary_key=True)
    image_uri        = Column(String, nullable=False)  # s3://training-module-proj03/raw/<id>.<ext>
    storage_path     = Column(String, nullable=False)
    source_dataset   = Column(String)                  # yfcc | ava_subset
    split            = Column(String)                  # train | val | test (deterministic: last hex digit of image_id)
    status           = Column(String, default="pending")
    embedding_status = Column(String, default="pending")  # pending | embedded | failed
    embedded_at      = Column(DateTime, nullable=True)
    model_version    = Column(String, nullable=True)      # e.g. clip-ViT-B-32
    created_at       = Column(DateTime, default=datetime.utcnow)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ImageMetadata(Base):
    __tablename__ = "image_metadata"
    image_id        = Column(String, primary_key=True)
    text            = Column(Text)    # caption / description for semantic search
    source_dataset  = Column(String)  # yfcc | ava_subset
    width           = Column(Integer)
    height          = Column(Integer)
    format          = Column(String)
    exif_json       = Column(Text)
    tags            = Column(Text)
    captured_at     = Column(DateTime)
    normalized_at   = Column(DateTime)
    aesthetic_score         = Column(Float, nullable=True)   # 0.0–1.0; stored pre-scale, normalised to 0–10 in manifests
    dataset_aesthetic_score = Column(Float, nullable=True)
    aesthetic_model_version = Column(String, nullable=True)
    aesthetic_score_date    = Column(DateTime, nullable=True)


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    job_id        = Column(String, primary_key=True)
    image_id      = Column(String, nullable=False)
    job_type      = Column(String, nullable=False)
    status        = Column(String, default="queued")
    error_message = Column(Text)     # failure reason if status=failed
    retry_count   = Column(Integer, default=0)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __init__(self, **kwargs):
        kwargs.setdefault("status", "queued")
        super().__init__(**kwargs)


class FeedbackEvent(Base):
    __tablename__ = "feedback_events"
    event_id        = Column(String, primary_key=True)
    user_id         = Column(String)
    query_id        = Column(String)
    image_id        = Column(String, nullable=False)
    shown_rank      = Column(Integer)
    clicked         = Column(Boolean)
    favorited       = Column(Boolean)
    semantic_score  = Column(Float)
    model_version   = Column(String)
    timestamp       = Column(DateTime, default=datetime.utcnow)


class DatasetSnapshot(Base):
    __tablename__ = "dataset_snapshots"
    snapshot_id    = Column(String, primary_key=True)
    version_tag    = Column(String)
    manifest_path  = Column(String)
    record_count   = Column(Integer)
    split_strategy = Column(String)  # e.g. hash_hex_62_19_19
    manifest_type  = Column(String)  # "semantic" or "aesthetic"
    created_at     = Column(DateTime, default=datetime.utcnow)


class DriftMetric(Base):
    __tablename__ = "drift_metrics"
    id                    = Column(Integer, primary_key=True, autoincrement=True)
    version_tag           = Column(Text, nullable=False)
    reference_version_tag = Column(Text, nullable=True)
    manifest_kind         = Column(Text, nullable=False)   # 'semantic' | 'aesthetic'
    column_name           = Column(Text, nullable=False)
    drift_score           = Column(Float, nullable=True)
    drift_detected        = Column(Boolean, nullable=True)
    stattest_name         = Column(Text, nullable=True)
    created_at            = Column(DateTime, default=datetime.utcnow)
