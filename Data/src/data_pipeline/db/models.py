from sqlalchemy import Column, String, Boolean, Float, Integer, DateTime, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class Image(Base):
    __tablename__ = "images"
    image_id       = Column(String, primary_key=True)
    image_uri      = Column(String, nullable=False)  # s3://photoprism-proj03/images/...
    storage_path   = Column(String, nullable=False)
    source_dataset = Column(String)                  # yfcc | ava_subset
    split          = Column(String)                  # train | val (deterministic: int(image_id[-1], 16) < 12)
    status         = Column(String, default="pending")
    created_at     = Column(DateTime, default=datetime.utcnow)
    updated_at     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ImageMetadata(Base):
    __tablename__ = "image_metadata"
    image_id       = Column(String, primary_key=True)
    text           = Column(Text)    # caption / description for semantic search
    source_dataset = Column(String)  # yfcc | ava_subset
    width          = Column(Integer)
    height         = Column(Integer)
    format         = Column(String)
    exif_json      = Column(Text)
    tags           = Column(Text)
    captured_at    = Column(DateTime)
    normalized_at  = Column(DateTime)


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
    aesthetic_score = Column(Float)
    model_version   = Column(String)
    timestamp       = Column(DateTime, default=datetime.utcnow)


class DatasetSnapshot(Base):
    __tablename__ = "dataset_snapshots"
    snapshot_id    = Column(String, primary_key=True)
    version_tag    = Column(String)
    manifest_path  = Column(String)
    record_count   = Column(Integer)
    split_strategy = Column(String)  # e.g. hash_hex_75_25
    created_at     = Column(DateTime, default=datetime.utcnow)
