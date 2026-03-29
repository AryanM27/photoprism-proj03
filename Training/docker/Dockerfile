FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY Training/requirements-training.txt /app/Training/requirements-training.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/Training/requirements-training.txt

COPY Training /app/Training

WORKDIR /app/Training

CMD ["python", "-m", "src.semantic.evaluate", "--config", "configs/semantic/openclip_vitb32_baseline.yaml"]