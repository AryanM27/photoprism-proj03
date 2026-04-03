FROM python:3.11-slim

WORKDIR /serving

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/
COPY app/ app/

RUN mkdir -p /serving/onnx_models && \
    python scripts/export_onnx.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
