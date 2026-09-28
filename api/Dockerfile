# -----------------------------------------------------------------------------
# ECG Edge API — pure Python, no apt-get
# -----------------------------------------------------------------------------
FROM python:3.13-slim

# Build tools already present in the base image? No, but we don't need them:
# every package below ships a pre-built wheel.

WORKDIR /app

# Install PyTorch CPU-only first (avoids ~2.4 GB of CUDA wheels)
COPY requirements-torch.txt .
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    -r requirements-torch.txt

# Install the rest of the dependencies (all binary wheels)
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY edge/ ./edge/
COPY api/ ./api/
COPY configs/ ./configs/
COPY models/ecg_model_int8.onnx ./models/

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
