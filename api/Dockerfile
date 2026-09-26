# -----------------------------------------------------------------------------
# ECG Edge API
# -----------------------------------------------------------------------------
FROM python:3.13-slim

# System deps: psycopg2 needs libpq, wfdb needs no extras, ONNX Runtime is CPU wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY edge/ ./edge/
COPY api/ ./api/
COPY configs/ ./configs/

# Copy model (small enough to include)
COPY models/ecg_model_int8.onnx ./models/

# Create a non-root user for safety
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
