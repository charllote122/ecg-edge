# ECG Edge

A production-style ECG classification demo for multi-label cardiac diagnosis using PTB-XL data, a 1D CNN, ONNX inference, MQTT alerting, a FastAPI backend, PostgreSQL persistence, and a live React dashboard.

## Overview

This project turns a research-grade ECG model into a small edge AI system:

- trains a 1D ResNet-style CNN on 12-lead ECG signals
- maps 71 PTB-XL SCP codes into 5 diagnostic superclasses
- exports the model to ONNX and benchmarks FP32 vs INT8 variants
- runs inference on edge devices with ONNX Runtime
- publishes alert payloads over MQTT
- stores alerts in PostgreSQL
- displays recent predictions in a React dashboard

## What is implemented now

The repository is no longer just a model-training sandbox. It includes the full stack for alerting and monitoring:

- model training and evaluation pipeline in Python
- ONNX export and quantization output under `models/`
- edge inference engine in `edge/`
- MQTT publisher and subscriber demo
- FastAPI API with `/health`, `/predict`, and `/alerts`
- PostgreSQL database setup for alert persistence
- Vite + React dashboard in `frontend/`

## Project status

| Area | Status |
|---|---|
| PTB-XL preprocessing | Complete |
| Model training and evaluation | Complete |
| ONNX export and quantization | Complete |
| Edge inference | Complete |
| MQTT alerting | Complete |
| FastAPI API | Complete |
| PostgreSQL storage | Complete |
| React dashboard | Complete |

## Model and performance

### Test metrics

Using the held-out PTB-XL test split:

| Metric | Value |
|---|---:|
| Macro F1 | 0.6895 |
| Micro F1 | 0.7578 |
| Weighted F1 | 0.7636 |

Per-class results from `results/metrics.json`:

| Class | F1 | Precision | Recall | AUC |
|---|---:|---:|---:|---:|
| NORM | 0.8758 | 0.8180 | 0.9423 | 0.9462 |
| MI | 0.5461 | 0.4591 | 0.6738 | 0.8964 |
| STTC | 0.7697 | 0.6980 | 0.8577 | 0.9359 |
| CD | 0.7665 | 0.7320 | 0.8043 | 0.9154 |
| HYP | 0.4897 | 0.4185 | 0.5901 | 0.8433 |

### Quantization benchmark

| Model | Size | Macro F1 | Latency |
|---|---:|---:|---:|
| PyTorch FP32 | 46.4 MB | 0.6895 | 96.9 ms/sample |
| ONNX FP32 | 15.4 MB | 0.6895 | 126.5 ms/sample |
| ONNX INT8 | 3.9 MB | 0.6884 | 1290.6 ms/sample |

The INT8 model is much smaller, but it is slower on the tested CPU. It is best treated as a size-optimized deployment option, with hardware-specific latency validation required before production use.

## Architecture

```text
PTB-XL ECG data
  -> preprocessing and label mapping
  -> 1D CNN training
  -> ONNX export
  -> ONNX Runtime inference
  -> MQTT alert publication
  -> FastAPI ingestion
  -> PostgreSQL alert storage
  -> React dashboard
```

## Diagnostic classes

The model predicts 5 multi-label classes:

- NORM
- MI
- STTC
- CD
- HYP

These are multi-label outputs, not mutually exclusive classes. A single ECG can have multiple positive classes at the same time.

## Repository layout

```text
ecg-edge/
├── api/                  FastAPI application and database access
├── configs/              YAML configuration files
├── data/
│   ├── raw/              PTB-XL raw data
│   └── processed/        Cached .npy training/validation/test arrays
├── docs/                 project notes and docs
├── edge/                 inference, MQTT publisher/subscriber, demo scripts
├── frontend/             Vite + React dashboard
├── models/               checkpoints and ONNX exports
├── notebooks/            analysis and training diagnostics
├── results/              metrics and benchmark outputs
├── scripts/              helper scripts
├── src/                  training, preprocessing, modeling, export code
├── tests/                project tests
├── docker-compose.db.yml PostgreSQL service definition
├── requirements.txt      Python dependencies
├── pyproject.toml        packaging metadata
├── LICENSE
├── README.md
└── .gitignore
```

## Quickstart

### 1) Create a Python environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Start PostgreSQL

```bash
docker compose -f docker-compose.db.yml up -d
```

### 4) Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

The API exposes:

- `GET /health`
- `POST /predict`
- `GET /alerts`
- `POST /alerts`

### 5) Start the React dashboard

```bash
cd frontend
npm install
npm run dev
```

The dashboard reads from the API via the Vite proxy and shows recent alerts and alert statistics.

### 6) Generate edge alerts

Open a second terminal and run the MQTT publisher demo:

```bash
python -m edge.run_demo --n 10
```

Optional: forward those alerts into the API database:

```bash
python -m edge.mqtt_to_api
```

### 7) Watch live MQTT output

```bash
python -m edge.subscriber
```

## Data and preprocessing

The project uses PTB-XL, a large public ECG dataset with 12-lead recordings sampled at 500 Hz. The preprocessing pipeline includes:

- signal loading via WFDB
- 0.5–40 Hz bandpass filtering
- per-channel z-normalization
- label mapping from 71 SCP codes into 5 classes
- caching of processed arrays to `data/processed/`

## Notes on deployment

This project is intended for edge monitoring, not direct clinical use. The public MQTT broker used by the demo is for development/testing only. For real deployments, use a private broker with authentication and encrypted transport.

## License

MIT. See [LICENSE](LICENSE).

## References

- PTB-XL dataset: https://physionet.org/content/ptb-xl/1.0.3/
- ONNX Runtime: https://onnxruntime.ai/
- FastAPI: https://fastapi.tiangolo.com/
- React + Vite: https://vite.dev/
