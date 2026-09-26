# ECG Edge — Multi-Label Time-Series Classification with Edge Deployment

A 1D CNN for multi-label classification of 12-channel ECG signals, trained on the PTB-XL benchmark and optimized with ONNX and INT8 quantization. A Python ONNX Runtime edge demo publishes classification alerts over MQTT; the application stack is planned next.

---

## Overview

This project builds a reproducible ML pipeline for multi-label ECG classification and edge deployment:

- **Data pipeline** — 19,624 twelve-channel signals, bandpass filtered, z-normalized, cached
- **Model** — custom 1D ResNet-style CNN (3.86M parameters), no pretrained weights
- **Training** — weighted BCE loss for class imbalance, time-series augmentation
- **Evaluation** — macro F1, per-class AUC, confusion matrix, on standardized splits
- **Optimization** — ONNX export + INT8 quantization (about 4× smaller)
- **Edge demo** — Python ONNX Runtime inference, MQTT publisher, and live subscriber
- **Planned application stack** — FastAPI, React, PostgreSQL, and Docker

---

## Current Progress

| Phase | Status | What's done |
|-------|--------|-------------|
| 1 | Complete | Dataset pipeline verified on 19,624 records |
| 2 | Complete | 1D ResNet CNN trained and evaluated; test macro F1 = 0.6895 |
| 3 | Complete | ONNX export and INT8 quantization; 4× smaller with 0.16% macro F1 loss |
| 4 | Complete | Python ONNX Runtime inference + MQTT publisher/subscriber demo |
| 5 | Planned | FastAPI + React + PostgreSQL + Docker |

**Built and tested:**
- Reproducible data pipeline (download, filter, normalize, cache to `.npy`)
- Config system with YAML → typed attribute access
- Multi-label mapping from 71 SCP codes → 5 superclasses
- 1D ResNet CNN with 3,861,893 parameters, trained on a Google Colab T4
- ONNX FP32 export verified against PyTorch (maximum absolute difference < 1e-6)
- INT8 ONNX model and full held-out test-set evaluation
- Python edge inference wrapper publishing predictions to MQTT, with a live subscriber receiving alerts

**Verified preprocessed outputs** (cached as `.npy`, 4.4 GB total):

| Split | Shape | Records |
|-------|-------|---------|
| train | `(15673, 5000, 12)` | 15,673 |
| val | `(1967, 5000, 12)` | 1,967 |
| test | `(1984, 5000, 12)` | 1,984 |

**Class distribution** (consistent across all splits):
- NORM 48% · CD 28% · STTC 25% · MI 12% · HYP 11.5%

---

## Test Set Results

The best checkpoint was selected using validation macro F1. Results below are from the held-out test set (1,984 records), using a 0.5 decision threshold.

| Metric | Value |
|--------|-------|
| **Macro F1** | **0.6895** |
| Micro F1 | 0.7578 |
| Weighted F1 | 0.7636 |

| Class | F1 | Precision | Recall | AUC | Support |
|-------|----|-----------|--------|-----|---------|
| NORM | 0.8758 | 0.8180 | 0.9423 | 0.9462 | 954 |
| MI | 0.5461 | 0.4591 | 0.6738 | 0.8964 | 233 |
| STTC | 0.7697 | 0.6980 | 0.8577 | 0.9359 | 485 |
| CD | 0.7665 | 0.7320 | 0.8043 | 0.9154 | 557 |
| HYP | 0.4897 | 0.4185 | 0.5901 | 0.8433 | 222 |

**Observations:**
- Per-class AUC (0.84–0.95) exceeds thresholded F1, so ranking quality is stronger than the fixed 0.5 operating point suggests.
- MI and HYP have lower F1, consistent with their lower support.
- Training diagnostics show a widening train/validation gap; see [`notebooks/03_training_diagnostics.ipynb`](notebooks/03_training_diagnostics.ipynb).

---

## Quantization Results

Benchmarked the PyTorch checkpoint and ONNX variants on the full test set (1,984 samples) using a Windows x86 CPU. Latency is hardware- and runtime-dependent.

| Model | Size | Macro F1 | Latency |
|-------|------|----------|---------|
| PyTorch FP32 | 46.4 MB | 0.6895 | 96.9 ms/sample |
| ONNX FP32 | 15.4 MB | 0.6895 | 126.5 ms/sample |
| **ONNX INT8** | **3.9 MB** | **0.6884** | 1290.6 ms/sample |

INT8 reduced ONNX model size by about 75% with a 0.0011 absolute macro F1 difference (about 0.16% relative). It was slower on this CPU, so quantization should be viewed as a size optimization here; latency benefits depend on hardware and execution provider.

**Artifacts:**
- `models/ecg_model_fp32.onnx` — framework-agnostic FP32 model
- `models/ecg_model_int8.onnx` — quantized ONNX model
- [`results/quantization_benchmark.json`](results/quantization_benchmark.json) — benchmark report
- [`results/metrics.json`](results/metrics.json) — test-set metrics

---

## Edge Inference + MQTT Demo

The Phase 4 demo loads the INT8 ONNX model, runs inference on preprocessed test signals, and publishes classification results to an MQTT broker. A separate subscriber receives and prints the alerts. Only prediction metadata is sent; raw ECG signals are not published.

Install the MQTT client dependency if it is not already available:

```bash
pip install paho-mqtt
```

With the dataset, preprocessed test arrays, and `models/ecg_model_int8.onnx` in place, run the subscriber in one terminal:

```bash
python -m edge.subscriber
```

Then run the demo in another terminal:

```bash
python -m edge.run_demo --n 10
```

The demo defaults to device ID `edge-001` and publishes to `ecg-edge/edge-001/predictions`. The subscriber listens to `ecg-edge/+/predictions`. Options include `--device-id` and `--delay`, for example `python -m edge.run_demo --device-id pi-01 --n 5 --delay 0.2`.

In the recorded 10-sample run, the subscriber received all 10 alerts. Nine samples matched when judged by the demo's top-class comparison. On the remaining sample, the true label was CD, the top class was NORM, and CD was also present in the model's thresholded `detected` classes. This illustrates why multi-label output should be interpreted using the detected class set, not only the single top class. This small demo is an integration smoke test, not a model accuracy estimate; use the held-out test results above for evaluation.

The demo uses the public `test.mosquitto.org:1883` broker without authentication. Do not send sensitive data or use this broker for production; production deployments need a private broker and appropriate transport security and authentication. The recorded inference latency was about 2.4–4.7 seconds per sample on the tested CPU. Latency on other hardware must be measured rather than inferred from this run.

---

## Why This Project

Continuous monitoring generates continuous data, but cloud-based inference has three problems:

- **Latency** — time-critical events need millisecond alerts, not seconds
- **Privacy** — raw signals are sensitive; sending them to the cloud exposes them
- **Bandwidth/Power** — streaming raw data drains battery and costs money

**Solution:** Run inference on the edge. Only the classification result is transmitted.

---

## Dataset: PTB-XL

- **Source:** [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
- **Size:** 21,799 raw records from 18,869 patients → 19,624 after label filtering
- **Format:** 12-channel time-series, 10 seconds per record, 500 Hz (5000 timesteps per channel)
- **Labels:** Expert-annotated, 71 fine-grained statements mapped to 5 diagnostic superclasses
- **Splits:** Standardized 10-fold split (folds 1–8 train, 9 validation, 10 test)

### Why PTB-XL

| Criterion | Why it matters |
|-----------|---------------|
| Community standard | Results are directly comparable to published baselines |
| Standard splits | Reproducible; no ambiguity in train/val/test |
| Expert labels | High-quality targets, not crowdsourced |
| Multi-label | Realistic — records can belong to multiple classes simultaneously |
| Right size | Large enough to train a deep model, small enough for free-tier compute |

### Alternative datasets considered

- **MIT-BIH** — 48 records, too small for deep learning
- **CODE** — 1.6M records, too large for free-tier compute, few published baselines
- **CPSC 2018 / Chapman-Shaoxing** — smaller, less community standardization

---

## Labels: 5 Diagnostic Superclasses

| Class | Meaning |
|-------|---------|
| NORM | Normal |
| MI | Myocardial infarction |
| STTC | ST/T-wave change |
| CD | Conduction disturbance |
| HYP | Hypertrophy |

Labels are **multi-label** — a single record can belong to more than one class. This requires binary cross-entropy loss and per-class sigmoid outputs (not softmax).

---

## Architecture

```text
Raw signal (12 channels × 5000 timesteps)
↓
Bandpass filter (0.5–40 Hz) + z-normalization
↓
1D ResNet-style CNN (3.86M params)
↓
Sigmoid → 5 independent class probabilities
↓
ONNX export + INT8 quantization
↓
Python edge inference (ONNX Runtime) → MQTT alerts → planned application stack
```

---

## Domain Shift: A Critical Limitation

The training data comes from a specific source population. Deployment targets may differ in:

- **Disease prevalence** — different conditions dominate in different regions
- **Population characteristics** — signal morphology varies across demographics
- **Equipment and signal quality** — clinical-grade vs consumer wearable

Public datasets may not represent the intended deployment population (for example, populations and clinical settings in Kenya or elsewhere in Africa). This is a significant limitation: the model may not generalize without external validation and possible domain adaptation.

Proposed mitigation:
- Domain adaptation via fine-tuning on target-domain samples
- Federated learning across institutions
- Self-supervised pretraining on unlabeled target data
- Data collection partnerships

These are proposed mitigations; no target-domain validation has been performed yet.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/charllote122/ecg-edge.git
cd ecg-edge

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate         # Linux/macOS
# .venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download and extract PTB-XL from PhysioNet into data/raw/ptb-xl/
#    (The dataset is not included in this repository.)

# 5. Preprocess (filter, normalize, cache as .npy)
python -m src.data.preprocess

# 6. Train the model
python -m src.training.train

# 7. Evaluate the best checkpoint on the test set
python -m src.training.evaluate

# 8. Export and quantize to ONNX INT8
python -m src.export.to_onnx
python -m src.export.quantize
python -m src.export.benchmark
```

---

## Project Structure

```text
ecg-edge/
├── configs/                    Hyperparameters and paths
├── data/
│   ├── raw/                    Raw dataset (gitignored, ~2.6 GB)
│   └── processed/              Cached .npy tensors (gitignored, ~4.4 GB)
├── docs/                       Reserved for project documentation
├── edge/                       ONNX Runtime inference and MQTT demo/subscriber
├── models/                     Trained checkpoints and ONNX exports (gitignored)
├── notebooks/                  Exploration and diagnostics
├── results/                    Test metrics and benchmark report
├── scripts/                    Reserved for helper scripts (currently empty)
├── src/
│   ├── data/                   PTB-XL metadata, label mapping, preprocessing, Dataset
│   ├── models/                 1D ResNet CNN definition
│   ├── training/               Training loop, evaluation, metrics
│   ├── export/                 ONNX export, quantization, benchmark
│   └── utils/                  Config, logging, seeding
├── tests/                      Unit tests for data pipeline
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Dataset | PTB-XL | 19,624 labeled 12-channel records |
| Language | Python 3.10+ | Training and model export |
| Training | PyTorch 2.6 | Custom 1D CNN, random initialization |
| Preprocessing | WFDB, SciPy, NumPy | Load, filter, normalize |
| Model | 1D ResNet CNN | <5M parameters |
| Imbalance | Weighted BCE + augmentation | Handle minority classes |
| Export | ONNX + ONNX Runtime | Framework-agnostic inference |
| Quantization | ONNX Runtime INT8 | About 4× smaller than ONNX FP32 |
| Edge | Python + ONNX Runtime | INT8 inference demo |
| Messaging | MQTT | Prediction-alert publisher and subscriber demo |
| Backend | FastAPI | Planned HTTP + WebSocket API |
| Frontend | React + Tailwind CSS | Planned signal + alert dashboard |
| Database | PostgreSQL + pgvector | Planned audit log and version tracking |
| Deployment | Docker / Hugging Face Spaces | Planned deployment target |

---

## Key Design Decisions

| Decision | Reasoning |
|----------|-----------|
| 500 Hz sampling | Preserves fine waveform morphology for CD/HYP classes |
| 0.5–40 Hz bandpass | Removes baseline wander (<0.5) and powerline/muscle noise (>40) |
| Zero-phase filtering | No time shift — waveform landmarks stay aligned |
| Per-channel z-normalization | Removes amplitude variability while preserving inter-channel structure |
| Multi-label BCE loss | Records can have multiple simultaneous diagnoses |
| Weighted loss + augmentation | Dataset is imbalanced; rewards minority-class accuracy |
| Official 10-fold split | Comparable to published baselines; no leakage |
| Cached preprocessed `.npy` | Preprocessing runs once, training runs many times |
| Dynamic INT8 quantization | Reduces model size without retraining; latency gains depend on hardware |

---

## Evaluation Metrics

- **Macro F1** — primary metric; treats all classes equally despite imbalance
- **Per-class AUC** — threshold-independent ranking quality per class
- **Confusion matrix** — diagnostics on where the model fails
- **Per-class precision/recall** — context for F1

---

## Roadmap (5 Phases)

- [x] **Phase 1** — Data acquisition, preprocessing, reproducible pipeline
- [x] **Phase 2** — 1D ResNet CNN, training and evaluation (test macro F1: 0.6895)
- [x] **Phase 3** — ONNX export + INT8 quantization (about 4× smaller, about 0.16% relative F1 loss)
- [x] **Phase 4** — Python ONNX Runtime edge inference + MQTT publisher/subscriber demo
- [ ] **Phase 5** — FastAPI + React + PostgreSQL + Docker deployment

---

## License

MIT — see [LICENSE](LICENSE).