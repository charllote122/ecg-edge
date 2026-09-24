# ECG Edge — Multi-Label Time-Series Classification with Edge Deployment

A 1D CNN for multi-label classification of 12-channel time-series signals, trained on the PTB-XL benchmark, optimized with ONNX + INT8, and deployed end-to-end with FastAPI, React, MQTT, and PostgreSQL.

---

## Overview

This project implements a complete ML lifecycle for multi-label time-series classification:

- **Data pipeline** — 21,799 twelve-channel signals, bandpass filtered, z-normalized, cached
- **Model** — custom 1D ResNet-style CNN (<5M parameters), no pretrained weights
- **Training** — weighted BCE loss for class imbalance, time-series augmentation
- **Evaluation** — macro F1, per-class AUC, confusion matrix, on standardized splits
- **Optimization** — ONNX export + INT8 quantization (4× smaller, ~2× faster)
- **Edge deployment** — C++ inference wrapper, MQTT alerts only
- **Full stack** — FastAPI backend, React dashboard, PostgreSQL + pgvector audit log
- **Infrastructure** — Docker on Hugging Face Spaces



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
- **Size:** 21,799 records from 18,869 patients
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

## Domain Shift: A Critical Limitation

The training data comes from a specific source population. Deployment targets may differ in:

- **Disease prevalence** — different conditions dominate in different regions
- **Population characteristics** — signal morphology varies across demographics
- **Equipment and signal quality** — clinical-grade vs consumer wearable

**No public dataset exists for the target population** (e.g., Kenya / Africa / WHO regions). This is a real gap in medical AI. The model may not generalize without domain adaptation.

Proposed mitigation:
- Domain adaptation via fine-tuning on target-domain samples
- Federated learning across institutions
- Self-supervised pretraining on unlabeled target data
- Data collection partnerships

See [`docs/domain_shift.md`](docs/domain_shift.md) for the full discussion.



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

# 4. Download the dataset (~2.6 GB extracted)
bash scripts/download_data.sh

# 5. Preprocess (filter, normalize, cache as .npy)
bash scripts/preprocess_data.sh

# 6. Train the model (Phase 2)
python -m src.training.train



---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Dataset | PTB-XL | 21,799 labeled 12-channel records |
| Language | Python 3.11 | Training, backend, orchestration |
| Training | PyTorch | Custom 1D CNN, random initialization |
| Preprocessing | WFDB, SciPy, NumPy | Load, filter, normalize |
| Model | 1D ResNet CNN | <5M parameters |
| Imbalance | Weighted BCE + augmentation | Handle minority classes |
| Export | ONNX + ONNX Runtime | Framework-agnostic inference |
| Quantization | INT8 | 4× smaller, ~2× faster |
| Edge | C++ + ONNX Runtime | Deployed inference |
| Messaging | MQTT | Alert-only communication |
| Backend | FastAPI | HTTP + WebSocket |
| Frontend | React + Tailwind CSS | Signal + alert dashboard |
| Database | PostgreSQL + pgvector | Audit log, version tracking |
| Deployment | Docker on HF Spaces | Live demo |



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




---

## Evaluation Metrics

- **Macro F1** — primary metric; treats all classes equally despite imbalance
- **Per-class AUC** — threshold-independent ranking quality per class
- **Confusion matrix** — diagnostics on where the model fails
- **Per-class precision/recall** — context for F1

---

## Status

Phase 1 complete. Currently implementing Phase 2 (model training and evaluation).

---

## License

MIT — see [LICENSE](LICENSE).