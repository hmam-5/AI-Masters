<div align="center">

# AI Masters

### Brain Tumor Detection & Classification Framework

**Enterprise-grade multi-model ensemble system for automated brain MRI analysis**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](#tech-stack)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](#tech-stack)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white)](#tech-stack)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](#tech-stack)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](#quick-start)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

---

**Upload a brain MRI scan. Get an instant AI diagnosis powered by 4 neural networks.**

[Quick Start](#quick-start) · [How It Works](#how-it-works) · [API Reference](#api-reference) · [Training](#model-training) · [Architecture](#architecture)

</div>

---

## Overview

AI Masters is a full-stack diagnostic platform that analyzes brain MRI scans using an ensemble of 4 deep learning models running in parallel. Each model independently classifies the scan, and the system combines their predictions into a single consensus diagnosis with a confidence score.

**Supported classifications:**

| Class | Description |
|-------|-------------|
| `glioma` | Glioma tumor detected |
| `meningioma` | Meningioma tumor detected |
| `pituitary` | Pituitary tumor detected |
| `no_tumor` | No tumor detected |

**Ensemble models:**

| Model | Architecture | Strength |
|-------|-------------|----------|
| **Custom CNN** | Lightweight 2D CNN | Fast inference, low resource usage |
| **ResNet-50** | 50-layer residual network | Deep hierarchical feature extraction |
| **EfficientNet-B0** | Compound-scaled network | Best accuracy-to-compute ratio |
| **DenseNet-121** | Dense connectivity pattern | Feature reuse, parameter efficiency |

All 4 models run simultaneously via Celery workers. The final prediction uses weighted voting with Test-Time Augmentation (TTA) for robustness.

---

## Quick Start

### Prerequisites

You only need one thing installed:

- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (includes Docker Compose)

> No Python, Node.js, or GPU drivers required. Docker handles everything.

### Step 1 — Clone or download this project

Place the project folder anywhere on your machine.

### Step 2 — Open a terminal in the project folder

**Windows:**
```
Right-click the project folder → "Open in Terminal"
```

**macOS / Linux:**
```bash
cd /path/to/brain-tumor-ai-framework
```

### Step 3 — Build and start all services

```bash
docker compose up --build
```

> First run downloads ~2 GB of Docker images. This is a one-time setup.

### Step 4 — Wait for startup to complete

Watch for this line in the terminal output:

```
brain-tumor-backend  | INFO:     Application startup complete.
```

Once you see it, all 7 services are ready.

### Step 5 — Open the application

| URL | What It Opens |
|-----|---------------|
| **http://localhost** | Web application (main UI) |
| **http://localhost:8000/docs** | Interactive API documentation (Swagger) |
| **http://localhost:9001** | MinIO console (file storage admin) |

### Step 6 — Upload a brain MRI scan

1. Open **http://localhost**
2. Click **Upload** and select a brain MRI image (`.png`, `.jpg`, `.jpeg`, `.dcm`, `.nii`, `.nii.gz`)
3. The system runs all 4 models and shows:
   - Predicted tumor type
   - Confidence score per model
   - Ensemble consensus result
   - Clinical recommendations and findings

### Stopping the application

Press `Ctrl + C` in the terminal, then:

```bash
docker compose down
```

To also remove stored data (volumes):

```bash
docker compose down -v
```

---

## How It Works

```
                                ┌──────────────────┐
                                │   Upload MRI      │
                                │   (PNG/JPG/DICOM) │
                                └────────┬─────────┘
                                         │
                                         ▼
                               ┌─────────────────────┐
                               │   FastAPI Backend    │
                               │   Validates + stores │
                               │   image in MinIO     │
                               └────────┬────────────┘
                                        │
                                        ▼
                               ┌─────────────────────┐
                               │   Celery Worker      │
                               │   Picks up job from  │
                               │   Redis queue        │
                               └────────┬────────────┘
                                        │
                    ┌───────────┬───────┴────────┬───────────┐
                    ▼           ▼                ▼           ▼
              ┌──────────┐ ┌──────────┐  ┌──────────┐ ┌──────────┐
              │Custom CNN │ │ ResNet-50│  │EfficientN│ │DenseNet  │
              │           │ │          │  │et-B0     │ │-121      │
              └─────┬─────┘ └────┬─────┘  └────┬─────┘ └────┬─────┘
                    │            │              │            │
                    └────────────┴──────┬───────┴────────────┘
                                        │
                                        ▼
                              ┌──────────────────────┐
                              │  Weighted Ensemble    │
                              │  + Test-Time          │
                              │    Augmentation (TTA) │
                              └────────┬─────────────┘
                                       │
                              ┌────────┴─────────────┐
                              │                      │
                              ▼                      ▼
                    ┌─────────────────┐   ┌─────────────────┐
                    │ FalkorDB Graph  │   │ WebSocket Push   │
                    │ (store results) │   │ (live updates)   │
                    └─────────────────┘   └─────────────────┘
```

1. **Upload** — The frontend sends the MRI image to the backend API
2. **Store** — The backend validates the file and saves it to MinIO object storage
3. **Queue** — A Celery task is dispatched through the Redis message broker
4. **Inference** — The Celery worker loads all 4 PyTorch models and runs inference in parallel with TTA
5. **Ensemble** — Predictions are combined using weighted voting to produce a consensus
6. **Persist** — Results are stored in the FalkorDB graph database for analytics and history
7. **Notify** — The frontend receives real-time updates via WebSocket

---

## Architecture

### Services (7 containers)

| Service | Image | Port | Role |
|---------|-------|------|------|
| **Nginx** | `nginx:alpine` | `80` | Reverse proxy — routes `/api/*` to backend, everything else to frontend |
| **Frontend** | Node 18 → Nginx (multi-stage) | `3000` | React + TypeScript web interface |
| **Backend** | Python 3.11 | `8000` | FastAPI REST API + WebSocket server |
| **Celery Worker** | Python 3.11 | — | Async inference worker (GPU-optional) |
| **Redis** | `redis:7-alpine` | `6379` | Message broker + result backend for Celery |
| **FalkorDB** | `falkordb/falkordb` | `6381` | Graph database — stores patients, scans, results, analytics |
| **MinIO** | `minio/minio` | `9000` / `9001` | S3-compatible object storage for MRI files |

### Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Backend** | FastAPI + Pydantic | 0.104 / 2.5 |
| **ML Framework** | PyTorch + TorchVision | 2.1 |
| **Task Queue** | Celery + Redis | 5.3 |
| **Graph Database** | FalkorDB | 1.0 |
| **Object Storage** | MinIO (S3-compatible) | Latest |
| **Frontend** | React + TypeScript | 18 |
| **HTTP Client** | Axios | 1.6 |
| **Reverse Proxy** | Nginx | Alpine |
| **Containerization** | Docker Compose | v2 |

---

## Project Structure

```
brain-tumor-ai-framework/
│
├── docker-compose.yml                 # Orchestrates all 7 services
├── Dockerfile.backend                 # Python 3.11 backend image
├── Dockerfile.frontend                # Multi-stage Node → Nginx image
├── Dockerfile.worker                  # Celery worker image
│
├── backend/
│   ├── requirements.txt               # Python dependencies (26 packages)
│   └── app/
│       ├── main.py                    # FastAPI entry point + lifespan events
│       ├── train.py                   # CLI training script for all 4 models
│       ├── api/
│       │   └── routes.py              # 13 REST + WebSocket endpoints
│       ├── config/
│       │   └── settings.py            # Pydantic settings (env-based config)
│       ├── dataset/
│       │   ├── models.py              # MODEL_REGISTRY — 4 neural network architectures
│       │   ├── ensemble.py            # Ensemble inference + TTA logic
│       │   ├── trainer.py             # Training loop (per-model)
│       │   └── downloader.py          # Kaggle dataset download + folder organization
│       ├── models/
│       │   └── database.py            # JobStatus enum + TumorSubregion definitions
│       ├── services/
│       │   ├── graph_db.py            # FalkorDB client — Cypher queries + schema
│       │   └── storage.py             # MinIO/S3 storage abstraction
│       ├── workers/
│       │   └── celery_worker.py       # analyze_image task — runs ensemble inference
│       └── utils/
│           └── validators.py          # File validators (DICOM, NIfTI, standard images)
│
├── frontend/
│   ├── package.json                   # React + TypeScript + Axios
│   ├── tsconfig.json                  # TypeScript configuration
│   ├── nginx.conf                     # Standalone frontend Nginx config
│   ├── public/
│   │   └── index.html                 # HTML entry point
│   └── src/
│       ├── index.tsx                  # React DOM render
│       ├── App.tsx                    # Root component
│       ├── components/
│       │   ├── Dashboard.tsx          # Main layout — orchestrates all panels
│       │   ├── CommandEntry.tsx       # Landing page — model info + features
│       │   ├── SmartIngestion.tsx     # File upload with validation preview
│       │   ├── PipelineView.tsx       # Processing pipeline visualization
│       │   ├── DiagnosticWorkspace.tsx# Results — 4-model cards + recommendations
│       │   ├── PipelineLog.tsx        # Real-time processing log
│       │   └── AIChatAssistant.tsx    # AI assistant chat panel
│       ├── services/
│       │   └── api.ts                 # Axios API client (4 methods)
│       ├── types/
│       │   └── api.ts                 # TypeScript interfaces
│       └── styles/
│           └── crw.css                # Complete design system (~1500 lines)
│
├── models/                            # Pre-trained weights (mounted into containers)
│   ├── brain_tumor_2d.pth             # Custom CNN weights
│   ├── brain_tumor_resnet50.pth       # ResNet-50 weights
│   ├── brain_tumor_efficientnet.pth   # EfficientNet-B0 weights
│   └── brain_tumor_densenet.pth       # DenseNet-121 weights
│
├── data/                              # Training datasets (optional, for retraining)
│   ├── combined/                      # Organized train/val splits
│   ├── datasets/                      # Kaggle-sourced data
│   └── raw/                           # Raw downloaded datasets
│
└── infra/
    └── nginx.conf                     # Production reverse proxy config
```

---

## API Reference

Base URL: `http://localhost:8000/api/v1`

Interactive docs: **http://localhost:8000/docs** (Swagger UI)

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Upload an MRI image for ensemble analysis. Returns a `job_id`. |
| `GET` | `/inference/{job_id}/results` | Retrieve analysis results for a completed job. |
| `WS` | `/ws/job/{job_id}` | WebSocket — receive real-time job status updates. |
| `GET` | `/health` | Service health check. |

### Analytics & History

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/analytics/history/{mrn}` | Full analysis history for a patient (by MRN). |
| `GET` | `/analytics/grades` | Aggregated statistics by tumor grade. |
| `GET` | `/analytics/dataset` | Dataset composition overview. |
| `GET` | `/analytics/training` | Training run history and metrics. |

### Graph Database

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/graph/stats` | Global tumor type statistics from the graph. |
| `GET` | `/graph/patient/{mrn}/history` | Patient scan history from the graph. |
| `GET` | `/graph/similar/{tumor_grade}` | Find similar cases by tumor grade. |

### Model Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/training/start` | Start training pipeline for all 4 models. |
| `GET` | `/ensemble/status` | Check loaded models and ensemble readiness. |

### Example: Analyze an image

```bash
# Upload an image
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@brain_scan.png"

# Response
{
  "job_id": "abc-123",
  "status": "processing"
}

# Get results
curl http://localhost:8000/api/v1/analyze/abc-123/results
```

---

## Model Training

The `models/` directory ships with 4 pre-trained `.pth` weight files. You can retrain from scratch if needed.

### Option A — Train via the API (easiest)

While the application is running:

```bash
curl -X POST http://localhost:8000/api/v1/training/start
```

This triggers the full training pipeline inside the backend container.

### Option B — Train via CLI

```bash
docker compose exec backend python -m app.train --store-graph
```

### Option C — Provide your own dataset

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) (free account required)
2. Organize folders under `data/combined/`:

```
data/combined/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
└── val/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/
```

3. Run training:

```bash
docker compose exec backend python -m app.train --store-graph
```

### Check model status

```bash
curl http://localhost:8000/api/v1/ensemble/status
```

---

## Environment Variables

All settings have sensible defaults. Override via a `.env` file in the project root or through `docker-compose.yml` environment sections.

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `CELERY_BROKER_URL` | `redis://redis:6379/0` | Celery broker URL |
| `MINIO_ENDPOINT` | `minio:9000` | MinIO server address |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `FALKORDB_HOST` | `falkordb` | FalkorDB server hostname |
| `FALKORDB_PORT` | `6379` | FalkorDB port |
| `STORAGE_BACKEND` | `minio` | Storage backend (`minio` or `s3`) |
| `ENSEMBLE_MODELS` | `custom_cnn,resnet50,efficientnet,densenet` | Active ensemble models |
| `DEBUG` | `false` | Enable debug mode |
| `ENVIRONMENT` | `development` | Runtime environment |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device for Celery worker |

### GPU Support (optional)

To enable GPU inference on the Celery worker, uncomment the `deploy` section in `docker-compose.yml`:

```yaml
celery-worker:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `docker compose` not found | Install Docker Desktop (v4.0+) which includes Compose v2 |
| Port 80 already in use | Stop other web servers or change the port in `docker-compose.yml` |
| Build fails on first run | Run `docker compose build --no-cache` to retry from scratch |
| Models not loading | Ensure the `models/` directory contains all 4 `.pth` files |
| MinIO connection refused | Wait for the health check — MinIO takes ~10s to start |
| Frontend shows blank page | Check that Nginx container is running: `docker compose ps` |
| Out of memory during inference | Reduce `inference_batch_size` in `backend/app/config/settings.py` |

### Useful Commands

```bash
# View all running containers
docker compose ps

# View backend logs
docker compose logs -f backend

# View Celery worker logs
docker compose logs -f celery-worker

# Restart a single service
docker compose restart backend

# Rebuild without cache
docker compose build --no-cache

# Remove everything (containers + volumes + images)
docker compose down -v --rmi all
```

---

## License

MIT
| **Nginx** | Reverse proxy |
| **Docker** | Containerization |

---

## Troubleshooting

**Can't open http://localhost?**
Make sure Docker Desktop is running and you ran `docker compose up --build`. Wait 1-2 minutes.

**Port already in use?**
Another app is using port 80. Close it or stop the other service.

**Want to start completely fresh?**
```bash
docker compose down -v
docker compose up --build
```

**Want to see logs?**
```bash
docker compose logs backend
docker compose logs celery-worker
```
```
