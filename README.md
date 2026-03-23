<div align="center">

# AI Masters

### Brain Tumor Detection & Classification Framework

**Enterprise-grade multi-model ensemble system for automated brain MRI analysis**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](#tech-stack)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](#tech-stack)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)](#tech-stack)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](#tech-stack)
[![FalkorDB](https://img.shields.io/badge/FalkorDB-Graph_DB-FF6B35?logo=redis&logoColor=white)](#falkordb-graph-database)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](#option-1--docker-full-stack)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

---

**Upload a brain MRI scan. Get an instant AI diagnosis powered by 4 neural networks.**

[Local Setup](#option-2--local-development-recommended-for-beginners) · [Docker Setup](#option-1--docker-full-stack) · [Database Setup](#step-3--start-falkordb-graph-database) · [How It Works](#how-it-works) · [API Reference](#api-reference) · [Training](#model-training) · [Architecture](#architecture)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Option 1 — Docker (Full Stack)](#option-1--docker-full-stack)
- [Option 2 — Local Development (Recommended for Beginners)](#option-2--local-development-recommended-for-beginners)
  - [Step 1 — Install Python Dependencies](#step-1--install-python-dependencies)
  - [Step 2 — Install Frontend Dependencies](#step-2--install-frontend-dependencies)
  - [Step 3 — Start FalkorDB Graph Database](#step-3--start-falkordb-graph-database)
  - [Step 4 — Configure Environment](#step-4--configure-environment)
  - [Step 5 — Start the Backend](#step-5--start-the-backend)
  - [Step 6 — Start the Frontend](#step-6--start-the-frontend)
  - [Step 7 — Open the Application](#step-7--open-the-application)
  - [Step 8 — Seed the Database (Optional)](#step-8--seed-the-database-optional)
- [FalkorDB Graph Database](#falkordb-graph-database)
- [Dataset & Training](#dataset--training)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Overview

AI Masters is a full-stack diagnostic platform that analyzes brain MRI scans using an ensemble of 4 deep learning models. Each model independently classifies the scan, and the system combines their predictions into a single consensus diagnosis with a confidence score.

**Supported classifications:**

| Class | Description |
|-------|-------------|
| `glioma` | Glioma tumor detected |
| `meningioma` | Meningioma tumor detected |
| `pituitary` | Pituitary tumor detected |
| `no_tumor` | No tumor detected |

**Ensemble models:**

| Model | Architecture | Weight | Strength |
|-------|-------------|--------|----------|
| **Custom CNN** | Lightweight 2D CNN | 1.0 | Fast inference, low resource usage |
| **ResNet-50** | 50-layer residual network | 1.5 | Deep hierarchical feature extraction |
| **EfficientNet-B0** | Compound-scaled network | 1.5 | Best accuracy-to-compute ratio |
| **DenseNet-121** | Dense connectivity pattern | 1.3 | Feature reuse, parameter efficiency |

The final prediction uses **weighted logit fusion** with **Test-Time Augmentation** (TTA — original + 3 augmented variants) and temperature scaling (T=0.8) for calibrated, robust predictions.

---

## Prerequisites

Choose your setup path:

| Requirement | Docker Mode | Local Mode |
|-------------|:-----------:|:----------:|
| **Docker Desktop** | Required | Only for FalkorDB |
| **Python 3.11+** | Not needed | Required |
| **Node.js 18+** | Not needed | Required |
| **Git** | Recommended | Recommended |

> **Tip:** If you're new to the project, use **Local Mode** — it's simpler, faster to iterate, and doesn't require building Docker images.

---

## Option 1 — Docker (Full Stack)

This spins up all 10 services (backend, frontend, Celery worker, Redis, FalkorDB, MinIO, Nginx, Jaeger, Prometheus, Grafana) with a single command.

### Step 1 — Clone the project

```bash
git clone <your-repo-url>
cd brain-tumor-ai-framework
```

### Step 2 — Build and start all services

```bash
docker compose up --build
```

> First run downloads ~2 GB of Docker images. This is a one-time setup.

### Step 3 — Wait for startup

Watch for this line:

```
brain-tumor-backend  | INFO:     Application startup complete.
```

### Step 4 — Open the application

| URL | What It Opens |
|-----|---------------|
| **http://localhost** | Web application (main UI) |
| **http://localhost:8000/docs** | Interactive API documentation (Swagger) |
| **http://localhost:9001** | MinIO console (file storage admin) |
| **http://localhost:16686** | Jaeger UI (distributed tracing) |
| **http://localhost:9090** | Prometheus (metrics explorer) |
| **http://localhost:3001** | Grafana dashboards (admin/admin) |

### Stopping

```bash
# Stop all services
docker compose down

# Stop and remove all stored data
docker compose down -v
```

---

## Option 2 — Local Development (Recommended for Beginners)

This runs the backend and frontend directly on your machine. No need to build Docker images. Faster to start, easier to debug, and fully functional.

### What's different in Local Mode?

| Feature | Docker Mode | Local Mode |
|---------|-------------|------------|
| File storage | MinIO (S3) | Local filesystem (`./data/uploads/`) |
| Inference | Celery + Redis (async) | Synchronous (in-process) |
| Database | FalkorDB (auto-started) | FalkorDB via Docker (1 container) |
| Monitoring | Jaeger, Prometheus, Grafana | Disabled (logs only) |

> **Your AI models, accuracy, and results are identical in both modes.**

---

### Step 1 — Install Python Dependencies

Open a terminal in the **project root folder** (`brain-tumor-ai-framework/`):

**Windows (PowerShell):**
```powershell
# Create a virtual environment in the project root
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install all Python packages
pip install -r backend/requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

> **Important:** The virtual environment (`.venv/`) must be created in the **project root**, not inside `backend/`. This is because the backend references model files at `../models/` relative to itself.

<details>
<summary><strong>Key Python packages installed (42 packages)</strong></summary>

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.104.1 | Web framework |
| `uvicorn` | 0.24.0 | ASGI server |
| `torch` | 2.2.2 | Deep learning framework |
| `torchvision` | 0.17.2 | Image transforms & pretrained models |
| `falkordb` | 1.0.9 | Graph database client |
| `onnxruntime` | ≥1.16.3 | Optimized model inference |
| `pydantic` | 2.5.0 | Data validation |
| `python-jose` | 3.3.0 | JWT tokens |
| `passlib` | 1.7.4 | Password hashing (bcrypt) |
| `slowapi` | 0.1.9 | Rate limiting |
| `Pillow` | 10.2.0 | Image processing |
| `nibabel` | 5.1.0 | NIfTI medical image format |
| `pydicom` | 2.4.0 | DICOM medical image format |
| `kaggle` | 1.6.14 | Dataset downloader |
| `prometheus-fastapi-instrumentator` | 6.1.0 | Metrics |

</details>

---

### Step 2 — Install Frontend Dependencies

Open a **new terminal** (keep the backend terminal open):

```powershell
cd frontend
npm install
```

This installs React 18, TypeScript 5.3, Vite 5.4, and Axios.

---

### Step 3 — Start FalkorDB Graph Database

FalkorDB is the graph database that stores patients, scans, inference results, doctors, tags, and audit logs. It runs as a single Docker container.

```powershell
docker run -d -p 6381:6379 --name falkordb falkordb/falkordb:latest
```

**What this does:**
- Downloads the FalkorDB image (~150 MB, one-time)
- Starts the database on **port 6381** (mapped from container port 6379)
- Container name: `falkordb` (easy to manage)

**Verify it's running:**
```powershell
docker ps --filter name=falkordb
```

You should see:
```
CONTAINER ID   IMAGE                       STATUS          PORTS
abc123...      falkordb/falkordb:latest     Up 10 seconds   0.0.0.0:6381->6379/tcp
```

<details>
<summary><strong>FalkorDB management commands</strong></summary>

```powershell
# Stop the database (data is preserved)
docker stop falkordb

# Start it again later
docker start falkordb

# Remove the container entirely (data lost)
docker rm -f falkordb

# View database logs
docker logs falkordb
```

</details>

> **Can I skip FalkorDB?** Yes. Set `SKIP_FALKORDB=true` in the `.env` file (Step 4). The app will use an in-memory fallback. Data won't persist between restarts, but everything else works.

---

### Step 4 — Configure Environment

The backend needs a `.env` file to know it's running in local mode. A template is already provided:

```powershell
cd backend
copy .env.local .env
```

**macOS / Linux:**
```bash
cd backend
cp .env.local .env
```

The default `.env` is pre-configured for local development:

| Setting | Value | Meaning |
|---------|-------|---------|
| `STORAGE_BACKEND` | `local` | Store files on disk (not MinIO) |
| `USE_SYNC_INFERENCE` | `true` | Run inference synchronously (no Celery/Redis) |
| `FALKORDB_HOST` | `localhost` | FalkorDB on your machine |
| `FALKORDB_PORT` | `6381` | FalkorDB port (matches Docker command) |
| `SKIP_FALKORDB` | `false` | Use real FalkorDB (set `true` to skip) |
| `DEBUG` | `true` | Enable debug logging |

> **No other configuration needed.** All settings have sensible defaults.

---

### Step 5 — Start the Backend

Make sure your virtual environment is activated (you should see `(.venv)` in your terminal prompt).

```powershell
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output:**
```
INFO:     Loading ensemble models...
INFO:     ✓ custom_cnn loaded (brain_tumor_2d.pth)
INFO:     ✓ resnet50 loaded (brain_tumor_resnet50.pth)
INFO:     ✓ efficientnet loaded (brain_tumor_efficientnet.pth)
INFO:     ✓ densenet loaded (brain_tumor_densenet.pth)
INFO:     FalkorDB connected (localhost:6381)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Verify the backend is healthy:**
```powershell
curl http://localhost:8000/api/v1/health
```

Response: `{"status": "ok"}`

> **Keep this terminal running.** The `--reload` flag auto-restarts on code changes.

---

### Step 6 — Start the Frontend

Open a **second terminal** (the backend must stay running):

```powershell
cd frontend
npm run dev
```

**Expected output:**
```
  VITE v5.4.x  ready in 300ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: http://192.168.x.x:3000/
```

The frontend dev server starts on **port 3000** and automatically proxies API calls (`/api/*`) to the backend on port 8000.

---

### Step 7 — Open the Application

| URL | What It Opens |
|-----|---------------|
| **http://localhost:3000** | Web application (main UI) |
| **http://localhost:8000/docs** | API documentation (Swagger UI) |
| **http://localhost:8000/api/v1/health** | Backend health check |

**To analyze a brain MRI scan:**

1. Open **http://localhost:3000**
2. Click **Upload** and select a brain MRI image (`.png`, `.jpg`, `.jpeg`, `.dcm`, `.nii`, `.nii.gz`)
3. Wait for all 4 models to finish (progress shown in real-time)
4. View:
   - Ensemble consensus prediction
   - Individual model predictions with confidence scores
   - Clinical recommendations
   - Grad-CAM heatmap (which brain region triggered the diagnosis)

---

### Step 8 — Seed the Database (Optional)

After starting the backend and FalkorDB, you can populate the database with sample data (model versions, demo patient, doctors, audit logs):

```powershell
cd backend
python seed_falkordb.py
```

**Expected output:**
```
=======================================================
  FalkorDB Seeder — Brain Tumor AI Framework
=======================================================

[1/7] Initializing schema...
      Schema ready (indexes + tumor types/grades seeded).
[2/7] Registering model versions...
      + BrainTumorCNN v1.0.0 (acc=92%, path=../models/brain_tumor_2d.pth)
      + ResNet50 v1.0.0 (acc=95%, path=../models/brain_tumor_resnet50.pth)
      + EfficientNet-B0 v1.0.0 (acc=94%, path=../models/brain_tumor_efficientnet.pth)
      + DenseNet-121 v1.0.0 (acc=93%, path=../models/brain_tumor_densenet.pth)
[3/7] Registering dataset images...
      (skipped if no dataset downloaded yet)
[4/7] Registering demo patient and existing scans...
      Patient: DEMO-001
[5/7] Registering doctors...
      + Dr. Sarah Chen (Neuroradiology)
      + Dr. Ahmed Hassan (Diagnostic Radiology)
      + Dr. Maria Garcia (Neuropathology)
[6/7] Creating audit log entries...
      4 audit log entries created.
[7/7] Verifying database contents...

  ┌─────────────────────────────┬───────┐
  │ Node Label                  │ Count │
  ├─────────────────────────────┼───────┤
  │ AuditLog                    │    10 │
  │ TumorGrade                  │     5 │
  │ ModelVersion                │     4 │
  │ TumorType                   │     3 │
  │ Doctor                      │     3 │
  │ ...                         │   ... │
  ├─────────────────────────────┼───────┤
  │ TOTAL NODES                 │    36 │
  └─────────────────────────────┴───────┘
```

> The seeder is **idempotent** — running it multiple times won't create duplicates.

---

## Summary: Quick Start Commands (Copy & Paste)

Here's everything in order, from a fresh clone to a running application:

```powershell
# 1. Clone the project
git clone <your-repo-url>
cd brain-tumor-ai-framework

# 2. Start FalkorDB (requires Docker)
docker run -d -p 6381:6379 --name falkordb falkordb/falkordb:latest

# 3. Set up Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

# 4. Configure backend
cd backend
copy .env.local .env

# 5. Seed the database (optional)
python seed_falkordb.py

# 6. Start the backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**In a second terminal:**

```powershell
# 7. Start the frontend
cd frontend
npm install
npm run dev
```

**Open http://localhost:3000** — you're ready to analyze brain MRI scans!

---

## FalkorDB Graph Database

FalkorDB is a high-performance graph database built on Redis. It stores the entire clinical data model:

### Graph Schema (15 Entity Types)

| Entity | Properties | Key Relationships |
|--------|-----------|------------------|
| **Patient** | mrn, name, age, sex | HAS_SCAN, REVIEWED (M:N Doctor) |
| **Scan** | scan_id, modality, uploaded_at | ANALYZED_BY, TAGGED_WITH (M:N Tag) |
| **InferenceJob** | job_id, status, created_at | PRODUCED, ASSIGNED_TO (M:N Doctor) |
| **AnalysisResult** | result_id, confidence | CLASSIFIED_AS |
| **TumorType** | type_name | GRADED_AS |
| **TumorGrade** | grade, description | — |
| **SubregionResult** | region_name, probability | — |
| **ClassificationResult** | model_name, predicted_class | — |
| **DatasetImage** | path, split, class_label | — |
| **TrainingRun** | run_id, accuracy, loss | — |
| **Doctor** | doctor_id, name, specialty | REVIEWED (M:N Patient), ASSIGNED_TO (M:N Job) |
| **AuditLog** | log_id, action, timestamp | — |
| **ModelVersion** | version_id, architecture, active | SUPERSEDES (chain) |
| **Tag** | name | TAGGED_WITH (M:N Scan) |
| **User** | username, email, role, hashed_pw | — |

### Many-to-Many Relationships

- **Doctor ↔ Patient** (`:REVIEWED`) — a doctor can review multiple patients, a patient can have multiple doctors
- **Doctor ↔ InferenceJob** (`:ASSIGNED_TO`) — doctors assigned to specific analysis jobs
- **Scan ↔ Tag** (`:TAGGED_WITH`) — scans tagged with multiple labels

### Database Persistence

FalkorDB data lives inside the Docker container. It persists across restarts (`docker stop`/`docker start`) but is lost if the container is removed (`docker rm`).

**If you clone this project on a new machine**, the database will be empty. Run `seed_falkordb.py` to repopulate it:

```powershell
cd backend
python seed_falkordb.py
```

**To add persistent storage across container removal:**
```powershell
docker run -d -p 6381:6379 --name falkordb -v falkordb_data:/data falkordb/falkordb:latest
```

---

## Dataset & Training

### Pre-trained Models (Included)

The `models/` directory ships with 4 pre-trained `.pth` weight files. **No training is needed** to use the application — just upload an MRI and get predictions.

| File | Model | Accuracy |
|------|-------|----------|
| `brain_tumor_2d.pth` | Custom CNN | ~92% |
| `brain_tumor_resnet50.pth` | ResNet-50 | ~95% |
| `brain_tumor_efficientnet.pth` | EfficientNet-B0 | ~94% |
| `brain_tumor_densenet.pth` | DenseNet-121 | ~93% |

### Downloading the Dataset (For Retraining)

The training script can automatically download 3 Kaggle datasets and merge them:

**Step 1 — Set up Kaggle API credentials:**

1. Create a free account at [kaggle.com](https://www.kaggle.com)
2. Go to **Account** → **API** → **Create New Token**
3. This downloads `kaggle.json`. Place it at:
   - **Windows:** `C:\Users\<YourUser>\.kaggle\kaggle.json`
   - **macOS/Linux:** `~/.kaggle/kaggle.json`

**Step 2 — Download datasets:**

```powershell
cd backend
python -m app.train --download
```

This downloads and organizes 3 datasets into `data/combined/`:

| Dataset | Source | Images |
|---------|--------|--------|
| Brain Tumor Classification MRI | `sartajbhuvaji/brain-tumor-classification-mri` | ~3,000 |
| Brain Tumor MRI Dataset | `masoudnickparvar/brain-tumor-mri-dataset` | ~7,000 |
| Brain Tumors 256x256 | `thomasdubail/brain-tumors-256x256` | ~4,500 |

```
data/combined/
├── train/
│   ├── glioma/        # ~3,500 images
│   ├── meningioma/    # ~3,200 images
│   ├── pituitary/     # ~3,600 images
│   └── no_tumor/      # ~2,500 images
└── val/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/
```

### Training the Models

**Option A — Train from CLI (local mode):**

```powershell
cd backend
python -m app.train --store-graph
```

**Training CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--download` | — | Download datasets from Kaggle before training |
| `--data-dir` | `../data/combined` | Path to dataset directory |
| `--model-path` | `../models` | Path to save trained weights |
| `--epochs` | `15` | Number of training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--image-size` | `224` | Input image resolution |
| `--store-graph` | — | Save training metrics to FalkorDB |

**Option B — Train via the API (while the app is running):**

```bash
curl -X POST http://localhost:8000/api/v1/training/start
```

**Option C — Train in Docker:**

```bash
docker compose exec backend python -m app.train --store-graph
```

### Check Model Status

```bash
curl http://localhost:8000/api/v1/ensemble/status
```

---

## How It Works

### Local Mode Pipeline

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
                               │   image locally      │
                               └────────┬────────────┘
                                        │
                    ┌───────────┬───────┴────────┬───────────┐
                    ▼           ▼                ▼           ▼
              ┌──────────┐ ┌──────────┐  ┌──────────┐ ┌──────────┐
              │Custom CNN │ │ ResNet-50│  │EfficientN│ │DenseNet  │
              │  (w=1.0) │ │  (w=1.5) │  │et (w=1.5)│ │  (w=1.3) │
              └─────┬─────┘ └────┬─────┘  └────┬─────┘ └────┬─────┘
                    │            │              │            │
                    └────────────┴──────┬───────┴────────────┘
                                        │
                                        ▼
                              ┌──────────────────────┐
                              │  Weighted Ensemble    │
                              │  + TTA (4 variants)   │
                              │  + Temperature (0.8)  │
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

### Inference Pipeline (7 Steps)

The sync inference worker (`backend/app/workers/sync_inference.py`) executes these steps for each uploaded scan:

1. **Validate** — Check file type (PNG/JPG/DICOM/NIfTI), size limits
2. **Store** — Save the image to local filesystem under `data/uploads/scans/{patient_mrn}/`
3. **Preprocess** — Resize to 224×224, normalize, create TTA variants (original, horizontal flip, ±15° rotations)
4. **Inference** — Run all 4 models on each TTA variant
5. **Ensemble** — Weighted logit fusion (CNN:1.0, ResNet:1.5, EfficientNet:1.5, DenseNet:1.3) with temperature scaling
6. **Persist** — Save results to FalkorDB (InferenceJob, AnalysisResult, ClassificationResult nodes)
7. **Notify** — Push results to frontend via WebSocket

---

## Architecture

### Docker Mode (10 Containers)

| Service | Image | Port | Role |
|---------|-------|------|------|
| **Nginx** | `nginx:alpine` | `80` | Reverse proxy — routes `/api/*` to backend, everything else to frontend |
| **Frontend** | Node 20 + Vite → Nginx (multi-stage) | `3000` | React + TypeScript + Vite web interface |
| **Backend** | Python 3.11 | `8000` | FastAPI REST API + WebSocket server |
| **Celery Worker** | Python 3.11 | — | Async inference worker (GPU-optional) |
| **Redis** | `redis:7-alpine` | `6379` | Message broker + result backend + response cache |
| **FalkorDB** | `falkordb/falkordb` | `6381` | Graph database — patients, scans, results, doctors, tags, audit logs |
| **MinIO** | `minio/minio` | `9000` / `9001` | S3-compatible object storage for MRI files |
| **Jaeger** | `jaegertracing/all-in-one` | `16686` | Distributed tracing — visualize request flow across services |
| **Prometheus** | `prom/prometheus` | `9090` | Metrics collection — scrapes `/metrics` from backend |
| **Grafana** | `grafana/grafana` | `3001` | Monitoring dashboards — request rate, latency, errors |

### Local Mode (3 Processes)

| Process | Port | Role |
|---------|------|------|
| **Backend** (uvicorn) | `8000` | FastAPI + sync inference + local file storage |
| **Frontend** (Vite) | `3000` | React dev server with hot reload |
| **FalkorDB** (Docker) | `6381` | Graph database (1 container) |

### Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Backend** | FastAPI + Pydantic | 0.104.1 / 2.5.0 |
| **ML Framework** | PyTorch + TorchVision | 2.2.2 / 0.17.2 |
| **ML Optimization** | ONNX Runtime (INT8 quantization) | ≥1.16.3 |
| **Task Queue** | Celery + Redis | 5.3.4 (Docker mode) |
| **Graph Database** | FalkorDB | 1.0.9 |
| **Object Storage** | MinIO (Docker) / Local FS | — |
| **Frontend** | React + TypeScript + Vite | 18.2 / 5.3 / 5.4 |
| **HTTP Client** | Axios | 1.6.2 |
| **Auth** | OAuth2 + JWT (bcrypt) | HS256 |
| **Rate Limiting** | slowapi | 0.1.9 |
| **Tracing** | OpenTelemetry + Jaeger | 1.22 (Docker mode) |
| **Metrics** | Prometheus + Grafana | Docker mode |
| **XAI** | Grad-CAM heatmaps | — |
| **Medical Formats** | nibabel (NIfTI) + pydicom (DICOM) | 5.1 / 2.4 |

---

## Project Structure

```
brain-tumor-ai-framework/
│
├── README.md                          # This file
├── docker-compose.yml                 # Orchestrates all 10 Docker services
├── Dockerfile.backend                 # Python 3.11 backend image
├── Dockerfile.frontend                # Multi-stage Vite → Nginx image
├── Dockerfile.worker                  # Celery worker image
├── setup_local.ps1                    # PowerShell setup helper script
│
├── backend/
│   ├── requirements.txt               # Python dependencies (42 packages)
│   ├── .env.local                     # Environment template for local dev
│   ├── seed_falkordb.py               # Database seeder script (7 steps)
│   └── app/
│       ├── main.py                    # FastAPI entry + rate limiting + Prometheus + Jaeger
│       ├── train.py                   # CLI training script (--download, --store-graph, etc.)
│       ├── auth.py                    # OAuth2 + JWT authentication (register/login/refresh)
│       ├── api/
│       │   └── routes.py              # 30+ REST + WebSocket endpoints
│       ├── config/
│       │   └── settings.py            # Pydantic settings (env-based configuration)
│       ├── dataset/
│       │   ├── models.py              # MODEL_REGISTRY — 4 neural network architectures
│       │   ├── ensemble.py            # Ensemble inference + TTA + weighted logit fusion
│       │   ├── trainer.py             # Training loop (per-model)
│       │   ├── downloader.py          # Kaggle dataset download + merge 3 sources
│       │   ├── gradcam.py             # Grad-CAM explainability heatmaps
│       │   └── onnx_engine.py         # ONNX export, INT8 quantization, runtime inference
│       ├── models/
│       │   └── database.py            # JobStatus enum
│       ├── services/
│       │   ├── graph_db.py            # FalkorDB + InMemoryGraphDB — 15 entities, M:N rels
│       │   └── storage.py             # MinIO/S3/Local storage abstraction
│       ├── workers/
│       │   └── sync_inference.py      # Synchronous 7-step inference pipeline
│       ├── schemas/
│       │   └── __init__.py            # 23 Pydantic models (validation + serialization)
│       └── utils/
│           └── validators.py          # File validators (DICOM, NIfTI, standard images)
│
├── frontend/
│   ├── package.json                   # React 18 + TypeScript 5.3 + Vite 5.4
│   ├── vite.config.ts                 # Vite config (dev server port 3000, proxy to :8000)
│   ├── tsconfig.json                  # TypeScript configuration
│   ├── index.html                     # Vite HTML entry point
│   └── src/
│       ├── index.tsx                  # React DOM render
│       ├── App.tsx                    # Root component (state machine: 7 phases)
│       ├── components/
│       │   ├── Dashboard.tsx          # Main layout — orchestrates all panels
│       │   ├── CommandEntry.tsx       # Landing page — model info + features
│       │   ├── SmartIngestion.tsx     # File upload with validation preview
│       │   ├── PipelineView.tsx       # Processing pipeline visualization
│       │   ├── DiagnosticWorkspace.tsx# Results — 4-model cards + recommendations
│       │   ├── PipelineLog.tsx        # Real-time processing log
│       │   └── AIChatAssistant.tsx    # AI assistant chat panel
│       ├── services/
│       │   └── api.ts                 # Axios API client (base: /api/v1)
│       ├── types/
│       │   └── api.ts                 # TypeScript interfaces
│       └── styles/
│           └── crw.css                # Complete design system (~1500 lines)
│
├── models/                            # Pre-trained weights (4 files)
│   ├── brain_tumor_2d.pth             # Custom CNN (~2 MB)
│   ├── brain_tumor_resnet50.pth       # ResNet-50 (~98 MB)
│   ├── brain_tumor_efficientnet.pth   # EfficientNet-B0 (~21 MB)
│   └── brain_tumor_densenet.pth       # DenseNet-121 (~32 MB)
│
├── data/
│   └── combined/                      # Training dataset (download separately)
│       ├── train/
│       │   ├── glioma/
│       │   ├── meningioma/
│       │   ├── pituitary/
│       │   └── no_tumor/
│       └── val/
│           ├── glioma/
│           ├── meningioma/
│           ├── pituitary/
│           └── no_tumor/
│
└── infra/                             # Docker infrastructure configs
    ├── nginx.conf                     # Reverse proxy config
    ├── prometheus.yml                 # Prometheus scrape config
    └── grafana/                       # Grafana provisioning + dashboards
```

---

## Advanced Features

### 1. Explainable AI — Grad-CAM Heatmaps

The system generates visual heatmaps showing **which regions of the MRI** each model focused on when making its prediction. Red = high attention, blue = low attention. Each model uses a different target layer:

| Model | Grad-CAM Target Layer |
|-------|-----------------------|
| Custom CNN | Last conv layer |
| ResNet-50 | `layer4[-1]` |
| EfficientNet-B0 | `features[-1]` |
| DenseNet-121 | `features.denseblock4` |

```bash
curl -X POST http://localhost:8000/api/v1/explainability/gradcam/{job_id}?model_name=resnet50
```

### 2. ONNX Runtime Optimization

Export PyTorch models to ONNX format for **3-5x faster inference** with optional INT8 quantization:

```bash
# Export a model
curl -X POST http://localhost:8000/api/v1/onnx/export/resnet50

# Run inference on the optimized model
curl -X POST http://localhost:8000/api/v1/onnx/predict -F "file=@brain_scan.png"
```

### 3. Distributed Tracing (Jaeger — Docker mode)

Every API request is traced end-to-end through FastAPI → Redis → FalkorDB → MinIO. Open **http://localhost:16686** to visualize request timelines, bottlenecks, and service dependencies.

### 4. Metrics & Monitoring (Prometheus + Grafana — Docker mode)

The backend exposes a `/metrics` endpoint scraped by Prometheus every 15 seconds. A pre-built Grafana dashboard at **http://localhost:3001** shows request rate, P95 latency, 5xx error rate, active requests, and inference latency.

### 5. OAuth2 + JWT Authentication

Full authentication flow with bcrypt password hashing, access tokens (30 min), and refresh tokens (7 days). Role-based access control (admin/doctor/researcher).

```bash
# Register
curl -X POST "http://localhost:8000/api/v1/auth/register?username=dr_smith&email=dr@hospital.com&password=secret&role=doctor"

# Login → returns JWT
curl -X POST "http://localhost:8000/api/v1/auth/login?username=dr_smith&password=secret"
```

### 6. Rate Limiting

IP-based rate limiting via `slowapi`:
- Default: 200 requests/minute per IP
- `/analyze`: 10 requests/minute (heavy inference workload)
- `/onnx/export`: 2 requests/minute
- `/onnx/predict`: 20 requests/minute

### 7. CI/CD Pipeline (GitHub Actions)

Automated CI on every push/PR:
- Python linting (Ruff) + type checking (mypy)
- TypeScript type checking + Vite build
- Docker image builds for all 3 services (with GitHub Actions cache)

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

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/register` | Register a new user (doctor/admin/researcher). |
| `POST` | `/auth/login` | Login with credentials → returns JWT access + refresh tokens. |
| `POST` | `/auth/refresh` | Refresh an expired access token. |

### Doctor & M:N Relationships

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/doctors` | Create a new doctor record. |
| `GET` | `/doctors/{doctor_id}` | Get doctor details. |
| `POST` | `/doctors/{doctor_id}/assign-patient/{mrn}` | Assign doctor to patient (M:N). |
| `POST` | `/doctors/{doctor_id}/assign-job/{job_id}` | Assign doctor to job (M:N). |
| `GET` | `/doctors/{doctor_id}/patients` | Get all patients for a doctor. |
| `GET` | `/patients/{mrn}/doctors` | Get all doctors for a patient. |

### Tags (M:N)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/scans/{scan_id}/tags/{tag_name}` | Tag a scan (M:N Scan↔Tag). |
| `GET` | `/tags/{tag_name}/scans` | Get all scans with a given tag. |

### Explainability & ONNX

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/explainability/gradcam/{job_id}` | Generate Grad-CAM heatmap for a completed job. |
| `POST` | `/onnx/export/{model_name}` | Export PyTorch model to ONNX with quantization. |
| `GET` | `/onnx/models` | List available ONNX models. |
| `POST` | `/onnx/predict` | Run ensemble inference via ONNX Runtime. |

### Audit & Versioning

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/audit-logs` | Retrieve system audit logs. |
| `POST` | `/models/versions` | Register a new model version. |
| `GET` | `/models/versions` | List all model versions (with SUPERSEDES chain). |

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
curl http://localhost:8000/api/v1/inference/abc-123/results
```

---

## Environment Variables

### Local Mode (backend/.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_BACKEND` | `local` | `local` or `minio` |
| `LOCAL_STORAGE_DIR` | `./data/uploads` | Where uploaded files are saved |
| `USE_SYNC_INFERENCE` | `true` | Synchronous inference (no Celery) |
| `FALKORDB_HOST` | `localhost` | FalkorDB hostname |
| `FALKORDB_PORT` | `6381` | FalkorDB port |
| `SKIP_FALKORDB` | `false` | Skip FalkorDB (use in-memory fallback) |
| `DEBUG` | `true` | Enable debug logging |
| `API_PORT` | `8000` | Backend port |
| `CORS_ORIGINS` | `["http://localhost:3000", ...]` | Allowed CORS origins |

### Docker Mode (docker-compose.yml)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `MINIO_ENDPOINT` | `minio:9000` | MinIO server address |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `FALKORDB_HOST` | `falkordb` | FalkorDB server hostname |
| `FALKORDB_PORT` | `6379` | FalkorDB port |
| `STORAGE_BACKEND` | `minio` | Storage backend |
| `ENSEMBLE_MODELS` | `custom_cnn,resnet50,efficientnet,densenet` | Active ensemble models |
| `JAEGER_HOST` | `jaeger` | Jaeger agent hostname |
| `TRACING_ENABLED` | `true` | Enable OpenTelemetry tracing |
| `RATE_LIMIT_DEFAULT` | `200/minute` | Default rate limit per IP |
| `RATE_LIMIT_ANALYZE` | `10/minute` | Rate limit for /analyze |
| `JWT_SECRET_KEY` | (auto-generated) | Secret for JWT signing |

### GPU Support (optional — Docker mode)

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

### Local Mode Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'app'` | Make sure you're running from the `backend/` directory |
| `FileNotFoundError: .env` | Run `copy .env.local .env` inside `backend/` |
| Models not loading / file not found | Ensure `.venv/` is in the **project root**, not inside `backend/` |
| `SKIP_FALKORDB` warning on startup | FalkorDB container isn't running. Start it or set `SKIP_FALKORDB=true` |
| Frontend shows network error | Check that backend is running on port 8000 |
| `npm run dev` fails | Run `npm install` first in the `frontend/` directory |
| Port 8000 already in use | Kill the process using port 8000 or change `API_PORT` in `.env` |
| Python version error | Ensure Python 3.11+ is installed (`python --version`) |

### Docker Mode Issues

| Problem | Solution |
|---------|----------|
| `docker compose` not found | Install Docker Desktop (v4.0+) which includes Compose v2 |
| Port 80 already in use | Stop other web servers or change the port in `docker-compose.yml` |
| Build fails on first run | Run `docker compose build --no-cache` to retry from scratch |
| MinIO connection refused | Wait for the health check — MinIO takes ~10s to start |
| Frontend shows blank page | Check that Nginx container is running: `docker compose ps` |
| Out of memory during inference | Reduce `inference_batch_size` in `backend/app/config/settings.py` |

### Useful Commands

```powershell
# ===================== LOCAL MODE =====================

# Activate virtual environment (Windows)
.\.venv\Scripts\Activate.ps1

# Check backend health
curl http://localhost:8000/api/v1/health

# Check ensemble status
curl http://localhost:8000/api/v1/ensemble/status

# Seed the database
cd backend; python seed_falkordb.py

# ===================== DOCKER MODE ====================

# View all running containers
docker compose ps

# View backend logs
docker compose logs -f backend

# Restart a single service
docker compose restart backend

# Rebuild without cache
docker compose build --no-cache

# Remove everything (containers + volumes + images)
docker compose down -v --rmi all

# ===================== FALKORDB ========================

# Start FalkorDB
docker start falkordb

# Stop FalkorDB
docker stop falkordb

# View FalkorDB logs
docker logs falkordb

# Query node count (requires redis-cli or falkordb-cli)
docker exec falkordb redis-cli GRAPH.QUERY brain_tumor "MATCH (n) RETURN count(n)"
```

---

## License

MIT
