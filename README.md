# AI Masters

AI-powered brain tumor detection and classification from MRI images (PNG, JPG, JPEG).

- **2D CNN** for tumor grade classification (Grade I–IV)
- **Segmentation analysis** for enhancing tumor, edema, and necrotic core
- **AI-generated explanations** with findings and clinical recommendations
- **Async processing** with Celery workers
- **Modern dark UI** with animated results dashboard

## Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

## How to Run

```bash
cd brain-tumor-ai-framework
docker compose up --build
```

Wait for all services to start (first build downloads ~2GB of dependencies). Once ready:

| Service         | URL                        |
|-----------------|----------------------------|
| **App (Main)**  | http://localhost            |
| Frontend        | http://localhost:3000       |
| Backend API     | http://localhost:8000       |
| API Docs        | http://localhost:8000/docs  |
| MinIO Console   | http://localhost:9001       |

To stop everything:

```bash
docker compose down
```

## Configuration

Copy and edit the environment file to customize settings:

```bash
cp .env.example .env
```

Key settings in `.env`:

| Variable             | Default                        | Description          |
|----------------------|--------------------------------|----------------------|
| POSTGRES_PASSWORD    | secure_password_change_in_prod | Database password    |
| MINIO_ACCESS_KEY     | minioadmin                     | MinIO username       |
| MINIO_SECRET_KEY     | minioadmin                     | MinIO password       |
| DEBUG                | false                          | Enable debug mode    |

## Project Structure

```
brain-tumor-ai-framework/
├── backend/                    # FastAPI + Python
│   ├── app/
│   │   ├── main.py             # App entry point
│   │   ├── api/routes.py       # API endpoints (/analyze, /results)
│   │   ├── config/settings.py  # Configuration
│   │   ├── models/             # Database models (SQLAlchemy)
│   │   ├── schemas/            # Request/response schemas (Pydantic)
│   │   ├── services/storage.py # S3/MinIO file storage
│   │   ├── utils/validators.py # Image validation (PNG/JPG)
│   │   └── workers/            # Celery async tasks (2D CNN inference)
│   └── requirements.txt
├── frontend/                   # React + TypeScript
│   ├── src/
│   │   ├── components/         # Dashboard, UploadForm
│   │   ├── services/api.ts     # API client
│   │   └── types/api.ts        # TypeScript types
│   └── package.json
├── infra/nginx.conf            # Nginx reverse proxy config
├── Dockerfile.backend
├── Dockerfile.frontend
├── Dockerfile.worker
├── docker-compose.yml
└── .env.example
```

## Services

The app runs 7 Docker containers:

| Container   | Purpose                                  |
|-------------|------------------------------------------|
| **nginx**   | Reverse proxy — main entry (port 80)     |
| **backend** | FastAPI server (port 8000)               |
| **frontend**| React app served via Nginx (port 3000)   |
| **celery**  | Background worker for AI inference       |
| **postgres**| PostgreSQL database (port 5432)          |
| **redis**   | Message broker + cache (port 6379)       |
| **minio**   | S3-compatible file storage (port 9000)   |

## API Quick Reference

### Image Analysis
```
POST /api/v1/analyze                          # Upload image for AI analysis
GET  /api/v1/inference/{job_id}/results       # Get full analysis results
GET  /api/v1/inference/{job_id}               # Check job status
```

### Patients
```
POST /api/v1/patients              # Create patient
GET  /api/v1/patients/{mrn}        # Get patient by MRN
```

### Health
```
GET  /health                       # Health check
```

## Troubleshooting

**Docker Desktop not running:**
Start Docker Desktop and wait for it to fully initialize before running `docker compose up`.

**Port already in use:**
Stop any services using ports 80, 3000, 5432, 6379, 8000, 9000, or 9001.

**Check container logs:**
```bash
docker compose logs backend        # Backend logs
docker compose logs celery-worker  # Worker logs
docker compose logs postgres       # Database logs
```

**Rebuild from scratch:**
```bash
docker compose down -v             # Stop and remove volumes
docker compose up --build          # Rebuild everything
```
