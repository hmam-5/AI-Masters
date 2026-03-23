<# 
.SYNOPSIS
    One-click setup script for Brain Tumor AI Framework (local mode)
.DESCRIPTION
    This script sets up the project for local development WITHOUT Docker.
    It installs Python dependencies, configures local mode, and starts the backend.
.NOTES
    Run this script from the project root folder.
    Prerequisites: Python 3.11+, Node.js 18+
#>

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Brain Tumor AI Framework - Local Setup    " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right folder
if (-not (Test-Path "backend" -PathType Container)) {
    Write-Host "ERROR: Please run this script from the project root folder!" -ForegroundColor Red
    Write-Host "Expected folder structure: backend/, frontend/, models/, data/" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python is not installed!" -ForegroundColor Red
    Write-Host "Please install Python 3.11+ from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}
Write-Host "      Found: $pythonVersion" -ForegroundColor Green

Write-Host ""
Write-Host "[2/6] Creating Python virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "backend\.venv")) {
    python -m venv backend\.venv
    Write-Host "      Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "      Virtual environment already exists." -ForegroundColor Green
}

Write-Host ""
Write-Host "[3/6] Installing Python dependencies..." -ForegroundColor Yellow
& backend\.venv\Scripts\python.exe -m pip install --upgrade pip -q
& backend\.venv\Scripts\pip.exe install -r backend\requirements.txt -q
Write-Host "      Dependencies installed." -ForegroundColor Green

Write-Host ""
Write-Host "[4/6] Configuring local mode..." -ForegroundColor Yellow
if (-not (Test-Path "backend\.env")) {
    Copy-Item "backend\.env.local" "backend\.env"
    Write-Host "      Created .env from .env.local" -ForegroundColor Green
} else {
    Write-Host "      .env already exists (not overwritten)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[5/6] Creating upload directory..." -ForegroundColor Yellow
if (-not (Test-Path "data\uploads")) {
    New-Item -ItemType Directory -Path "data\uploads" -Force | Out-Null
    Write-Host "      Created data/uploads/" -ForegroundColor Green
} else {
    Write-Host "      data/uploads/ already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "[6/6] Checking Node.js installation..." -ForegroundColor Yellow
$nodeVersion = node --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Node.js is not installed!" -ForegroundColor Yellow
    Write-Host "Frontend won't work. Install from https://nodejs.org/" -ForegroundColor Yellow
} else {
    Write-Host "      Found: Node.js $nodeVersion" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    Push-Location frontend
    npm install -q 2>&1 | Out-Null
    Pop-Location
    Write-Host "      Frontend dependencies installed." -ForegroundColor Green
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETE!                           " -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Start the backend (Terminal 1):" -ForegroundColor White
Write-Host "     cd backend" -ForegroundColor Yellow
Write-Host "     .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "     uvicorn app.main:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor Yellow
Write-Host ""
Write-Host "  2. Start the frontend (Terminal 2):" -ForegroundColor White
Write-Host "     cd frontend" -ForegroundColor Yellow
Write-Host "     npm run dev" -ForegroundColor Yellow
Write-Host ""
Write-Host "  3. Open in browser:" -ForegroundColor White
Write-Host "     http://localhost:5173 (frontend)" -ForegroundColor Yellow
Write-Host "     http://localhost:8000/docs (API docs)" -ForegroundColor Yellow
Write-Host ""
Write-Host "(Optional) Start FalkorDB for full database support:" -ForegroundColor Cyan
Write-Host "     docker run -d -p 6381:6379 --name falkordb falkordb/falkordb:latest" -ForegroundColor Yellow
Write-Host ""
