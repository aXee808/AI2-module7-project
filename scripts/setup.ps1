# ============================================
# Script d'installation MLOps - Windows
# PowerShell 5.1+
# ============================================

# Configuration
$ErrorActionPreference = "Stop"

# Couleurs
function Write-Header($message) {
    Write-Host "`n╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Blue
    Write-Host "║  $message" -ForegroundColor Blue
    Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Blue
}

function Write-Success($message) {
    Write-Host "✅ $message" -ForegroundColor Green
}

function Write-Warning($message) {
    Write-Host "⚠️  $message" -ForegroundColor Yellow
}

function Write-Error($message) {
    Write-Host "❌ $message" -ForegroundColor Red
}

function Write-Info($message) {
    Write-Host "ℹ️  $message" -ForegroundColor Cyan
}

# Vérifier les prérequis
function Test-Prerequisites {
    Write-Header "Vérification des prérequis"

    $missing = @()

    # Python
    try {
        $pythonVersion = python --version 2>&1
        Write-Success "Python: $pythonVersion"
    }
    catch {
        $missing += "Python"
        Write-Error "Python non trouvé"
    }

    # pip
    try {
        $null = pip --version 2>&1
        Write-Success "pip installé"
    }
    catch {
        $missing += "pip"
        Write-Error "pip non trouvé"
    }

    # Git
    try {
        $null = git --version 2>&1
        Write-Success "Git installé"
    }
    catch {
        $missing += "Git"
        Write-Error "Git non trouvé"
    }

    # Docker
    try {
        $null = docker --version 2>&1
        Write-Success "Docker installé"
    }
    catch {
        Write-Warning "Docker non trouvé (optionnel)"
    }

    # kubectl
    try {
        $null = kubectl version --client 2>&1
        Write-Success "kubectl installé"
    }
    catch {
        Write-Warning "kubectl non trouvé (nécessaire pour Kubernetes)"
    }

    if ($missing.Count -gt 0) {
        Write-Error "Prérequis manquants: $($missing -join ', ')"
        Write-Info "Installer les dépendances manquantes et relancer le script"
        exit 1
    }
}

# Créer l'environnement virtuel
function New-VirtualEnv {
    Write-Header "Configuration de l'environnement virtuel Python"

    $venvDir = "venv"

    if (Test-Path $venvDir) {
        Write-Warning "L'environnement virtuel existe déjà"
        $response = Read-Host "Voulez-vous le recréer? (y/N)"
        if ($response -eq 'y' -or $response -eq 'Y') {
            Remove-Item -Recurse -Force $venvDir
        }
        else {
            Write-Info "Utilisation de l'environnement existant"
            return
        }
    }

    Write-Info "Création de l'environnement virtuel..."
    python -m venv $venvDir

    Write-Info "Activation de l'environnement..."
    & "$venvDir\Scripts\Activate.ps1"

    Write-Info "Mise à jour de pip..."
    python -m pip install --upgrade pip

    Write-Success "Environnement virtuel créé: $venvDir"
}

# Installer les dépendances
function Install-Dependencies {
    Write-Header "Installation des dépendances Python"

    # Activer le venv si pas déjà fait
    if (-not $env:VIRTUAL_ENV) {
        & "venv\Scripts\Activate.ps1"
    }

    if (Test-Path "requirements.txt") {
        Write-Info "Installation depuis requirements.txt..."
        pip install -r requirements.txt
        Write-Success "Dépendances installées"
    }
    else {
        Write-Error "requirements.txt non trouvé"
        exit 1
    }
}

# Créer la structure des répertoires
function New-DirectoryStructure {
    Write-Header "Création de la structure des répertoires"

    $directories = @(
        "data\raw",
        "data\processed",
        "models",
        "logs",
        "monitoring\grafana\provisioning\dashboards",
        "monitoring\grafana\provisioning\datasources",
        "notebooks",
        "tests"
    )

    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Success "Créé: $dir"
        }
    }
}

# Créer les fichiers de configuration
function New-ConfigFiles {
    Write-Header "Création des fichiers de configuration"

    # Prometheus config
    $prometheusConfig = @"
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'stock-prediction-api'
    static_configs:
      - targets: ['api:5000']
    metrics_path: '/metrics'
"@
    $prometheusConfig | Out-File -FilePath "monitoring\prometheus.yml" -Encoding utf8
    Write-Success "Créé: monitoring\prometheus.yml"

    # .gitignore
    $gitignore = @"
# Python
__pycache__/
*.py[cod]
*`$py.class
*.so
.Python
venv/
ENV/
.venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Data
data/raw/*.csv
data/processed/*.csv

# Models
models/*.joblib
models/*.pkl

# Logs
logs/
*.log

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Coverage
.coverage
htmlcov/

# MLflow
mlruns/
mlflow.db
"@
    $gitignore | Out-File -FilePath ".gitignore" -Encoding utf8
    Write-Success "Créé: .gitignore"
}

# Entraîner le modèle
function Start-Training {
    Write-Header "Entraînement du modèle"

    # Activer le venv si pas déjà fait
    if (-not $env:VIRTUAL_ENV) {
        & "venv\Scripts\Activate.ps1"
    }

    Write-Info "Génération des données et entraînement..."
    python src\train.py

    if (Test-Path "models\model.joblib") {
        Write-Success "Modèle entraîné et sauvegardé"
    }
    else {
        Write-Error "Échec de l'entraînement"
        exit 1
    }
}

# Tester l'installation
function Test-Installation {
    Write-Header "Test de l'installation"

    # Activer le venv si pas déjà fait
    if (-not $env:VIRTUAL_ENV) {
        & "venv\Scripts\Activate.ps1"
    }

    Write-Info "Exécution des tests..."

    try {
        python -m pytest tests\ -v --tb=short
        Write-Success "Tests passés"
    }
    catch {
        Write-Warning "Certains tests ont échoué"
    }
}

# Démarrer l'API
function Start-API {
    Write-Header "Démarrage de l'API"

    # Activer le venv si pas déjà fait
    if (-not $env:VIRTUAL_ENV) {
        & "venv\Scripts\Activate.ps1"
    }

    Write-Info "Démarrage du serveur Flask..."
    Write-Info "API disponible sur: http://localhost:5000"
    Write-Info "Health check: http://localhost:5000/health"
    Write-Info "Démo: http://localhost:5000/predict/demo"
    Write-Info ""
    Write-Info "Appuyez sur Ctrl+C pour arrêter"

    python src\app.py
}

# Afficher les instructions finales
function Show-NextSteps {
    Write-Header "Installation terminée!"

    Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║              Installation terminée avec succès!               ║
╚══════════════════════════════════════════════════════════════╝

Prochaines étapes:

1. Activer l'environnement virtuel:
   .\venv\Scripts\Activate.ps1

2. Lancer l'API en local:
   python src\app.py

3. Lancer avec Docker:
   docker-compose up -d

4. Accéder aux services:
   - API:        http://localhost:5000
   - MLflow:     http://localhost:5001
   - Prometheus: http://localhost:9090
   - Grafana:    http://localhost:3000 (admin/admin)

5. Tester une prédiction:
   Invoke-RestMethod http://localhost:5000/predict/demo

"@ -ForegroundColor Green
}

# Menu principal
function Main {
    param (
        [string]$Action = "all"
    )

    switch ($Action) {
        "all" {
            Write-Header "Installation MLOps Stock Prediction"
            Test-Prerequisites
            New-DirectoryStructure
            New-ConfigFiles
            New-VirtualEnv
            Install-Dependencies
            Start-Training
            Test-Installation
            Show-NextSteps
        }
        "deps" {
            New-VirtualEnv
            Install-Dependencies
        }
        "train" {
            Start-Training
        }
        "test" {
            Test-Installation
        }
        "start" {
            Start-API
        }
        default {
            Write-Host "Usage: .\setup.ps1 [all|deps|train|test|start]"
        }
    }
}

# Exécuter
$action = if ($args.Count -gt 0) { $args[0] } else { "all" }
Main -Action $action
