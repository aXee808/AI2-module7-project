# MLOps Tutorial Complet - Prédiction de Stocks

Un tutoriel pratique et complet pour apprendre le MLOps avec 3 approches différentes.

## Vue d'ensemble

Ce tutoriel vous guide à travers la mise en place d'un pipeline MLOps complet pour une application de prédiction de stocks. Vous apprendrez 3 approches différentes :

| Approche | Outils | Complexité | Use Case |
|----------|--------|------------|----------|
| **Approche 1** | GitHub Actions + ArgoCD + K8s | Moyenne | Production enterprise |
| **Approche 2** | Kubeflow | Élevée | ML à grande échelle |
| **Approche 3** | MLflow | Faible | Démarrage rapide |

## Prérequis

- Python 3.9+
- Docker Desktop
- Git
- Un compte GitHub
- Un compte Docker Hub (gratuit)

## Structure du Projet

```
mlops-stock-prediction/
├── README.md
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── app.py              # API Flask
│   ├── data_processing.py  # Traitement données
│   ├── feature_engineering.py
│   ├── train.py            # Entraînement
│   └── predict.py          # Prédictions
│
├── models/
│   └── .gitkeep
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── kubernetes/             # Manifests K8s
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ...
│
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
│
├── kubeflow/              # Pipelines Kubeflow
│   └── pipeline.py
│
├── mlflow/                # Config MLflow
│   └── MLproject
│
└── scripts/
    ├── setup-mac.sh
    ├── setup-linux.sh
    └── setup-windows.ps1
```

## Démarrage Rapide

```bash
# 1. Cloner le projet
git clone https://github.com/votre-username/mlops-stock-prediction.git
cd mlops-stock-prediction

# 2. Créer l'environnement virtuel
python -m venv venv

# 3. Activer l'environnement
# Mac/Linux:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Entraîner le modèle
python src/train.py

# 6. Lancer l'API
python src/app.py

# 7. Tester
curl http://localhost:5000/health
```

## Documentation

1. [Théorie MLOps](docs/01-theorie-mlops.md)
2. [Approche 1 : GitHub Actions + ArgoCD](docs/02-github-actions-argocd.md)
3. [Approche 2 : Kubeflow](docs/03-kubeflow.md)
4. [Approche 3 : MLflow](docs/04-mlflow.md)
5. [Installation Multi-OS](docs/05-installation.md)

## API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Info de l'API |
| `/health` | GET | Health check |
| `/predict` | POST | Prédiction |
| `/model/info` | GET | Info du modèle |
| `/metrics` | GET | Métriques Prometheus |

Pour démarrer rapidement

  cd /Users/andric/Documents/dev/claude/mlops-complet

  # Créer l'environnement virtuel
  python -m venv venv
  source venv/bin/activate  # Mac/Linux
  # .\venv\Scripts\activate  # Windows

  # Installer les dépendances
  pip install -r requirements.txt

  # Entraîner le modèle
  python src/train.py

  # Lancer l'API Flask
  python src/app.py

  # Tester
  curl http://localhost:5000/health
  curl http://localhost:5000/predict/demo

  Avec Docker Compose (tout-en-un)

  docker-compose up -d
  # API: http://localhost:5000
  # MLflow: http://localhost:5001
  # Prometheus: http://localhost:9090
  # Grafana: http://localhost:3000
