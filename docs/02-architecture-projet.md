# Architecture du Projet

Ce document décrit l'architecture complète du projet MLOps de prédiction de stocks.

---

## Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MLOPS PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  Données │───▶│ Features │───▶│  Train   │───▶│  Modèle  │             │
│  │  Brutes  │    │Engineering│    │          │    │ .joblib  │             │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│                                                         │                   │
│                                                         ▼                   │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                         API Flask                                 │      │
│  │  /health  /predict  /predict/demo  /predict/batch  /metrics      │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Structure des Fichiers

```
mlops-complet/
│
├── 📁 src/                          # Code source Python
│   ├── __init__.py                  # Package Python
│   ├── data_processing.py           # Génération et chargement des données
│   ├── feature_engineering.py       # Création des features techniques
│   ├── train.py                     # Entraînement du modèle
│   └── app.py                       # API Flask
│
├── 📁 models/                       # Modèles entraînés (généré)
│   ├── model.joblib                 # Modèle sérialisé
│   ├── scaler.joblib                # Scaler pour normalisation
│   └── metadata.json                # Métadonnées du modèle
│
├── 📁 data/                         # Données (généré)
│   └── raw/
│       └── stock_data.csv           # Données de stock
│
├── 📁 docs/                         # Documentation
│   ├── 01-theorie-mlops.md          # Théorie MLOps
│   ├── 02-architecture-projet.md    # Ce fichier
│   ├── 03-approche-github-actions-argocd.md
│   ├── 04-approche-kubeflow-mlflow.md
│   └── 05-guide-demarrage-rapide.md
│
├── 📁 kubernetes/                   # Manifests Kubernetes
│   ├── namespace.yaml               # Namespace dédié
│   ├── deployment.yaml              # Déploiement de l'API
│   ├── service.yaml                 # Service, ConfigMap, HPA
│   └── argocd-application.yaml      # Configuration ArgoCD
│
├── 📁 kubeflow/                     # Pipeline Kubeflow
│   └── pipeline.py                  # Définition du pipeline KFP
│
├── 📁 mlflow/                       # Scripts MLflow
│   └── train_with_mlflow.py         # Entraînement avec tracking
│
├── 📁 scripts/                      # Scripts d'installation
│   ├── setup.sh                     # Mac/Linux
│   └── setup.ps1                    # Windows PowerShell
│
├── 📁 .github/workflows/            # CI/CD GitHub Actions
│   └── ml-pipeline.yml              # Pipeline complet
│
├── 📁 monitoring/                   # Configuration monitoring
│   ├── prometheus.yml               # Config Prometheus
│   └── grafana/
│       └── dashboards/              # Dashboards Grafana
│
├── Dockerfile                       # Image Docker multi-stage
├── docker-compose.yml               # Stack complète
├── requirements.txt                 # Dépendances Python
├── .gitignore                       # Fichiers ignorés
└── README.md                        # Documentation principale
```

---

## Composants Détaillés

### 1. Module de Données (`src/data_processing.py`)

```python
# Fonctions principales
generate_synthetic_stock_data(days, ticker, seed)  # Génère des données OHLCV
load_or_generate_data(path, days, seed)            # Charge ou génère les données
split_data(df, train_ratio, val_ratio)             # Split temporel
```

**Données générées :**
| Colonne | Description |
|---------|-------------|
| Date | Date de trading |
| Ticker | Symbole de l'action |
| Open | Prix d'ouverture |
| High | Plus haut du jour |
| Low | Plus bas du jour |
| Close | Prix de clôture |
| Volume | Volume échangé |

### 2. Module de Features (`src/feature_engineering.py`)

```python
# Indicateurs techniques calculés
calculate_sma(df, windows)          # Moyennes mobiles simples
calculate_ema(df, spans)            # Moyennes mobiles exponentielles
calculate_rsi(df, period)           # Relative Strength Index
calculate_macd(df)                  # MACD et Signal
calculate_bollinger_bands(df)       # Bandes de Bollinger
create_features(df)                 # Pipeline complet
prepare_training_data(df)           # Préparation X, y
```

**Features créées (22 total) :**

| Catégorie | Features |
|-----------|----------|
| Moyennes Mobiles | SMA_5, SMA_10, SMA_20, SMA_50, EMA_5, EMA_10, EMA_20 |
| Momentum | RSI, MACD, MACD_Signal, MACD_Hist |
| Volatilité | BB_Width, BB_Position, Volatility_10d, Volatility_20d |
| Returns | Return_1d, Return_5d, Return_10d |
| Volume | Volume_Ratio |
| Prix | High_Low_Range, Close_Position, Gap |

### 3. Module d'Entraînement (`src/train.py`)

```python
class StockPredictor:
    def __init__(model_type, params)    # Initialisation
    def train(X, y)                     # Entraînement
    def predict(X)                      # Prédiction
    def predict_proba(X)                # Probabilités
    def evaluate(X, y)                  # Métriques
    def save(path)                      # Sauvegarde
    def load(path)                      # Chargement

# Modèles supportés
- random_forest (défaut)
- gradient_boosting
- logistic
```

**Métriques calculées :**
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

### 4. API Flask (`src/app.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                        API ENDPOINTS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GET  /              → Page d'accueil avec documentation        │
│  GET  /health        → Health check (pour K8s probes)           │
│  GET  /model/info    → Informations sur le modèle chargé        │
│  POST /predict       → Prédiction avec features fournies        │
│  GET  /predict/demo  → Prédiction de démonstration              │
│  POST /predict/batch → Prédictions en lot                       │
│  GET  /metrics       → Métriques Prometheus                     │
│  POST /reload        → Recharger le modèle                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Exemple de requête `/predict` :**

```json
// Request
POST /predict
{
  "features": [105.2, 104.8, 103.5, 100.2, 105.5, 105.0, 104.0,
               55.0, 0.8, 0.5, 0.3, 0.05, 0.6, 0.01, 0.03,
               0.05, 0.015, 0.018, 1.2, 0.02, 0.7, 0.002]
}

// Response
{
  "prediction": 1,
  "prediction_label": "UP",
  "probability": 0.73,
  "probabilities": {"DOWN": 0.27, "UP": 0.73},
  "model_version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## Pipeline de Données

```
┌──────────────┐
│ Données CSV  │
│ (OHLCV)      │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│              Feature Engineering                      │
│                                                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │   SMA   │ │   RSI   │ │  MACD   │ │Bollinger│   │
│  │ 5,10,20 │ │  (14)   │ │ 12,26,9 │ │  Bands  │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│                                                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │   EMA   │ │ Returns │ │Volatilité│ │ Volume  │   │
│  │ 5,10,20 │ │ 1,5,10d │ │ 10,20d  │ │  Ratio  │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│                   22 Features                         │
│  + Target (1 = UP, 0 = DOWN)                         │
└──────────────────────────┬───────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  Train   │    │Validation│    │   Test   │
    │   70%    │    │   15%    │    │   15%    │
    └──────────┘    └──────────┘    └──────────┘
```

---

## Architecture Docker

### Dockerfile Multi-Stage

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder
# Installation des dépendances
# Compilation des wheels

# Stage 2: Runtime
FROM python:3.11-slim as runtime
# Copie des dépendances compilées
# Configuration utilisateur non-root
# Healthcheck intégré
```

**Avantages :**
- Image finale légère (~200MB vs ~800MB)
- Pas d'outils de build en production
- Sécurité renforcée (non-root)

### Docker Compose Stack

```yaml
services:
  api:           # Port 5000 - API Flask
  mlflow:        # Port 5001 - Tracking server
  prometheus:    # Port 9090 - Métriques
  grafana:       # Port 3000 - Dashboards
```

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network                           │
│                                                             │
│   ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌───────┐ │
│   │   API   │◄───│Prometheus│───▶│  Grafana │    │MLflow │ │
│   │  :5000  │    │  :9090  │    │  :3000   │    │ :5001 │ │
│   └─────────┘    └─────────┘    └──────────┘    └───────┘ │
│        │                                             │      │
│        └─────────────────────────────────────────────┘      │
│                    Volume: mlflow_data                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture Kubernetes

```
┌─────────────────────────────────────────────────────────────────┐
│                    Namespace: stock-prediction                   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      Deployment                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │   │
│  │  │  Pod 1  │  │  Pod 2  │  │  Pod N  │  (HPA: 2-10)   │   │
│  │  │   API   │  │   API   │  │   API   │                 │   │
│  │  └─────────┘  └─────────┘  └─────────┘                 │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                       │
│  ┌──────────────────────┴──────────────────────────────────┐   │
│  │                    Service (ClusterIP)                   │   │
│  │                      Port 80 → 5000                      │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                       │
│  ┌──────────────────────┴──────────────────────────────────┐   │
│  │              Ingress (optionnel)                         │   │
│  │              api.example.com                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│  │  ConfigMap  │  │ServiceAccount│  │        HPA          │    │
│  │ LOG_LEVEL   │  │     RBAC    │  │ CPU: 70%, Mem: 80% │    │
│  └─────────────┘  └─────────────┘  └─────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Flux de Données Complet

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ GitHub  │────▶│ Actions │────▶│  Build  │────▶│  Push   │
│  Push   │     │   CI    │     │ Docker  │     │Registry │
└─────────┘     └─────────┘     └─────────┘     └────┬────┘
                                                      │
                                                      ▼
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ ArgoCD  │◀────│  Sync   │◀────│  K8s    │◀────│  Image  │
│   UI    │     │         │     │Manifests│     │  Ready  │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Kubernetes Cluster                         │
│                                                             │
│   Pods → Service → Ingress → Users                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration Requise

### Développement Local

| Composant | Minimum | Recommandé |
|-----------|---------|------------|
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Disque | 10 GB | 20 GB |
| Python | 3.9 | 3.11 |

### Production (par pod)

| Ressource | Request | Limit |
|-----------|---------|-------|
| CPU | 100m | 500m |
| Mémoire | 128Mi | 512Mi |

---

## Variables d'Environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `FLASK_ENV` | Environnement Flask | production |
| `LOG_LEVEL` | Niveau de log | INFO |
| `MODEL_PATH` | Chemin du modèle | models/ |
| `MLFLOW_TRACKING_URI` | URI du serveur MLflow | http://localhost:5001 |
| `WORKERS` | Nombre de workers Gunicorn | 2 |

---

## Prochaines Étapes

1. **[03-approche-github-actions-argocd.md](03-approche-github-actions-argocd.md)** - Détails sur l'approche GitOps
2. **[04-approche-kubeflow-mlflow.md](04-approche-kubeflow-mlflow.md)** - Détails sur Kubeflow et MLflow
3. **[05-guide-demarrage-rapide.md](05-guide-demarrage-rapide.md)** - Guide pas à pas
