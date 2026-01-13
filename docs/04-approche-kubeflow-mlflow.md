# Approche 2 & 3 : Kubeflow et MLflow

Ce document détaille deux approches complémentaires pour le MLOps : Kubeflow pour l'orchestration de pipelines ML à grande échelle, et MLflow pour le tracking d'expériences et la gestion de modèles.

---

## Partie A : Kubeflow

### Vue d'Ensemble

Kubeflow est une plateforme open-source conçue pour déployer des workflows de Machine Learning sur Kubernetes. Elle fournit un ensemble d'outils pour chaque étape du cycle de vie ML.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KUBEFLOW PLATFORM                                  │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Notebooks   │  │  Pipelines   │  │   KFServing  │  │   Katib      │   │
│  │  (Jupyter)   │  │  (Workflow)  │  │  (Serving)   │  │  (AutoML)    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Kubernetes Cluster                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Composants Principaux

| Composant | Description | Usage |
|-----------|-------------|-------|
| **Kubeflow Pipelines** | Orchestration de workflows ML | Automatiser les pipelines |
| **Notebooks** | Jupyter notebooks sur K8s | Exploration et développement |
| **KFServing** | Serving de modèles | Inférence en production |
| **Katib** | Hyperparameter tuning | Optimisation automatique |
| **Training Operators** | TFJob, PyTorchJob | Entraînement distribué |

### Kubeflow Pipelines (KFP)

#### Concepts Clés

```
┌─────────────────────────────────────────────────────────────────┐
│                       KUBEFLOW PIPELINE                          │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │Component │───▶│Component │───▶│Component │───▶│Component │ │
│  │  Data    │    │ Feature  │    │  Train   │    │  Deploy  │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│       │               │               │               │        │
│       ▼               ▼               ▼               ▼        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ Dataset  │    │ Dataset  │    │  Model   │    │ Endpoint │ │
│  │ Artifact │    │ Artifact │    │ Artifact │    │          │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

| Concept | Description |
|---------|-------------|
| **Pipeline** | DAG de composants définissant le workflow complet |
| **Component** | Unité de travail conteneurisée avec inputs/outputs |
| **Artifact** | Données produites/consommées par les composants |
| **Run** | Exécution d'un pipeline avec des paramètres |
| **Experiment** | Groupe logique de runs |

#### Structure d'un Component

```python
from kfp.dsl import component, Output, Input, Dataset, Model

@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def my_component(
    input_data: Input[Dataset],      # Input artifact
    output_model: Output[Model],     # Output artifact
    param: int = 10                  # Parameter
):
    """Description du composant."""
    import pandas as pd

    # Lire l'input
    df = pd.read_csv(input_data.path)

    # Traitement...

    # Écrire l'output
    model.save(output_model.path)
```

### Notre Pipeline Stock Prediction

#### Composant 1 : Génération des Données

```python
@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "numpy"]
)
def generate_data(
    output_data: Output[Dataset],
    days: int = 500,
    seed: int = 42
):
    """Génère des données synthétiques de stock."""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    np.random.seed(seed)

    # Générer les dates (jours ouvrés)
    dates = []
    current_date = datetime(2023, 1, 1)
    while len(dates) < days:
        if current_date.weekday() < 5:
            dates.append(current_date)
        current_date += timedelta(days=1)

    # Prix avec mouvement brownien géométrique
    start_price = 100.0
    volatility = 0.02
    returns = np.random.normal(0.0001, volatility, days)
    close_prices = start_price * np.exp(np.cumsum(returns))

    # OHLCV
    daily_range = volatility * close_prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = start_price
    high_prices = np.maximum(open_prices, close_prices) + \
                  np.abs(np.random.normal(0.5, 0.2, days)) * daily_range
    low_prices = np.minimum(open_prices, close_prices) - \
                 np.abs(np.random.normal(0.5, 0.2, days)) * daily_range
    volume = (1_000_000 * np.random.uniform(0.5, 1.5, days)).astype(int)

    df = pd.DataFrame({
        'Date': dates,
        'Ticker': 'SYNTH',
        'Open': np.round(open_prices, 2),
        'High': np.round(high_prices, 2),
        'Low': np.round(low_prices, 2),
        'Close': np.round(close_prices, 2),
        'Volume': volume
    })

    df.to_csv(output_data.path, index=False)
    print(f"Données générées: {len(df)} lignes")
```

#### Composant 2 : Feature Engineering

```python
@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "numpy"]
)
def feature_engineering(
    input_data: Input[Dataset],
    output_data: Output[Dataset]
):
    """Crée les features techniques."""
    import pandas as pd
    import numpy as np

    df = pd.read_csv(input_data.path, parse_dates=['Date'])

    # Moyennes mobiles
    for window in [5, 10, 20, 50]:
        df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
        if window <= 20:
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)
    avg_gains = gains.ewm(com=13).mean()
    avg_losses = losses.ewm(com=13).mean()
    rs = avg_gains / avg_losses
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_Width'] = (2 * std20 * 2) / sma20
    df['BB_Position'] = (df['Close'] - (sma20 - 2*std20)) / (4*std20)

    # Returns & Volatilité
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Volatility_10d'] = df['Return_1d'].rolling(10).std()
    df['Volatility_20d'] = df['Return_1d'].rolling(20).std()

    # Volume relatif
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # Prix relatifs
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Close_Position'] = (df['Close'] - df['Low']) / \
                           (df['High'] - df['Low'] + 1e-10)
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # Target: le prix monte demain ?
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Supprimer les NaN
    df = df.dropna()

    df.to_csv(output_data.path, index=False)
    print(f"Features créées: {len(df)} lignes, {len(df.columns)} colonnes")
```

#### Composant 3 : Entraînement

```python
@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "joblib"]
)
def train_model(
    input_data: Input[Dataset],
    output_model: Output[Model],
    output_metrics: Output[Metrics],
    model_type: str = "random_forest",
    test_size: float = 0.2
):
    """Entraîne le modèle de classification."""
    import pandas as pd
    import numpy as np
    import joblib
    import os
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    df = pd.read_csv(input_data.path)

    # Features
    feature_cols = [
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
        'EMA_5', 'EMA_10', 'EMA_20',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Width', 'BB_Position',
        'Return_1d', 'Return_5d', 'Return_10d',
        'Volatility_10d', 'Volatility_20d',
        'Volume_Ratio', 'High_Low_Range', 'Close_Position', 'Gap'
    ]

    X = df[feature_cols].values
    y = df['Target'].values

    # Split temporel (pas random pour les séries temporelles)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sélection du modèle
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        'logistic': LogisticRegression(max_iter=1000, random_state=42)
    }
    model = models.get(model_type, models['random_forest'])
    model.fit(X_train_scaled, y_train)

    # Évaluation
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    # Logger les métriques
    for name, value in metrics.items():
        output_metrics.log_metric(name, value)
        print(f"  {name}: {value:.4f}")

    # Sauvegarder le modèle
    os.makedirs(output_model.path, exist_ok=True)
    joblib.dump(model, f"{output_model.path}/model.joblib")
    joblib.dump(scaler, f"{output_model.path}/scaler.joblib")
    joblib.dump(feature_cols, f"{output_model.path}/features.joblib")

    print(f"Modèle entraîné: {model_type}")
```

#### Composant 4 : Décision de Déploiement

```python
@component(
    base_image="python:3.11-slim"
)
def evaluate_and_decide(
    metrics: Input[Metrics],
    accuracy_threshold: float = 0.55
) -> NamedTuple('Outputs', [('deploy', bool)]):
    """Décide si le modèle doit être déployé."""
    from collections import namedtuple

    # Note: Dans un vrai pipeline, lire depuis metrics
    accuracy = 0.60  # Placeholder

    deploy = accuracy >= accuracy_threshold

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Seuil: {accuracy_threshold}")
    print(f"Déployer: {deploy}")

    Outputs = namedtuple('Outputs', ['deploy'])
    return Outputs(deploy)
```

#### Composant 5 : Déploiement

```python
@component(
    base_image="python:3.11-slim"
)
def deploy_model(
    model: Input[Model],
    deploy: bool
):
    """Déploie le modèle si approuvé."""
    if deploy:
        print(f"Déploiement du modèle depuis {model.path}")
        # Actions possibles:
        # - kubectl apply pour KFServing
        # - Mise à jour d'un endpoint Seldon
        # - Push vers un model registry
    else:
        print("Déploiement ignoré (métriques insuffisantes)")
```

#### Pipeline Complet

```python
from kfp import dsl

@dsl.pipeline(
    name="Stock Prediction Pipeline",
    description="Pipeline MLOps pour la prédiction de stocks"
)
def stock_prediction_pipeline(
    days: int = 500,
    seed: int = 42,
    model_type: str = "random_forest",
    accuracy_threshold: float = 0.55
):
    """Pipeline complet."""

    # Étape 1: Générer les données
    data_task = generate_data(days=days, seed=seed)

    # Étape 2: Feature engineering
    features_task = feature_engineering(
        input_data=data_task.outputs['output_data']
    )

    # Étape 3: Entraînement
    train_task = train_model(
        input_data=features_task.outputs['output_data'],
        model_type=model_type
    )

    # Étape 4: Décision
    decision_task = evaluate_and_decide(
        metrics=train_task.outputs['output_metrics'],
        accuracy_threshold=accuracy_threshold
    )

    # Étape 5: Déploiement conditionnel
    deploy_task = deploy_model(
        model=train_task.outputs['output_model'],
        deploy=decision_task.outputs['deploy']
    )


# Compilation
if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=stock_prediction_pipeline,
        package_path="pipeline.yaml"
    )
    print("Pipeline compilé: pipeline.yaml")
```

### Installation de Kubeflow

#### Option 1 : Kubeflow sur Minikube (Simplifié)

```bash
# Prérequis: Minikube avec suffisamment de ressources
minikube start --cpus=4 --memory=8192 --disk-size=40g

# Installer kfctl
curl -LO https://github.com/kubeflow/kfctl/releases/download/v1.2.0/kfctl_v1.2.0-0-gbc038f9_linux.tar.gz
tar -xvf kfctl_v1.2.0-0-gbc038f9_linux.tar.gz
sudo mv kfctl /usr/local/bin/

# Déployer Kubeflow
export KF_NAME=kubeflow
export BASE_DIR=/opt
export KF_DIR=${BASE_DIR}/${KF_NAME}
export CONFIG_URI="https://raw.githubusercontent.com/kubeflow/manifests/v1.2-branch/kfdef/kfctl_k8s_istio.v1.2.0.yaml"

mkdir -p ${KF_DIR}
cd ${KF_DIR}
kfctl apply -V -f ${CONFIG_URI}

# Accéder à l'UI
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80 &
# Ouvrir http://localhost:8080
```

#### Option 2 : Kubeflow Pipelines Standalone

```bash
# Plus léger, juste les pipelines
export PIPELINE_VERSION=2.0.0

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"

# Accéder
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 &
```

### Exécuter le Pipeline

```bash
# 1. Installer le SDK
pip install kfp

# 2. Compiler le pipeline
python kubeflow/pipeline.py
# Génère: pipeline.yaml

# 3. Upload via l'UI ou CLI
# UI: http://localhost:8080 > Pipelines > Upload
# CLI:
kfp pipeline upload pipeline.yaml

# 4. Créer un run
# Via l'UI ou:
kfp run submit -e "stock-experiment" -r "run-001" -f pipeline.yaml
```

---

## Partie B : MLflow

### Vue d'Ensemble

MLflow est une plateforme open-source pour gérer le cycle de vie complet du Machine Learning. C'est l'approche la plus simple et la plus rapide à mettre en place.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MLFLOW PLATFORM                                 │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│  │     Tracking     │  │     Projects     │  │      Models      │         │
│  │                  │  │                  │  │                  │         │
│  │  - Experiments   │  │  - Packaging     │  │  - Registry      │         │
│  │  - Parameters    │  │  - Dependencies  │  │  - Versioning    │         │
│  │  - Metrics       │  │  - Reproducible  │  │  - Staging       │         │
│  │  - Artifacts     │  │    runs          │  │  - Production    │         │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Composants MLflow

| Composant | Description |
|-----------|-------------|
| **MLflow Tracking** | API et UI pour logger paramètres, métriques, artifacts |
| **MLflow Projects** | Format standard pour packager le code ML |
| **MLflow Models** | Format standard pour packager les modèles |
| **Model Registry** | Store centralisé pour gérer les versions de modèles |

### MLflow Tracking

#### Concepts Clés

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLFLOW TRACKING                           │
│                                                                 │
│  Experiment: "stock-prediction"                                 │
│  │                                                              │
│  ├── Run 1 (random_forest)                                     │
│  │   ├── Parameters: n_estimators=100, max_depth=10            │
│  │   ├── Metrics: accuracy=0.65, f1=0.62                       │
│  │   └── Artifacts: model.joblib, confusion_matrix.png         │
│  │                                                              │
│  ├── Run 2 (gradient_boosting)                                 │
│  │   ├── Parameters: n_estimators=100, learning_rate=0.1       │
│  │   ├── Metrics: accuracy=0.68, f1=0.65                       │
│  │   └── Artifacts: model.joblib, feature_importance.csv       │
│  │                                                              │
│  └── Run 3 (logistic)                                          │
│      ├── Parameters: C=1.0, max_iter=1000                      │
│      ├── Metrics: accuracy=0.58, f1=0.55                       │
│      └── Artifacts: model.joblib                               │
└─────────────────────────────────────────────────────────────────┘
```

### Notre Script MLflow

#### Setup MLflow

```python
import mlflow
import os

def setup_mlflow(tracking_uri=None, experiment_name="stock-prediction"):
    """Configure MLflow."""
    # URI du serveur de tracking
    if tracking_uri is None:
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5001"
        )

    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    # Créer ou récupérer l'expérience
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={"project": "stock-prediction", "team": "ml"}
        )
        print(f"Expérience créée: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Expérience existante: {experiment_name}")

    mlflow.set_experiment(experiment_name)
    return experiment_id
```

#### Entraînement avec Tracking

```python
def train_with_mlflow(
    model_type="random_forest",
    params=None,
    register_model=False
):
    """Entraîne un modèle avec tracking MLflow complet."""
    from datetime import datetime
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    import json

    # Paramètres par défaut
    default_params = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    }
    if params is None:
        params = default_params.get(model_type, {})

    # Démarrer le run MLflow
    with mlflow.start_run(
        run_name=f"{model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ) as run:
        run_id = run.info.run_id
        print(f"MLflow Run: {run_id}")

        # 1. Logger les tags
        mlflow.set_tags({
            "model_type": model_type,
            "environment": "development",
            "author": os.environ.get("USER", "unknown")
        })

        # 2. Charger et préparer les données
        # ... (code de préparation des données)
        X_train, X_test, y_train, y_test = prepare_data()

        # 3. Logger les infos sur les données
        mlflow.log_params({
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "n_features": X_train.shape[1]
        })

        # 4. Logger les hyperparamètres
        mlflow.log_params(params)
        print(f"Params: {params}")

        # 5. Entraînement
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # 6. Évaluation
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        # 7. Logger la confusion matrix comme artifact
        cm = confusion_matrix(y_test, y_pred)
        with open("confusion_matrix.json", "w") as f:
            json.dump({"matrix": cm.tolist()}, f)
        mlflow.log_artifact("confusion_matrix.json")
        os.remove("confusion_matrix.json")

        # 8. Logger le modèle
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, y_pred)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5],
            registered_model_name="stock-prediction" if register_model else None
        )

        print(f"Run terminé: {run_id}")
        return run_id
```

#### Comparaison de Modèles

```python
def compare_models():
    """Compare plusieurs modèles avec MLflow."""
    print("=" * 60)
    print("COMPARAISON DES MODÈLES")
    print("=" * 60)

    models_to_test = [
        ("random_forest", {"n_estimators": 50, "max_depth": 5}),
        ("random_forest", {"n_estimators": 100, "max_depth": 10}),
        ("random_forest", {"n_estimators": 200, "max_depth": 15}),
        ("gradient_boosting", {"n_estimators": 100, "learning_rate": 0.1}),
        ("gradient_boosting", {"n_estimators": 100, "learning_rate": 0.05}),
        ("logistic", {"C": 0.1, "max_iter": 1000}),
        ("logistic", {"C": 1.0, "max_iter": 1000}),
    ]

    run_ids = []
    for model_type, params in models_to_test:
        print(f"\nTesting: {model_type} with {params}")
        run_id = train_with_mlflow(model_type=model_type, params=params)
        run_ids.append(run_id)

    print(f"\nRuns créés: {len(run_ids)}")
    print(f"Voir les résultats dans l'UI MLflow")
    return run_ids
```

### Model Registry

Le Model Registry permet de gérer les versions et les stages des modèles.

```
┌─────────────────────────────────────────────────────────────────┐
│                       MODEL REGISTRY                             │
│                                                                 │
│  Model: "stock-prediction"                                      │
│  │                                                              │
│  ├── Version 1                                                  │
│  │   ├── Stage: Archived                                        │
│  │   ├── Created: 2024-01-01                                   │
│  │   └── Metrics: accuracy=0.58                                │
│  │                                                              │
│  ├── Version 2                                                  │
│  │   ├── Stage: Staging                                        │
│  │   ├── Created: 2024-01-10                                   │
│  │   └── Metrics: accuracy=0.65                                │
│  │                                                              │
│  └── Version 3                                                  │
│      ├── Stage: Production                                      │
│      ├── Created: 2024-01-15                                   │
│      └── Metrics: accuracy=0.68                                │
└─────────────────────────────────────────────────────────────────┘
```

#### Stages disponibles

| Stage | Description |
|-------|-------------|
| **None** | Nouvellement enregistré |
| **Staging** | En test, validation |
| **Production** | En production |
| **Archived** | Archivé, plus utilisé |

#### Utilisation via Python

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Promouvoir un modèle en staging
client.transition_model_version_stage(
    name="stock-prediction",
    version=2,
    stage="Staging"
)

# Promouvoir en production
client.transition_model_version_stage(
    name="stock-prediction",
    version=3,
    stage="Production"
)

# Charger le modèle de production
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/stock-prediction/Production")
```

### Servir un Modèle MLflow

#### Option 1 : MLflow serve

```bash
# Servir le modèle du registry
mlflow models serve \
  -m "models:/stock-prediction/Production" \
  -p 5002 \
  --no-conda

# Tester
curl -X POST http://localhost:5002/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[105.2, 104.8, ...]]}'
```

#### Option 2 : Docker

```bash
# Construire l'image Docker
mlflow models build-docker \
  -m "models:/stock-prediction/Production" \
  -n "stock-prediction-server"

# Lancer le conteneur
docker run -p 5002:8080 stock-prediction-server

# Tester
curl -X POST http://localhost:5002/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[105.2, 104.8, ...]]}'
```

### Démarrage Rapide MLflow

```bash
# 1. Installer
pip install mlflow

# 2. Lancer le serveur de tracking
mlflow server \
  --host 0.0.0.0 \
  --port 5001 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

# 3. Dans un autre terminal, lancer l'entraînement
cd mlops-complet
python mlflow/train_with_mlflow.py

# 4. Comparer plusieurs modèles
python mlflow/train_with_mlflow.py --compare

# 5. Enregistrer le meilleur modèle
python mlflow/train_with_mlflow.py --register

# 6. Ouvrir l'UI
# http://localhost:5001

# 7. Servir le modèle
mlflow models serve -m "models:/stock-prediction/Production" -p 5002
```

---

## Comparaison : Kubeflow vs MLflow

| Critère | Kubeflow | MLflow |
|---------|----------|--------|
| **Complexité** | Élevée | Faible |
| **Infrastructure** | Kubernetes requis | Local ou serveur simple |
| **Scalabilité** | Très haute | Moyenne |
| **Cas d'usage** | Grandes équipes, production | Expérimentation, PME |
| **Temps de setup** | Heures/jours | Minutes |
| **Courbe d'apprentissage** | Abrupte | Douce |
| **Orchestration** | Pipeline complet | Tracking principalement |
| **Serving** | KFServing intégré | mlflow serve |

### Quand utiliser quoi ?

**Choisir Kubeflow si :**
- Vous avez déjà Kubernetes en production
- Équipe de 10+ data scientists
- Besoin de pipelines complexes et automatisés
- Entraînement distribué sur GPU

**Choisir MLflow si :**
- Démarrage rapide souhaité
- Équipe petite à moyenne (1-10 personnes)
- Focus sur le tracking d'expériences
- Pas d'infrastructure Kubernetes

**Utiliser les deux :**
- MLflow pour le tracking dans Kubeflow Pipelines
- Meilleur des deux mondes

---

## Ressources

### Kubeflow
- [Documentation officielle](https://www.kubeflow.org/docs/)
- [Kubeflow Pipelines SDK](https://kubeflow-pipelines.readthedocs.io/)
- [GitHub](https://github.com/kubeflow/kubeflow)

### MLflow
- [Documentation officielle](https://mlflow.org/docs/latest/index.html)
- [Tutoriels](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)
- [GitHub](https://github.com/mlflow/mlflow)
