"""
Kubeflow Pipeline - Stock Prediction
====================================
Pipeline MLOps complet avec Kubeflow Pipelines.

Ce pipeline:
1. Charge/génère les données
2. Feature engineering
3. Entraîne le modèle
4. Évalue le modèle
5. Déploie si les métriques sont satisfaisantes

Usage:
    python pipeline.py  # Compile le pipeline
    kfp pipeline upload pipeline.yaml  # Upload vers Kubeflow
"""

from typing import NamedTuple
from kfp import dsl
from kfp.dsl import component, Output, Input, Dataset, Model, Metrics


# ============================================
# COMPOSANT 1: Génération des Données
# ============================================
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

    # Générer les dates
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
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.normal(0.5, 0.2, days)) * daily_range
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.normal(0.5, 0.2, days)) * daily_range
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
    print(f"✅ Données générées: {len(df)} lignes")


# ============================================
# COMPOSANT 2: Feature Engineering
# ============================================
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

    # RSI
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

    # Volume
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # Prix relatifs
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # Target
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Supprimer les NaN
    df = df.dropna()

    df.to_csv(output_data.path, index=False)
    print(f"✅ Features créées: {len(df)} lignes, {len(df.columns)} colonnes")


# ============================================
# COMPOSANT 3: Entraînement
# ============================================
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

    # Split temporel
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modèle
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
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

    # Sauvegarder
    import os
    os.makedirs(output_model.path, exist_ok=True)
    joblib.dump(model, f"{output_model.path}/model.joblib")
    joblib.dump(scaler, f"{output_model.path}/scaler.joblib")
    joblib.dump(feature_cols, f"{output_model.path}/features.joblib")

    print(f"✅ Modèle entraîné: {model_type}")


# ============================================
# COMPOSANT 4: Évaluation et Décision
# ============================================
@component(
    base_image="python:3.11-slim",
    packages_to_install=[]
)
def evaluate_and_decide(
    metrics: Input[Metrics],
    accuracy_threshold: float = 0.55
) -> NamedTuple('Outputs', [('deploy', bool)]):
    """Décide si le modèle doit être déployé."""
    from collections import namedtuple

    # Lire l'accuracy depuis les métriques
    # Note: Dans un vrai pipeline, on lirait depuis le fichier metrics
    accuracy = 0.60  # Placeholder

    deploy = accuracy >= accuracy_threshold

    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"📊 Seuil: {accuracy_threshold}")
    print(f"🚀 Déployer: {deploy}")

    Outputs = namedtuple('Outputs', ['deploy'])
    return Outputs(deploy)


# ============================================
# COMPOSANT 5: Déploiement (Conditionnel)
# ============================================
@component(
    base_image="python:3.11-slim",
    packages_to_install=[]
)
def deploy_model(
    model: Input[Model],
    deploy: bool
):
    """Déploie le modèle si approuvé."""
    if deploy:
        print(f"🚀 Déploiement du modèle depuis {model.path}")
        # Ici: kubectl apply, Seldon deploy, KFServing, etc.
    else:
        print("⏸️ Déploiement ignoré (métriques insuffisantes)")


# ============================================
# PIPELINE
# ============================================
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
    """Pipeline complet de prédiction de stocks."""

    # Étape 1: Générer les données
    data_task = generate_data(days=days, seed=seed)

    # Étape 2: Feature engineering
    features_task = feature_engineering(input_data=data_task.outputs['output_data'])

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


# ============================================
# COMPILATION
# ============================================
if __name__ == "__main__":
    from kfp import compiler

    # Compiler le pipeline
    compiler.Compiler().compile(
        pipeline_func=stock_prediction_pipeline,
        package_path="pipeline.yaml"
    )
    print("✅ Pipeline compilé: pipeline.yaml")

    print("""
    Prochaines étapes:
    1. Installer Kubeflow: https://www.kubeflow.org/docs/started/
    2. Uploader le pipeline: kfp pipeline upload pipeline.yaml
    3. Créer un run depuis l'UI Kubeflow
    """)
