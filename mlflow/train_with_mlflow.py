"""
MLflow Training Script
======================
Entraînement avec tracking MLflow complet.

Ce script démontre l'utilisation de MLflow pour:
- Tracking des expériences
- Logging des paramètres et métriques
- Sauvegarde des modèles
- Comparaison des runs
- Déploiement du modèle

Usage:
    # Lancer le serveur MLflow d'abord:
    mlflow server --host 0.0.0.0 --port 5000

    # Puis lancer l'entraînement:
    python mlflow/train_with_mlflow.py
"""

import os
import sys
import json
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.model_selection import cross_val_score

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import load_or_generate_data, split_data
from feature_engineering import create_features, prepare_training_data


def setup_mlflow(tracking_uri: str = None, experiment_name: str = "stock-prediction"):
    """Configure MLflow."""
    # URI du serveur de tracking
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

    mlflow.set_tracking_uri(tracking_uri)
    print(f"📊 MLflow tracking URI: {tracking_uri}")

    # Créer ou récupérer l'expérience
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={"project": "stock-prediction", "team": "ml"}
        )
        print(f"✨ Expérience créée: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"📁 Expérience existante: {experiment_name} (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment_id


def train_with_mlflow(
    model_type: str = "random_forest",
    params: dict = None,
    data_path: str = "data/raw/stock_data.csv",
    register_model: bool = False
):
    """
    Entraîne un modèle avec tracking MLflow complet.

    Args:
        model_type: Type de modèle
        params: Hyperparamètres
        data_path: Chemin vers les données
        register_model: Enregistrer dans le Model Registry

    Returns:
        run_id: ID du run MLflow
    """
    # Paramètres par défaut
    default_params = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        },
        "logistic": {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42
        }
    }

    if params is None:
        params = default_params.get(model_type, {})

    # Créer le modèle
    models = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic": LogisticRegression
    }

    if model_type not in models:
        raise ValueError(f"Modèle non supporté: {model_type}")

    # Démarrer le run MLflow
    with mlflow.start_run(run_name=f"{model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
        run_id = run.info.run_id
        print(f"\n🚀 MLflow Run: {run_id}")
        print(f"   Model: {model_type}")

        # === ÉTAPE 1: Logger les tags ===
        mlflow.set_tags({
            "model_type": model_type,
            "environment": "development",
            "data_version": "v1",
            "author": os.environ.get("USER", "unknown")
        })

        # === ÉTAPE 2: Charger et préparer les données ===
        print("\n📊 Chargement des données...")
        df = load_or_generate_data(data_path, days=500, seed=42)
        df_features = create_features(df)
        X, y, feature_names = prepare_training_data(df_features)

        # Split
        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        # Logger les infos sur les données
        mlflow.log_params({
            "data_path": data_path,
            "n_samples_total": len(X),
            "n_samples_train": len(X_train),
            "n_samples_val": len(X_val),
            "n_samples_test": len(X_test),
            "n_features": X.shape[1],
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15
        })

        # === ÉTAPE 3: Normalisation ===
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # === ÉTAPE 4: Logger les hyperparamètres ===
        mlflow.log_params(params)
        print(f"   Params: {params}")

        # === ÉTAPE 5: Entraînement ===
        print("\n🏋️ Entraînement...")
        model = models[model_type](**params)
        model.fit(X_train_scaled, y_train)

        # === ÉTAPE 6: Évaluation ===
        print("\n📊 Évaluation...")

        # Prédictions
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Probabilités
        y_train_proba = model.predict_proba(X_train_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_train_pred
        y_val_proba = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_val_pred
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_test_pred

        # Métriques
        metrics = {}

        for prefix, y_true, y_pred, y_proba in [
            ("train", y_train, y_train_pred, y_train_proba),
            ("val", y_val, y_val_pred, y_val_proba),
            ("test", y_test, y_test_pred, y_test_proba)
        ]:
            metrics[f"{prefix}_accuracy"] = accuracy_score(y_true, y_pred)
            metrics[f"{prefix}_precision"] = precision_score(y_true, y_pred, zero_division=0)
            metrics[f"{prefix}_recall"] = recall_score(y_true, y_pred, zero_division=0)
            metrics[f"{prefix}_f1"] = f1_score(y_true, y_pred, zero_division=0)
            try:
                metrics[f"{prefix}_auc_roc"] = roc_auc_score(y_true, y_proba)
            except:
                metrics[f"{prefix}_auc_roc"] = 0.0

        # Logger les métriques
        mlflow.log_metrics(metrics)

        print("\n   Métriques:")
        for name, value in metrics.items():
            print(f"   {name}: {value:.4f}")

        # === ÉTAPE 7: Cross-validation ===
        print("\n📈 Cross-validation...")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        mlflow.log_metrics({
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std()
        })
        print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # === ÉTAPE 8: Confusion Matrix comme artefact ===
        cm = confusion_matrix(y_test, y_test_pred)
        cm_dict = {
            "confusion_matrix": cm.tolist(),
            "labels": ["DOWN", "UP"]
        }
        with open("confusion_matrix.json", "w") as f:
            json.dump(cm_dict, f)
        mlflow.log_artifact("confusion_matrix.json")
        os.remove("confusion_matrix.json")

        # === ÉTAPE 9: Feature Importance ===
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            importance_df.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            os.remove("feature_importance.csv")

            print("\n📈 Top 5 Features:")
            print(importance_df.head().to_string(index=False))

        # === ÉTAPE 10: Logger le modèle ===
        print("\n💾 Sauvegarde du modèle...")

        # Créer une signature
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train_scaled, y_train_pred)

        # Logger avec MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train_scaled[:5],
            registered_model_name="stock-prediction" if register_model else None
        )

        # Logger aussi le scaler
        mlflow.sklearn.log_model(scaler, artifact_path="scaler")

        # Logger les noms des features
        with open("feature_names.json", "w") as f:
            json.dump(feature_names, f)
        mlflow.log_artifact("feature_names.json")
        os.remove("feature_names.json")

        print(f"\n✅ Run terminé: {run_id}")
        print(f"   UI: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run_id}")

        return run_id


def compare_models():
    """Compare plusieurs modèles avec MLflow."""
    print("\n" + "=" * 60)
    print("🔬 COMPARAISON DES MODÈLES")
    print("=" * 60)

    models_to_test = [
        ("random_forest", {"n_estimators": 50, "max_depth": 5, "random_state": 42}),
        ("random_forest", {"n_estimators": 100, "max_depth": 10, "random_state": 42}),
        ("random_forest", {"n_estimators": 200, "max_depth": 15, "random_state": 42}),
        ("gradient_boosting", {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}),
        ("gradient_boosting", {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "random_state": 42}),
        ("logistic", {"C": 0.1, "max_iter": 1000, "random_state": 42}),
        ("logistic", {"C": 1.0, "max_iter": 1000, "random_state": 42}),
    ]

    run_ids = []
    for model_type, params in models_to_test:
        print(f"\n{'='*40}")
        print(f"Testing: {model_type} with {params}")
        run_id = train_with_mlflow(model_type=model_type, params=params)
        run_ids.append(run_id)

    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    print(f"   Runs créés: {len(run_ids)}")
    print(f"   Voir les résultats dans l'UI MLflow")
    print(f"   URL: {mlflow.get_tracking_uri()}")

    return run_ids


def serve_best_model(model_name: str = "stock-prediction"):
    """
    Sert le meilleur modèle du registry.

    Usage:
        mlflow models serve -m "models:/stock-prediction/Production" -p 5001
    """
    print(f"""
    Pour servir le modèle:

    1. Via MLflow CLI:
       mlflow models serve -m "models:/{model_name}/Production" -p 5001

    2. Ou avec Docker:
       mlflow models build-docker -m "models:/{model_name}/Production" -n "stock-prediction"
       docker run -p 5001:8080 stock-prediction

    3. Tester:
       curl -X POST http://localhost:5001/invocations \\
         -H "Content-Type: application/json" \\
         -d '{{"inputs": [[105.2, 104.8, 103.5, 100.2, 105.5, 105.0, 104.0, 55.0, 0.8, 0.5, 0.3, 0.05, 0.6, 0.01, 0.03, 0.05, 0.015, 0.018, 1.2, 0.02, 0.7, 0.002]]}}'
    """)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement MLflow")
    parser.add_argument("--model-type", default="random_forest",
                        choices=["random_forest", "gradient_boosting", "logistic"])
    parser.add_argument("--compare", action="store_true", help="Comparer plusieurs modèles")
    parser.add_argument("--register", action="store_true", help="Enregistrer dans le Model Registry")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI")

    args = parser.parse_args()

    # Setup
    setup_mlflow(tracking_uri=args.tracking_uri)

    if args.compare:
        compare_models()
    else:
        train_with_mlflow(
            model_type=args.model_type,
            register_model=args.register
        )
