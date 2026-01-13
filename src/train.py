"""
Training Module
===============
Entraînement et évaluation des modèles de prédiction de stocks.

Ce module contient :
- La classe StockPredictor pour encapsuler le modèle
- Les fonctions d'entraînement avec tracking MLflow
- L'évaluation et les métriques
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional, Any, List

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


class StockPredictor:
    """
    Classe principale pour la prédiction de stocks.

    Encapsule le modèle ML, le scaler, et fournit des méthodes
    pour l'entraînement, la prédiction et l'évaluation.

    Attributes:
        model_type: Type de modèle ('random_forest', 'gradient_boosting', 'logistic')
        model: Le modèle entraîné
        scaler: StandardScaler pour normaliser les features
        feature_names: Liste des noms de features
        metadata: Métadonnées du modèle
    """

    SUPPORTED_MODELS = {
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'logistic': LogisticRegression
    }

    DEFAULT_PARAMS = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'logistic': {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42
        }
    }

    def __init__(
        self,
        model_type: str = 'random_forest',
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise le prédicteur.

        Args:
            model_type: Type de modèle à utiliser
            params: Paramètres du modèle (optionnel)
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Modèle non supporté: {model_type}. "
                f"Choix: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.params = params or self.DEFAULT_PARAMS.get(model_type, {})
        self.metadata = {
            'model_type': model_type,
            'created_at': None,
            'trained_at': None,
            'version': '1.0.0'
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Entraîne le modèle.

        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            X_val: Features de validation (optionnel)
            y_val: Labels de validation (optionnel)
            feature_names: Noms des features (optionnel)

        Returns:
            Dictionnaire avec les métriques d'entraînement
        """
        print(f"\n🚀 Entraînement du modèle {self.model_type}")
        print(f"   Samples: {len(X_train)}")
        print(f"   Features: {X_train.shape[1]}")

        # Sauvegarder les noms de features
        if feature_names is not None:
            self.feature_names = feature_names

        # Normaliser les données
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Créer et entraîner le modèle
        model_class = self.SUPPORTED_MODELS[self.model_type]
        self.model = model_class(**self.params)
        self.model.fit(X_train_scaled, y_train)

        # Métriques sur train
        y_train_pred = self.model.predict(X_train_scaled)
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, zero_division=0)
        }

        # Métriques sur validation si disponible
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val, prefix='val')
            metrics.update(val_metrics)

        # Mise à jour des métadonnées
        self.metadata['trained_at'] = datetime.now().isoformat()
        self.metadata['n_samples'] = len(X_train)
        self.metadata['n_features'] = X_train.shape[1]
        self.metadata['params'] = self.params

        print(f"\n📊 Métriques d'entraînement:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")

        return metrics

    def predict(
        self,
        X: np.ndarray,
        return_proba: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédit les classes et probabilités.

        Args:
            X: Features pour la prédiction
            return_proba: Retourner aussi les probabilités

        Returns:
            Tuple de (predictions, probabilities)

        Raises:
            ValueError: Si le modèle n'est pas entraîné
        """
        if self.model is None:
            raise ValueError("Le modèle n'est pas entraîné. Appelez train() d'abord.")

        # Normaliser
        X_scaled = self.scaler.transform(X)

        # Prédire
        predictions = self.model.predict(X_scaled)

        # Probabilités
        if return_proba and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = predictions.astype(float)

        return predictions, probabilities

    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Prédit pour un seul échantillon à partir d'un dictionnaire de features.

        Args:
            features: Dictionnaire {feature_name: value}

        Returns:
            Dictionnaire avec la prédiction et les métadonnées
        """
        # Convertir en array
        X = np.array([[features.get(f, 0) for f in self.feature_names]])

        predictions, probabilities = self.predict(X)

        return {
            'prediction': int(predictions[0]),
            'probability': float(probabilities[0]),
            'direction': 'UP' if predictions[0] == 1 else 'DOWN',
            'confidence': float(max(probabilities[0], 1 - probabilities[0]))
        }

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prefix: str = 'test'
    ) -> Dict[str, Any]:
        """
        Évalue le modèle sur un ensemble de données.

        Args:
            X: Features
            y: Labels réels
            prefix: Préfixe pour les noms de métriques

        Returns:
            Dictionnaire avec toutes les métriques
        """
        predictions, probabilities = self.predict(X)

        metrics = {
            f'{prefix}_accuracy': accuracy_score(y, predictions),
            f'{prefix}_precision': precision_score(y, predictions, zero_division=0),
            f'{prefix}_recall': recall_score(y, predictions, zero_division=0),
            f'{prefix}_f1': f1_score(y, predictions, zero_division=0),
        }

        # AUC-ROC
        try:
            metrics[f'{prefix}_auc_roc'] = roc_auc_score(y, probabilities)
        except Exception:
            metrics[f'{prefix}_auc_roc'] = 0.0

        # Matrice de confusion
        metrics['confusion_matrix'] = confusion_matrix(y, predictions).tolist()

        return metrics

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Retourne l'importance des features (si disponible).

        Returns:
            DataFrame avec les features et leur importance, ou None
        """
        if self.model is None:
            return None

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            return None

        if not self.feature_names:
            self.feature_names = [f'feature_{i}' for i in range(len(importance))]

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df

    def save(self, path: str) -> str:
        """
        Sauvegarde le modèle et ses composants.

        Args:
            path: Répertoire de sauvegarde

        Returns:
            Chemin du répertoire de sauvegarde
        """
        os.makedirs(path, exist_ok=True)

        # Sauvegarder le modèle
        model_path = os.path.join(path, 'model.joblib')
        joblib.dump(self.model, model_path)

        # Sauvegarder le scaler
        scaler_path = os.path.join(path, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)

        # Sauvegarder les métadonnées
        self.metadata['feature_names'] = self.feature_names
        metadata_path = os.path.join(path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"💾 Modèle sauvegardé dans {path}")
        return path

    @classmethod
    def load(cls, path: str) -> 'StockPredictor':
        """
        Charge un modèle sauvegardé.

        Args:
            path: Répertoire contenant le modèle

        Returns:
            Instance de StockPredictor
        """
        # Charger les métadonnées
        metadata_path = os.path.join(path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Créer l'instance
        predictor = cls(model_type=metadata['model_type'])
        predictor.metadata = metadata
        predictor.feature_names = metadata.get('feature_names', [])

        # Charger le modèle
        model_path = os.path.join(path, 'model.joblib')
        predictor.model = joblib.load(model_path)

        # Charger le scaler
        scaler_path = os.path.join(path, 'scaler.joblib')
        predictor.scaler = joblib.load(scaler_path)

        print(f"📂 Modèle chargé depuis {path}")
        return predictor


def train_and_save(
    model_dir: str = "models",
    data_path: str = "data/raw/stock_data.csv",
    model_type: str = "random_forest"
) -> Dict[str, Any]:
    """
    Pipeline complet d'entraînement.

    Args:
        model_dir: Répertoire pour sauvegarder le modèle
        data_path: Chemin vers les données
        model_type: Type de modèle

    Returns:
        Dictionnaire avec les métriques finales
    """
    from data_processing import load_or_generate_data, split_data
    from feature_engineering import create_features, prepare_training_data

    print("=" * 60)
    print("🚀 PIPELINE D'ENTRAINEMENT")
    print("=" * 60)

    # 1. Charger les données
    print("\n📊 Étape 1: Chargement des données")
    df = load_or_generate_data(data_path, days=500, seed=42)

    # 2. Feature Engineering
    print("\n🔧 Étape 2: Feature Engineering")
    df_features = create_features(df)

    # 3. Préparer les données
    print("\n📋 Étape 3: Préparation des données")
    X, y, feature_names = prepare_training_data(df_features)

    # 4. Split
    print("\n✂️ Étape 4: Split train/val/test")
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 5. Entraîner
    print("\n🏋️ Étape 5: Entraînement")
    predictor = StockPredictor(model_type=model_type)
    train_metrics = predictor.train(
        X_train, y_train,
        X_val, y_val,
        feature_names=feature_names
    )

    # 6. Évaluation finale sur test
    print("\n📊 Étape 6: Évaluation sur Test")
    test_metrics = predictor.evaluate(X_test, y_test, prefix='test')

    print(f"\n🎯 Métriques sur Test:")
    for key, value in test_metrics.items():
        if key != 'confusion_matrix':
            print(f"   {key}: {value:.4f}")

    # 7. Feature Importance
    print("\n📈 Étape 7: Importance des Features")
    importance = predictor.get_feature_importance()
    if importance is not None:
        print(importance.head(10).to_string(index=False))

    # 8. Sauvegarder
    print("\n💾 Étape 8: Sauvegarde")
    predictor.save(model_dir)

    # Combiner les métriques
    all_metrics = {**train_metrics, **test_metrics}

    print("\n" + "=" * 60)
    print("✅ ENTRAINEMENT TERMINÉ")
    print("=" * 60)

    return all_metrics


# Point d'entrée principal
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraîner le modèle de prédiction de stocks")
    parser.add_argument('--model-dir', default='models', help='Répertoire de sortie')
    parser.add_argument('--data-path', default='data/raw/stock_data.csv', help='Chemin des données')
    parser.add_argument('--model-type', default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'logistic'])

    args = parser.parse_args()

    metrics = train_and_save(
        model_dir=args.model_dir,
        data_path=args.data_path,
        model_type=args.model_type
    )
