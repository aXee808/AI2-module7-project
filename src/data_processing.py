"""
Data Processing Module
======================
Génération et traitement des données pour la prédiction de stocks.

Ce module contient les fonctions pour :
- Générer des données synthétiques de stock
- Charger des données depuis des fichiers
- Diviser les données en train/validation/test
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional


def generate_synthetic_stock_data(
    ticker: str = "SYNTH",
    days: int = 500,
    start_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Génère des données synthétiques de stock réalistes.

    Les données simulées incluent les patterns typiques des marchés :
    - Tendance (haussière/baissière)
    - Volatilité
    - Saisonnalité (effet jour de la semaine)
    - Bruit aléatoire

    Args:
        ticker: Symbole du stock
        days: Nombre de jours à générer
        start_price: Prix initial
        volatility: Volatilité quotidienne (écart-type des rendements)
        trend: Tendance quotidienne moyenne
        seed: Graine pour la reproductibilité

    Returns:
        DataFrame avec colonnes: Date, Ticker, Open, High, Low, Close, Volume

    Example:
        >>> df = generate_synthetic_stock_data(days=100, seed=42)
        >>> print(df.head())
    """
    if seed is not None:
        np.random.seed(seed)

    # Générer les dates (jours ouvrés uniquement)
    dates = []
    current_date = datetime(2023, 1, 1)
    while len(dates) < days:
        # Exclure les weekends
        if current_date.weekday() < 5:
            dates.append(current_date)
        current_date += timedelta(days=1)

    # Générer les prix avec mouvement brownien géométrique
    returns = np.random.normal(trend, volatility, days)

    # Ajouter un effet de saisonnalité (lundi généralement moins bon)
    day_effects = np.array([dates[i].weekday() for i in range(days)])
    monday_effect = np.where(day_effects == 0, -0.001, 0)
    friday_effect = np.where(day_effects == 4, 0.0005, 0)
    returns = returns + monday_effect + friday_effect

    # Calculer les prix de clôture
    close_prices = start_price * np.exp(np.cumsum(returns))

    # Générer Open, High, Low basés sur Close
    daily_range = volatility * close_prices  # Range proportionnel au prix

    # Open : proche du close précédent avec un gap
    open_prices = np.zeros(days)
    open_prices[0] = start_price
    for i in range(1, days):
        gap = np.random.normal(0, volatility * 0.5) * close_prices[i-1]
        open_prices[i] = close_prices[i-1] + gap

    # High et Low
    high_extension = np.abs(np.random.normal(0.5, 0.2, days)) * daily_range
    low_extension = np.abs(np.random.normal(0.5, 0.2, days)) * daily_range

    high_prices = np.maximum(open_prices, close_prices) + high_extension
    low_prices = np.minimum(open_prices, close_prices) - low_extension

    # Volume : plus élevé les jours de forte variation
    price_change = np.abs(close_prices - open_prices) / open_prices
    base_volume = 1_000_000
    volume = base_volume * (1 + 10 * price_change) * np.random.uniform(0.5, 1.5, days)
    volume = volume.astype(int)

    # Créer le DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Ticker': ticker,
        'Open': np.round(open_prices, 2),
        'High': np.round(high_prices, 2),
        'Low': np.round(low_prices, 2),
        'Close': np.round(close_prices, 2),
        'Volume': volume
    })

    return df


def load_or_generate_data(
    filepath: str = "data/raw/stock_data.csv",
    generate_if_missing: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Charge les données depuis un fichier ou les génère si nécessaire.

    Args:
        filepath: Chemin vers le fichier CSV
        generate_if_missing: Si True, génère les données si le fichier n'existe pas
        **kwargs: Arguments passés à generate_synthetic_stock_data

    Returns:
        DataFrame avec les données de stock

    Raises:
        FileNotFoundError: Si le fichier n'existe pas et generate_if_missing=False
    """
    if os.path.exists(filepath):
        print(f"📂 Chargement des données depuis {filepath}")
        df = pd.read_csv(filepath, parse_dates=['Date'])
        return df

    if generate_if_missing:
        print("📊 Génération de données synthétiques...")
        df = generate_synthetic_stock_data(**kwargs)

        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Sauvegarder
        df.to_csv(filepath, index=False)
        print(f"💾 Données sauvegardées dans {filepath}")

        return df

    raise FileNotFoundError(f"Fichier non trouvé : {filepath}")


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    shuffle: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divise les données en ensembles train/validation/test.

    Pour les séries temporelles, on NE mélange PAS les données
    pour respecter l'ordre chronologique.

    Args:
        df: DataFrame à diviser
        train_ratio: Proportion pour l'entraînement (défaut: 70%)
        val_ratio: Proportion pour la validation (défaut: 15%)
        shuffle: Mélanger les données (False recommandé pour séries temporelles)

    Returns:
        Tuple de (train_df, val_df, test_df)

    Example:
        >>> train, val, test = split_data(df, train_ratio=0.7, val_ratio=0.15)
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """
    n = len(df)

    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculer les indices de séparation
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Diviser
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"📊 Split des données:")
    print(f"   Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    print(f"   Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    print(f"   Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")

    return train_df, val_df, test_df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Valide la qualité des données.

    Vérifie :
    - Présence des colonnes requises
    - Valeurs manquantes
    - Cohérence OHLC (High >= Low, etc.)
    - Types de données

    Args:
        df: DataFrame à valider

    Returns:
        Dictionnaire avec les résultats de validation
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    # Colonnes requises
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        results['valid'] = False
        results['errors'].append(f"Colonnes manquantes: {missing_cols}")
        return results

    # Valeurs manquantes
    null_counts = df[required_cols].isnull().sum()
    if null_counts.sum() > 0:
        results['warnings'].append(f"Valeurs manquantes: {null_counts.to_dict()}")

    # Cohérence OHLC
    if not (df['High'] >= df['Low']).all():
        results['valid'] = False
        results['errors'].append("High < Low pour certaines lignes")

    if not ((df['Open'] >= df['Low']) & (df['Open'] <= df['High'])).all():
        results['warnings'].append("Open hors de [Low, High] pour certaines lignes")

    if not ((df['Close'] >= df['Low']) & (df['Close'] <= df['High'])).all():
        results['warnings'].append("Close hors de [Low, High] pour certaines lignes")

    # Volumes négatifs
    if (df['Volume'] < 0).any():
        results['valid'] = False
        results['errors'].append("Volumes négatifs détectés")

    # Statistiques
    results['stats'] = {
        'n_rows': len(df),
        'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
        'price_range': f"{df['Close'].min():.2f} - {df['Close'].max():.2f}",
        'avg_volume': f"{df['Volume'].mean():,.0f}"
    }

    return results


# Point d'entrée pour tests
if __name__ == "__main__":
    # Test de génération
    print("=" * 50)
    print("Test de génération de données")
    print("=" * 50)

    df = generate_synthetic_stock_data(days=100, seed=42)
    print(f"\nDonnées générées : {len(df)} lignes")
    print(df.head(10))

    # Test de validation
    print("\n" + "=" * 50)
    print("Test de validation")
    print("=" * 50)

    validation = validate_data(df)
    print(f"Valid: {validation['valid']}")
    print(f"Stats: {validation['stats']}")

    # Test de split
    print("\n" + "=" * 50)
    print("Test de split")
    print("=" * 50)

    train, val, test = split_data(df)
