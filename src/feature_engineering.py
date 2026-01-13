"""
Feature Engineering Module
==========================
Création de features pour la prédiction de stocks.

Ce module implémente les indicateurs techniques classiques :
- Moyennes mobiles (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bandes de Bollinger
- Volatilité
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """
    Calcule la Moyenne Mobile Simple (SMA).

    La SMA est la moyenne arithmétique des N dernières valeurs.
    Utilisée pour identifier les tendances.

    Args:
        series: Série de prix
        window: Fenêtre de calcul (nombre de périodes)

    Returns:
        Série avec les valeurs SMA

    Formula:
        SMA = (P1 + P2 + ... + Pn) / n
    """
    return series.rolling(window=window, min_periods=window).mean()


def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    """
    Calcule la Moyenne Mobile Exponentielle (EMA).

    L'EMA donne plus de poids aux valeurs récentes.
    Plus réactive que la SMA aux changements de prix.

    Args:
        series: Série de prix
        window: Fenêtre de calcul

    Returns:
        Série avec les valeurs EMA

    Formula:
        EMA = Price(t) * k + EMA(t-1) * (1-k)
        où k = 2 / (N+1)
    """
    return series.ewm(span=window, adjust=False).mean()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calcule le RSI (Relative Strength Index).

    Le RSI mesure la vitesse et le changement des mouvements de prix.
    - RSI > 70 : Suracheté (potentielle baisse)
    - RSI < 30 : Survendu (potentielle hausse)

    Args:
        series: Série de prix de clôture
        window: Période de calcul (défaut: 14)

    Returns:
        Série avec les valeurs RSI (0-100)
    """
    delta = series.diff()

    # Séparer les gains et les pertes
    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)

    # Moyennes mobiles exponentielles
    avg_gains = gains.ewm(com=window - 1, min_periods=window).mean()
    avg_losses = losses.ewm(com=window - 1, min_periods=window).mean()

    # Calcul du RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcule le MACD (Moving Average Convergence Divergence).

    Le MACD est un indicateur de tendance et de momentum.
    - MACD > Signal : Signal haussier
    - MACD < Signal : Signal baissier

    Args:
        series: Série de prix de clôture
        fast: Période EMA rapide (défaut: 12)
        slow: Période EMA lente (défaut: 26)
        signal: Période de la ligne signal (défaut: 9)

    Returns:
        Tuple de (MACD, Signal, Histogram)
    """
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcule les Bandes de Bollinger.

    Les bandes mesurent la volatilité et identifient les niveaux de prix extrêmes.
    - Prix proche de la bande supérieure : Potentiellement suracheté
    - Prix proche de la bande inférieure : Potentiellement survendu

    Args:
        series: Série de prix
        window: Période de la moyenne mobile (défaut: 20)
        num_std: Nombre d'écarts-types (défaut: 2)

    Returns:
        Tuple de (upper_band, middle_band, lower_band)
    """
    middle = calculate_sma(series, window)
    std = series.rolling(window=window).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return upper, middle, lower


def calculate_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calcule la volatilité historique (écart-type des rendements).

    Args:
        series: Série de prix
        window: Fenêtre de calcul

    Returns:
        Série avec la volatilité
    """
    returns = series.pct_change()
    volatility = returns.rolling(window=window).std()
    return volatility


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée toutes les features techniques pour le modèle.

    Cette fonction est le point d'entrée principal pour le feature engineering.
    Elle ajoute de nombreux indicateurs techniques au DataFrame.

    Args:
        df: DataFrame avec colonnes OHLCV (Open, High, Low, Close, Volume)

    Returns:
        DataFrame avec les features ajoutées

    Features créées:
        - SMA (5, 10, 20, 50 jours)
        - EMA (5, 10, 20 jours)
        - RSI (14 jours)
        - MACD et composants
        - Bandes de Bollinger
        - Volatilité
        - Returns sur différentes périodes
        - Features de prix relatifs
    """
    df = df.copy()

    # === Moyennes Mobiles ===
    for window in [5, 10, 20, 50]:
        df[f'SMA_{window}'] = calculate_sma(df['Close'], window)

    for window in [5, 10, 20]:
        df[f'EMA_{window}'] = calculate_ema(df['Close'], window)

    # === RSI ===
    df['RSI'] = calculate_rsi(df['Close'], 14)

    # === MACD ===
    macd, signal, hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist

    # === Bandes de Bollinger ===
    upper, middle, lower = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower
    df['BB_Width'] = (upper - lower) / middle  # Largeur relative
    df['BB_Position'] = (df['Close'] - lower) / (upper - lower)  # Position dans les bandes

    # === Returns ===
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)

    # === Volatilité ===
    df['Volatility_10d'] = calculate_volatility(df['Close'], 10)
    df['Volatility_20d'] = calculate_volatility(df['Close'], 20)

    # === Features de Volume ===
    df['Volume_SMA_20'] = calculate_sma(df['Volume'], 20)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

    # === Features de Prix Relatifs ===
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # === Target: Direction du prix le lendemain ===
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df


def get_feature_columns() -> List[str]:
    """
    Retourne la liste des colonnes de features à utiliser pour le modèle.

    Returns:
        Liste des noms de colonnes
    """
    return [
        # Moyennes mobiles
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
        'EMA_5', 'EMA_10', 'EMA_20',
        # Indicateurs techniques
        'RSI',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Width', 'BB_Position',
        # Returns
        'Return_1d', 'Return_5d', 'Return_10d',
        # Volatilité
        'Volatility_10d', 'Volatility_20d',
        # Volume
        'Volume_Ratio',
        # Prix relatifs
        'High_Low_Range', 'Close_Position', 'Gap'
    ]


def prepare_training_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'Target'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prépare les données pour l'entraînement du modèle.

    - Supprime les lignes avec des valeurs manquantes
    - Sépare features et target
    - Convertit en numpy arrays

    Args:
        df: DataFrame avec features
        feature_cols: Liste des colonnes de features (défaut: get_feature_columns())
        target_col: Nom de la colonne cible

    Returns:
        Tuple de (X, y, feature_names)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    # Vérifier que toutes les colonnes existent
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")

    # Copier et supprimer les NaN
    df_clean = df[feature_cols + [target_col]].dropna()

    # Séparer X et y
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    print(f"📊 Données préparées : {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Distribution target: {np.bincount(y.astype(int))}")

    return X, y, feature_cols


# Point d'entrée pour tests
if __name__ == "__main__":
    from data_processing import generate_synthetic_stock_data

    print("=" * 50)
    print("Test du Feature Engineering")
    print("=" * 50)

    # Générer des données
    df = generate_synthetic_stock_data(days=200, seed=42)
    print(f"\nDonnées brutes : {len(df)} lignes")

    # Créer les features
    df_features = create_features(df)
    print(f"\nDonnées avec features : {len(df_features)} lignes, {len(df_features.columns)} colonnes")

    # Afficher les nouvelles colonnes
    new_cols = [col for col in df_features.columns if col not in df.columns]
    print(f"\nFeatures créées ({len(new_cols)}):")
    for col in new_cols:
        print(f"   - {col}")

    # Préparer les données
    print("\n" + "=" * 50)
    X, y, features = prepare_training_data(df_features)
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
