"""
Module for cleaning financial data
"""
import pandas as pd
import numpy as np
from typing import List, Optional


def handle_missing_values(df: pd.DataFrame, strategy: str = 'ffill') -> pd.DataFrame:
    """
    Handle missing values in financial data
    """
    if strategy == 'ffill':
        df = df.fillna(method='ffill')
    elif strategy == 'bfill':
        df = df.fillna(method='bfill')
    elif strategy == 'mean':
        df = df.fillna(df.mean())
    elif strategy == 'interpolate':
        df = df.interpolate()

    return df


def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect and handle outliers in specified column
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = z_scores > threshold

    return df[outliers]


def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from specified column
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df = df[z_scores <= threshold]

    return df


def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main cleaning function for financial data with stock indices and economic indicators
    """
    # Make a copy to avoid modifying original data
    df_clean = df.copy()

    # Ensure Date column is datetime
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])

    # Sort by date to ensure chronological order
    df_clean = df_clean.sort_values('Date').reset_index(drop=True)

    # Handle missing values with forward fill (common in financial data)
    df_clean = handle_missing_values(df_clean, strategy='ffill')

    # Remove extreme outliers in price columns (Open Price, Close Price, Daily High, Daily Low)
    price_columns = ['Open Price', 'Close Price', 'Daily High', 'Daily Low']
    for col in price_columns:
        if col in df_clean.columns:
            df_clean = remove_outliers(df_clean, col, method='iqr', threshold=3.0)

    # Ensure price values are non-negative
    for col in price_columns:
        if col in df_clean.columns:
            df_clean = df_clean[df_clean[col] >= 0]

    # Ensure Trading Volume is non-negative
    if 'Trading Volume' in df_clean.columns:
        df_clean = df_clean[df_clean['Trading Volume'] >= 0]

    # Special handling for Bitcoin data - if Stock Index is BTC
    if 'Stock Index' in df_clean.columns and df_clean['Stock Index'].nunique() == 1 and df_clean['Stock Index'].iloc[0] == 'BTC':
        # Additional Bitcoin-specific cleaning
        # Remove any rows with zero prices (which should not happen for Bitcoin)
        for col in price_columns:
            if col in df_clean.columns:
                df_clean = df_clean[df_clean[col] > 0]

        # If volume exists, remove negative values
        if 'Trading Volume' in df_clean.columns:
            df_clean = df_clean[df_clean['Trading Volume'] >= 0]

    return df_clean