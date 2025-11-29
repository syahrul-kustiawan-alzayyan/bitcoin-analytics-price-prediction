"""
Module for creating financial features
"""
import pandas as pd
import numpy as np
from typing import List, Optional


def calculate_returns(df: pd.DataFrame, price_col: str = 'Close Price') -> pd.DataFrame:
    """
    Calculate simple and log returns
    """
    df = df.copy()
    # Calculate simple returns
    df[f'{price_col}_return'] = df[price_col].pct_change()
    # Calculate log returns
    df[f'{price_col}_log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def calculate_volatility(df: pd.DataFrame, price_col: str = 'Close Price', window: int = 20) -> pd.DataFrame:
    """
    Calculate rolling volatility
    """
    df = df.copy()
    # Calculate rolling standard deviation of returns as a measure of volatility
    df[f'{price_col}_volatility'] = df[f'{price_col}_return'].rolling(window=window).std()
    return df


def calculate_moving_averages(df: pd.DataFrame, price_col: str = 'Close Price', windows: List[int] = [20, 50, 200]) -> pd.DataFrame:
    """
    Calculate moving averages
    """
    df = df.copy()
    for window in windows:
        df[f'{price_col}_MA_{window}'] = df[price_col].rolling(window=window).mean()
    return df


def calculate_rsi(df: pd.DataFrame, price_col: str = 'Close Price', window: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI)
    """
    df = df.copy()
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df[f'{price_col}_RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_bollinger_bands(df: pd.DataFrame, price_col: str = 'Close Price', window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """
    Calculate Bollinger Bands
    """
    df = df.copy()
    rolling_mean = df[price_col].rolling(window=window).mean()
    rolling_std = df[price_col].rolling(window=window).std()

    df[f'{price_col}_BB_upper'] = rolling_mean + (rolling_std * num_std)
    df[f'{price_col}_BB_lower'] = rolling_mean - (rolling_std * num_std)
    df[f'{price_col}_BB_middle'] = rolling_mean

    # Calculate Bollinger Band width and %B
    df[f'{price_col}_BB_width'] = df[f'{price_col}_BB_upper'] - df[f'{price_col}_BB_lower']
    df[f'{price_col}_BB_percent'] = (df[price_col] - df[f'{price_col}_BB_lower']) / df[f'{price_col}_BB_width']

    return df


def calculate_macd(df: pd.DataFrame, price_col: str = 'Close Price', fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    """
    df = df.copy()
    exp1 = df[price_col].ewm(span=fast).mean()
    exp2 = df[price_col].ewm(span=slow).mean()
    df[f'{price_col}_MACD'] = exp1 - exp2
    df[f'{price_col}_MACD_signal'] = df[f'{price_col}_MACD'].ewm(span=signal).mean()
    df[f'{price_col}_MACD_histogram'] = df[f'{price_col}_MACD'] - df[f'{price_col}_MACD_signal']

    return df


def add_technical_indicators(df: pd.DataFrame, price_col: str = 'Close Price') -> pd.DataFrame:
    """
    Add technical indicators like moving averages, RSI, Bollinger Bands, MACD
    """
    # Calculate returns
    df = calculate_returns(df, price_col)

    # Calculate volatility
    df = calculate_volatility(df, price_col)

    # Calculate moving averages
    df = calculate_moving_averages(df, price_col)

    # Calculate RSI
    df = calculate_rsi(df, price_col)

    # Calculate Bollinger Bands
    df = calculate_bollinger_bands(df, price_col)

    # Calculate MACD
    df = calculate_macd(df, price_col)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering function for the financial dataset
    """
    df_features = df.copy()

    # Check if Stock Index column exists (indicating original dataset)
    if 'Stock Index' in df_features.columns and df_features['Stock Index'].nunique() > 1:
        # Original dataset with multiple stocks
        # Sort by Stock Index and Date to ensure proper chronological order for each stock
        df_features = df_features.sort_values(['Stock Index', 'Date']).reset_index(drop=True)

        # Engineer features for each stock separately to avoid cross-contamination
        def process_stock_group(group):
            # Calculate returns
            group = calculate_returns(group)
            # Calculate volatility based on returns
            group = calculate_volatility(group)
            # Calculate moving averages
            group = calculate_moving_averages(group)
            # Calculate RSI
            group = calculate_rsi(group)
            # Calculate Bollinger Bands
            group = calculate_bollinger_bands(group)
            # Calculate MACD
            group = calculate_macd(group)
            return group

        # Apply the processing function to each stock group
        df_features = df_features.groupby('Stock Index', group_keys=False).apply(process_stock_group)

        # Engineer features for economic indicators (these are common across all stocks)
        economic_cols = [
            'GDP Growth (%)', 'Inflation Rate (%)', 'Unemployment Rate (%)',
            'Interest Rate (%)', 'Consumer Confidence Index', 'Government Debt (Billion USD)',
            'Corporate Profits (Billion USD)', 'Forex USD/EUR', 'Forex USD/JPY',
            'Crude Oil Price (USD per Barrel)', 'Gold Price (USD per Ounce)',
            'Real Estate Index', 'Retail Sales (Billion USD)', 'Bankruptcy Rate (%)',
            'Mergers & Acquisitions Deals', 'Venture Capital Funding (Billion USD)',
            'Consumer Spending (Billion USD)'
        ]

        # Add rolling statistics for economic indicators
        for col in economic_cols:
            if col in df_features.columns:
                df_features[f'{col}_MA_5'] = df_features[col].rolling(window=5).mean()
                df_features[f'{col}_MA_10'] = df_features[col].rolling(window=10).mean()
                df_features[f'{col}_change'] = df_features[col].pct_change()
                df_features[f'{col}_volatility'] = df_features[col].rolling(window=10).std()

        # Create lagged features for time series prediction (stock-specific)
        price_cols = ['Open Price', 'Close Price', 'Daily High', 'Daily Low']
        for col in price_cols:
            if col in df_features.columns:
                # Add lagged versions as features for prediction (grouped by stock to avoid data leakage between stocks)
                df_features[f'{col}_lag1'] = df_features.groupby('Stock Index')[col].shift(1)
                df_features[f'{col}_lag2'] = df_features.groupby('Stock Index')[col].shift(2)
                df_features[f'{col}_lag3'] = df_features.groupby('Stock Index')[col].shift(3)
                df_features[f'{col}_lag7'] = df_features.groupby('Stock Index')[col].shift(7)

        # Create technical ratios
        if all(col in df_features.columns for col in ['Close Price', 'Open Price', 'Daily High', 'Daily Low']):
            df_features['HL_ratio'] = (df_features['Daily High'] - df_features['Daily Low']) / df_features['Open Price']
            df_features['OC_ratio'] = (df_features['Close Price'] - df_features['Open Price']) / df_features['Open Price']
            df_features['body_size'] = abs(df_features['Close Price'] - df_features['Open Price'])
            df_features['shadow_upper'] = df_features['Daily High'] - df_features[['Close Price', 'Open Price']].max(axis=1)
            df_features['shadow_lower'] = df_features[['Close Price', 'Open Price']].min(axis=1) - df_features['Daily Low']
    else:
        # Bitcoin dataset - single asset
        df_features = df_features.sort_values(['Date']).reset_index(drop=True)

        # Calculate returns
        df_features = calculate_returns(df_features)
        # Calculate volatility based on returns
        df_features = calculate_volatility(df_features)
        # Calculate moving averages
        df_features = calculate_moving_averages(df_features)
        # Calculate RSI
        df_features = calculate_rsi(df_features)
        # Calculate Bollinger Bands
        df_features = calculate_bollinger_bands(df_features)
        # Calculate MACD
        df_features = calculate_macd(df_features)

        # Create lagged features for time series prediction
        price_cols = ['Open Price', 'Close Price', 'Daily High', 'Daily Low']
        for col in price_cols:
            if col in df_features.columns:
                df_features[f'{col}_lag1'] = df_features[col].shift(1)
                df_features[f'{col}_lag2'] = df_features[col].shift(2)
                df_features[f'{col}_lag3'] = df_features[col].shift(3)
                df_features[f'{col}_lag7'] = df_features[col].shift(7)

        # Create technical ratios
        if all(col in df_features.columns for col in ['Close Price', 'Open Price', 'Daily High', 'Daily Low']):
            df_features['HL_ratio'] = (df_features['Daily High'] - df_features['Daily Low']) / df_features['Open Price']
            df_features['OC_ratio'] = (df_features['Close Price'] - df_features['Open Price']) / df_features['Open Price']
            df_features['body_size'] = abs(df_features['Close Price'] - df_features['Open Price'])
            df_features['shadow_upper'] = df_features['Daily High'] - df_features[['Close Price', 'Open Price']].max(axis=1)
            df_features['shadow_lower'] = df_features[['Close Price', 'Open Price']].min(axis=1) - df_features['Daily Low']

        # Add volume features if available
        if 'Trading Volume' in df_features.columns:
            df_features['Trading Volume_MA_7'] = df_features['Trading Volume'].rolling(window=7).mean()
            df_features['Trading Volume_MA_30'] = df_features['Trading Volume'].rolling(window=30).mean()
            df_features['Volume_to_MA_ratio'] = df_features['Trading Volume'] / df_features['Trading Volume_MA_7']

    return df_features