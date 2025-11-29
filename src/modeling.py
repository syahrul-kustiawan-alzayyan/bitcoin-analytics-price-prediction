"""
Module for financial forecasting models
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
except ImportError:
    ARIMA = None

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except ImportError:
    Sequential = None


def prepare_data_for_prediction(df: pd.DataFrame, target_col: str = 'Close Price',
                               feature_cols: list = None, forecast_days: int = 1):
    """
    Prepare data for prediction by creating features and targets
    """
    df_clean = df.copy()

    # Define default feature columns if not provided
    if feature_cols is None:
        feature_cols = [col for col in df_clean.columns if col not in [
            'Date', 'Stock Index', target_col, f'{target_col}_return', f'{target_col}_log_return'
        ] and not col.startswith('Date') and not col.startswith('Stock')]

    # Remove any columns that might have NaN or infinite values
    feature_cols = [col for col in feature_cols if col in df_clean.columns and df_clean[col].notna().all()]

    # Create the feature matrix X and target vector y
    X = df_clean[feature_cols].fillna(method='ffill').fillna(method='bfill')
    y = df_clean[target_col]

    # Ensure no infinite or NaN values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

    return X, y, feature_cols


def arima_forecast(data: pd.Series, order: tuple = (1, 1, 1), forecast_steps: int = 30):
    """
    ARIMA forecasting model
    """
    if ARIMA is None:
        raise ImportError("statsmodels is required for ARIMA forecasting")

    # Fit ARIMA model
    model = ARIMA(data, order=order)
    fitted_model = model.fit()

    # Forecast
    forecast = fitted_model.forecast(steps=forecast_steps)
    confidence_intervals = fitted_model.get_forecast(steps=forecast_steps).conf_int()

    return forecast, confidence_intervals, fitted_model


def prophet_forecast(df: pd.DataFrame, target_col: str = 'Close Price', forecast_periods: int = 30):
    """
    Prophet forecasting model
    """
    if Prophet is None:
        raise ImportError("prophet is required for Prophet forecasting")

    # Prepare data for Prophet (needs 'ds' and 'y' columns)
    prophet_df = df[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'})

    # Create and fit Prophet model
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_periods)

    # Make predictions
    forecast = model.predict(future)

    return forecast, model


def create_lstm_data(data, time_steps=60):
    """
    Create dataset for LSTM model
    """
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def lstm_forecast(data: pd.Series, time_steps: int = 60, forecast_steps: int = 30):
    """
    LSTM forecasting model
    """
    if Sequential is None:
        raise ImportError("tensorflow is required for LSTM forecasting")

    # Normalize the data
    data_norm = (data - data.min()) / (data.max() - data.min())
    data_norm = data_norm.values.astype('float32')

    # Prepare the data for LSTM
    X, y = create_lstm_data(data_norm, time_steps)

    # Reshape X for LSTM [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split the data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=0)

    # Forecast future values
    last_sequence = X[-1]  # Get the last sequence from the data
    predictions = []

    current_sequence = last_sequence.copy()

    for _ in range(forecast_steps):
        # Reshape for prediction
        current_input = current_sequence.reshape((1, time_steps, 1))
        # Predict next value
        next_pred = model.predict(current_input, verbose=0)[0, 0]
        # Add to predictions
        predictions.append(next_pred)
        # Update sequence (remove first element and add prediction)
        current_sequence = np.append(current_sequence[1:], next_pred)

    # Denormalize predictions
    predictions = np.array(predictions) * (data.max() - data.min()) + data.min()

    return predictions, model


def random_forest_forecast(df: pd.DataFrame, target_col: str = 'Close Price',
                          feature_cols: list = None, forecast_days: int = 1):
    """
    Random Forest forecasting model
    """
    # Prepare data
    X, y, used_features = prepare_data_for_prediction(df, target_col, feature_cols, forecast_days)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Train on full dataset for forecasting
    rf_model_full = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_full.fit(X, y)

    return {
        'model': rf_model_full,
        'predictions': y_pred,
        'actual': y_test,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'features': used_features
    }


def linear_regression_forecast(df: pd.DataFrame, target_col: str = 'Close Price',
                              feature_cols: list = None, forecast_days: int = 1):
    """
    Linear Regression forecasting model
    """
    # Prepare data
    X, y, used_features = prepare_data_for_prediction(df, target_col, feature_cols, forecast_days)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Make predictions
    y_pred = lr_model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Train on full dataset for forecasting
    lr_model_full = LinearRegression()
    lr_model_full.fit(X, y)

    return {
        'model': lr_model_full,
        'predictions': y_pred,
        'actual': y_test,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'features': used_features
    }


def evaluate_model_performance(y_true, y_pred):
    """
    Evaluate model performance using common metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }