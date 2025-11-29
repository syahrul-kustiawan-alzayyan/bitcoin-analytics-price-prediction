"""
Module for financial data visualization
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


def plot_candlestick(df: pd.DataFrame, date_col: str = 'Date', open_col: str = 'Open Price',
                     high_col: str = 'Daily High', low_col: str = 'Daily Low',
                     close_col: str = 'Close Price', stock: str = None):
    """
    Create candlestick chart using Plotly
    """
    if stock:
        df = df[df['Stock Index'] == stock]

    fig = go.Figure(data=go.Candlestick(
        x=df[date_col],
        open=df[open_col],
        high=df[high_col],
        low=df[low_col],
        close=df[close_col]
    ))

    fig.update_layout(
        title=f'Candlestick Chart for {stock if stock else "All Stocks"}',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    fig.show()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, cols: list = None, method: str = 'pearson',
                             figsize: tuple = (15, 12), annot: bool = True):
    """
    Create correlation heatmap for selected columns
    """
    if cols is None:
        # Select numeric columns that are likely to be meaningful for correlation
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude columns with 'lag' to avoid too many features
        cols = [col for col in cols if 'lag' not in col.lower()][:20]  # Limit to first 20 numeric columns

    # Calculate correlation matrix
    corr_matrix = df[cols].corr(method=method)

    # Create the heatmap
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    heatmap = sns.heatmap(corr_matrix, mask=mask, annot=annot, cmap='coolwarm', center=0,
                          square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Heatmap of Financial Features')
    plt.tight_layout()
    plt.show()

    return corr_matrix


def plot_time_series(df: pd.DataFrame, columns: list, date_col: str = 'Date',
                     stock: str = None, figsize: tuple = (14, 8)):
    """
    Plot time series data
    """
    if stock:
        df = df[df['Stock Index'] == stock]

    plt.figure(figsize=figsize)

    for col in columns:
        if col in df.columns:
            plt.plot(df[date_col], df[col], label=col, linewidth=1)

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'Time Series Plot for {stock if stock else "All Stocks"}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_stock_comparison(df: pd.DataFrame, price_col: str = 'Close Price',
                          figsize: tuple = (14, 8)):
    """
    Plot closing prices for different stocks
    """
    unique_stocks = df['Stock Index'].unique()

    plt.figure(figsize=figsize)
    for stock in unique_stocks:
        stock_data = df[df['Stock Index'] == stock]
        plt.plot(stock_data['Date'], stock_data[price_col], label=stock, alpha=0.7)

    plt.xlabel('Date')
    plt.ylabel(price_col)
    plt.title('Stock Price Comparison')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_returns_distribution(df: pd.DataFrame, return_col: str = 'Close Price_return',
                              figsize: tuple = (10, 6)):
    """
    Plot distribution of returns
    """
    plt.figure(figsize=figsize)

    # Filter out NaN values
    returns = df[return_col].dropna()

    plt.hist(returns, bins=50, density=True, alpha=0.7, edgecolor='black')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.title(f'Distribution of {return_col}')
    plt.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_technical_indicators(df: pd.DataFrame, price_col: str = 'Close Price',
                              stock: str = None, figsize: tuple = (15, 12)):
    """
    Plot technical indicators together with price
    """
    if stock:
        df = df[df['Stock Index'] == stock]

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Plot 1: Price and moving averages
    axes[0].plot(df['Date'], df[price_col], label=price_col, linewidth=2)
    ma_cols = [col for col in df.columns if f'{price_col}_MA_' in col]
    for ma_col in ma_cols:
        axes[0].plot(df['Date'], df[ma_col], label=ma_col, alpha=0.7)
    axes[0].set_title(f'{price_col} and Moving Averages')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Trading volume
    if 'Trading Volume' in df.columns:
        axes[1].bar(df['Date'], df['Trading Volume'], alpha=0.6, width=1)
        axes[1].set_title('Trading Volume')
        axes[1].grid(True)

    # Plot 3: RSI
    rsi_col = f'{price_col}_RSI'
    if rsi_col in df.columns:
        axes[2].plot(df['Date'], df[rsi_col], label=rsi_col, color='purple')
        axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        axes[2].set_title('RSI (Relative Strength Index)')
        axes[2].set_ylabel('RSI')
        axes[2].legend()
        axes[2].grid(True)

    # Plot 4: Volatility
    vol_col = f'{price_col}_volatility'
    if vol_col in df.columns:
        axes[3].plot(df['Date'], df[vol_col], label=vol_col, color='orange')
        axes[3].set_title('Volatility')
        axes[3].set_ylabel('Volatility')
        axes[3].set_xlabel('Date')
        axes[3].grid(True)

    plt.tight_layout()
    plt.show()


def plot_forecast_results(actual, predicted, forecast=None,
                          dates=None, title="Forecast Results"):
    """
    Plot actual vs predicted values with optional future forecast
    """
    plt.figure(figsize=(14, 7))

    # Convert to pandas Series if they're not already (in case they're numpy arrays)
    if not isinstance(actual, pd.Series):
        actual = pd.Series(actual) if not isinstance(actual, np.ndarray) else pd.Series(actual)
    if not isinstance(predicted, pd.Series):
        predicted = pd.Series(predicted) if not isinstance(predicted, np.ndarray) else pd.Series(predicted)

    if dates is not None:
        plt.plot(dates[:len(actual)], actual, label='Actual', linewidth=2)
        plt.plot(dates[:len(predicted)], predicted, label='Predicted', linewidth=2)

        if forecast is not None:
            # Calculate future dates for forecast
            last_date = dates.iloc[-len(forecast)] if len(forecast) <= len(dates) else dates.iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast), freq='D')
            plt.plot(future_dates, forecast, label='Future Forecast', linestyle='--', linewidth=2)
    else:
        # Handle both Series and numpy arrays
        actual_values = actual.values if hasattr(actual, 'values') else actual
        predicted_values = predicted.values if hasattr(predicted, 'values') else predicted

        plt.plot(actual_values, label='Actual', linewidth=2)
        plt.plot(predicted_values, label='Predicted', linewidth=2)

        if forecast is not None:
            plt.plot(range(len(actual), len(actual) + len(forecast)), forecast,
                     label='Future Forecast', linestyle='--', linewidth=2)

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_interactive_dashboard(df: pd.DataFrame):
    """
    Create an interactive dashboard using Plotly
    """
    # Prepare data for the dashboard
    unique_stocks = df['Stock Index'].unique()

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Trends', 'Trading Volume', 'Returns Distribution', 'Correlation'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Plot 1: Price trends for first stock
    if len(unique_stocks) > 0:
        first_stock = unique_stocks[0]
        stock_data = df[df['Stock Index'] == first_stock]
        fig.add_trace(
            go.Scatter(x=stock_data['Date'], y=stock_data['Close Price'],
                      name=f'{first_stock} Close Price', line=dict(color='blue')),
            row=1, col=1
        )

    # Plot 2: Trading volume
    if 'Trading Volume' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Trading Volume'],
                      name='Trading Volume', line=dict(color='green')),
            row=1, col=2
        )

    # Plot 3: Returns distribution
    return_col = 'Close Price_return'
    if return_col in df.columns:
        returns = df[return_col].dropna()
        fig.add_trace(
            go.Histogram(x=returns, name='Returns Distribution', marker_color='orange'),
            row=2, col=1
        )

    # Plot 4: Simple correlation between close price and volume
    if 'Trading Volume' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Trading Volume'], y=df['Close Price'],
                      mode='markers', name='Price vs Volume', marker=dict(size=5, opacity=0.6)),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(height=800, showlegend=True, title_text="Financial Data Dashboard")

    return fig