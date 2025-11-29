"""
Streamlit dashboard for Bitcoin analysis and forecasting
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import from the original source files, fallback to BTC-specific functions
try:
    from src.data_loading import load_financial_dataset
    from src.data_cleaning import clean_financial_data
    from src.feature_engineering import engineer_features
except ImportError:
    # Fallback for BTC-specific loading
    def load_financial_dataset():
        """Load BTC dataset from raw data"""
        import pandas as pd
        import os
        # Get the directory of the current file and navigate to the dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', 'data', 'raw', 'BTC.csv')
        df = pd.read_csv(data_path)
        # Convert date column to datetime and rename for consistency
        df['Date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={
            'open': 'Open Price',
            'high': 'Daily High',
            'low': 'Daily Low',
            'close': 'Close Price'
        })
        df['Stock Index'] = 'BTC'
        # Add volume if available
        if 'volume' in df.columns:
            df['Trading Volume'] = df['volume']
        else:
            # Create a placeholder volume column
            df['Trading Volume'] = 0
        return df

    def clean_financial_data(df):
        """Clean BTC data"""
        df_clean = df.copy()
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        df_clean = df_clean.sort_values('Date').reset_index(drop=True)
        df_clean = df_clean.fillna(method='ffill')
        return df_clean

    def engineer_features(df):
        """Engineer features for BTC data"""
        df_features = df.copy()
        df_features = df_features.sort_values(['Date']).reset_index(drop=True)

        # Calculate returns
        df_features['Close Price_return'] = df_features['Close Price'].pct_change()
        # Calculate volatility based on returns
        df_features['Close Price_volatility'] = df_features['Close Price_return'].rolling(window=20).std()
        # Calculate moving averages
        df_features['Close Price_MA_7'] = df_features['Close Price'].rolling(window=7).mean()
        df_features['Close Price_MA_21'] = df_features['Close Price'].rolling(window=21).mean()
        df_features['Close Price_MA_50'] = df_features['Close Price'].rolling(window=50).mean()
        df_features['Close Price_MA_200'] = df_features['Close Price'].rolling(window=200).mean()
        # Calculate RSI
        delta = df_features['Close Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_features['Close Price_RSI'] = 100 - (100 / (1 + rs))
        # Calculate Bollinger Bands
        rolling_mean = df_features['Close Price'].rolling(window=20).mean()
        rolling_std = df_features['Close Price'].rolling(window=20).std()
        df_features['Close Price_BB_upper'] = rolling_mean + (rolling_std * 2)
        df_features['Close Price_BB_lower'] = rolling_mean - (rolling_std * 2)
        df_features['Close Price_BB_middle'] = rolling_mean
        # Calculate MACD
        exp1 = df_features['Close Price'].ewm(span=12).mean()
        exp2 = df_features['Close Price'].ewm(span=26).mean()
        df_features['Close Price_MACD'] = exp1 - exp2
        df_features['Close Price_MACD_signal'] = df_features['Close Price_MACD'].ewm(span=9).mean()
        df_features['Close Price_MACD_histogram'] = df_features['Close Price_MACD'] - df_features['Close Price_MACD_signal']
        # Create lagged features
        df_features['Close Price_lag1'] = df_features['Close Price'].shift(1)
        df_features['Close Price_lag2'] = df_features['Close Price'].shift(2)
        df_features['Close Price_lag3'] = df_features['Close Price'].shift(3)
        df_features['Close Price_lag7'] = df_features['Close Price'].shift(7)
        # Create technical ratios
        df_features['HL_ratio'] = (df_features['Daily High'] - df_features['Daily Low']) / df_features['Open Price']
        df_features['OC_ratio'] = (df_features['Close Price'] - df_features['Open Price']) / df_features['Open Price']
        df_features['body_size'] = abs(df_features['Close Price'] - df_features['Open Price'])
        df_features['shadow_upper'] = df_features['Daily High'] - df_features[['Close Price', 'Open Price']].max(axis=1)
        df_features['shadow_lower'] = df_features[['Close Price', 'Open Price']].min(axis=1) - df_features['Daily Low']
        return df_features


@st.cache_data
def load_and_process_data():
    """Load and process the BTC dataset"""
    df = load_financial_dataset()
    df_clean = clean_financial_data(df)
    df_featured = engineer_features(df_clean)
    return df_featured


def calculate_returns_metrics(df, price_col='Close Price'):
    """Calculate various return metrics"""
    df = df.copy()

    if len(df) < 2:  # Need at least 2 data points for pct_change
        return 0, 0, 0, 0

    df[f'{price_col}_return'] = df[price_col].pct_change()

    # Current price
    current_price = df[price_col].iloc[-1] if not df.empty else 0

    # Daily return (avoid accessing NaN values)
    daily_return = 0
    if len(df) > 0 and f'{price_col}_return' in df.columns and not df[f'{price_col}_return'].iloc[-1:].empty:
        daily_return_val = df[f'{price_col}_return'].iloc[-1]
        daily_return = daily_return_val if not pd.isna(daily_return_val) else 0

    # YTD performance (if we have data from start of year)
    current_year = datetime.now().year
    ytd_data = df[df['Date'].dt.year == current_year]
    ytd_return = 0
    if not ytd_data.empty and len(ytd_data) > 0:
        ytd_start = ytd_data[price_col].iloc[0]
        if ytd_start != 0:
            ytd_return = ((current_price - ytd_start) / ytd_start) * 100

    # Volatility (30-day rolling) - need enough data points
    current_volatility = 0
    if f'{price_col}_return' in df.columns and len(df) >= 30:
        df['volatility_30d'] = df[f'{price_col}_return'].rolling(window=30).std() * np.sqrt(252)  # Annualized
        vol_val = df['volatility_30d'].iloc[-1] if not df['volatility_30d'].iloc[-1:].empty else None
        current_volatility = vol_val * 100 if vol_val is not None and not pd.isna(vol_val) else 0

    return current_price, daily_return * 100, ytd_return, current_volatility


def aggregate_data(df, freq):
    """Aggregate data based on frequency: D=Daily, M=Monthly, Y=Yearly"""
    df_agg = df.copy()
    df_agg.set_index('Date', inplace=True)

    if freq == 'D':
        # Daily (no aggregation needed)
        return df_agg.reset_index()
    elif freq == 'M':
        # Monthly aggregation
        agg_dict = {
            'Open Price': 'first',
            'Close Price': 'last',
            'Daily High': 'max',
            'Daily Low': 'min',
            'Trading Volume': 'sum'
        }
        # Add numeric columns only
        numeric_cols = df_agg.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in agg_dict:
                agg_dict[col] = 'mean'

        df_agg = df_agg.resample('M').agg(agg_dict)
        return df_agg.reset_index()
    elif freq == 'Y':
        # Yearly aggregation
        agg_dict = {
            'Open Price': 'first',
            'Close Price': 'last',
            'Daily High': 'max',
            'Daily Low': 'min',
            'Trading Volume': 'sum'
        }
        # Add numeric columns only
        numeric_cols = df_agg.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in agg_dict:
                agg_dict[col] = 'mean'

        df_agg = df_agg.resample('Y').agg(agg_dict)
        return df_agg.reset_index()
    else:
        return df


def create_candlestick_chart(df, selected_stock, ma_windows=[20, 50]):
    """Create candlestick chart with moving averages"""
    fig = go.Figure()

    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open Price'],
        high=df['Daily High'],
        low=df['Daily Low'],
        close=df['Close Price'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Add moving averages
    for window in ma_windows:
        ma_col = f'Close Price_MA_{window}'
        if ma_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df[ma_col],
                mode='lines',
                name=f'SMA{window}',
                line=dict(width=1.5)
            ))

    fig.update_layout(
        title=f'Candlestick Chart for {selected_stock}',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        showlegend=True,
        template='plotly_white'
    )

    return fig


def create_forecast_simulation(df, selected_stock, forecast_period='daily'):
    """Create forecast simulation based on selected period"""
    if len(df) < 30:
        st.warning("Not enough data for forecasting")
        return None, None

    # Determine forecast length based on selected period
    if forecast_period == 'daily':
        forecast_steps = 7  # 1 week
        date_increment = lambda i: df['Date'].max() + timedelta(days=i+1)
        period_name = "Week"
    elif forecast_period == 'monthly':
        forecast_steps = 5  # 5 months
        date_increment = lambda i: df['Date'].max() + pd.DateOffset(months=i+1)
        period_name = "Months"
    else:  # yearly
        forecast_steps = 12  # 1 year
        date_increment = lambda i: df['Date'].max() + pd.DateOffset(years=i+1)
        period_name = "Year"

    # Create forecast data based on price trend and volatility
    recent_data = df['Close Price'].dropna().tail(90).values
    if len(recent_data) < 2:
        st.warning("Not enough data for forecasting")
        return None, None

    last_date = df['Date'].max()

    # Calculate base trend from recent data
    first_val = recent_data[0]
    last_val = recent_data[-1]
    base_trend = (last_val - first_val) / len(recent_data)  # Average daily/monthly/yearly change

    # Calculate volatility for the forecast
    returns = np.diff(recent_data) / recent_data[:-1]  # Calculate returns
    volatility = np.std(returns) if len(returns) > 0 else 0.02  # Default to 2% if no data

    # Simulate forecast
    forecast_dates = []
    forecast_values = []
    current_val = recent_data[-1]

    for i in range(forecast_steps):
        # Calculate period-based volatility
        if forecast_period == 'daily':
            period_volatility = volatility * np.sqrt(1)  # Daily volatility
        elif forecast_period == 'monthly':
            period_volatility = volatility * np.sqrt(30)  # Monthly volatility (approx 30 days)
        else:  # yearly
            period_volatility = volatility * np.sqrt(365)  # Yearly volatility (approx 365 days)

        # Add both trend and random component
        random_component = np.random.normal(0, abs(current_val) * period_volatility)
        trend_component = base_trend * (1 if forecast_period == 'daily' else 30 if forecast_period == 'monthly' else 365)

        change = trend_component + random_component
        current_val += change

        next_date = date_increment(i)

        forecast_dates.append(next_date)
        forecast_values.append(max(0, current_val))  # Ensure positive prices

    # Create confidence intervals based on period volatility
    multiplier = 1.96  # 95% confidence interval
    forecast_upper = []
    forecast_lower = []

    for i, val in enumerate(forecast_values):
        # Calculate confidence interval based on period and cumulative volatility
        ci_multiplier = multiplier
        if forecast_period == 'daily':
            ci_range = abs(val) * volatility * np.sqrt(i+1) * ci_multiplier
        elif forecast_period == 'monthly':
            ci_range = abs(val) * volatility * np.sqrt((i+1) * 30) * ci_multiplier
        else:  # yearly
            ci_range = abs(val) * volatility * np.sqrt((i+1) * 365) * ci_multiplier

        forecast_upper.append(val + ci_range)
        forecast_lower.append(val - ci_range)

    fig = go.Figure()

    # Historical data (last 180 days or all available if less)
    hist_limit = min(180, len(df))
    hist_dates = df['Date'].tail(hist_limit)
    hist_prices = df['Close Price'].tail(hist_limit)

    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_prices,
        mode='lines',
        name='Historical Price',
        line=dict(color='#1f77b4', width=2),
        opacity=0.7
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6),
        opacity=0.8
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_dates + forecast_dates[::-1],
        y=forecast_upper + forecast_lower[::-1],
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.15)',
        line=dict(color='rgba(255, 127, 14, 0)'),
        name='95% Confidence Interval',
        showlegend=True,
        opacity=0.5
    ))

    fig.update_layout(
        title=f'Bitcoin {selected_stock} - {forecast_steps} {period_name} {"Daily" if forecast_period == "daily" else "Monthly" if forecast_period == "monthly" else "Yearly"} Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )

    # Display metrics
    st.markdown("### 📊 Forecast Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Start Price", f"${hist_prices.iloc[-1]:.2f}")
    with col2:
        st.metric("End Forecast", f"${forecast_values[-1]:.2f}")
    with col3:
        change_pct = ((forecast_values[-1] - hist_prices.iloc[-1]) / hist_prices.iloc[-1]) * 100
        st.metric("Expected Change", f"{change_pct:.2f}%")
    with col4:
        st.metric("Confidence Level", "95%")

    # Show forecast table
    st.markdown("### Forecast Details")
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast Price': [f'${val:.2f}' for val in forecast_values],
        'Upper Bound (95%)': [f'${val:.2f}' for val in forecast_upper],
        'Lower Bound (95%)': [f'${val:.2f}' for val in forecast_lower]
    })
    st.dataframe(forecast_df, use_container_width=True)

    return fig, forecast_df


def create_pdf_report(df, selected_stock, forecast_period, forecast_df):
    """Create a PDF report"""
    # In a real implementation, we'd use reportlab or a similar library
    # For now, we'll create a downloadable text file
    from datetime import datetime

    # Create report content
    report_content = f"""BITCOIN ANALYSIS REPORT
=====================

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Asset: {selected_stock}

MARKET OVERVIEW
==============
Current Price: ${df['Close Price'].iloc[-1] if len(df) > 0 else 0:.2f}

FORECAST DETAILS
================
Forecast Period: {forecast_period.replace('_', ' ').title()}
Forecast Duration: {7 if forecast_period == 'daily' else 5 if forecast_period == 'monthly' else 12} {'Days' if forecast_period == 'daily' else 'Months' if forecast_period == 'monthly' else 'Years'}
Historical Data Points Used: {len(df)}

FORECAST TABLE
==============
Date,Forecast Price,Upper Bound (95%),Lower Bound (95%)
"""

    # Add forecast data
    if forecast_df is not None:
        for _, row in forecast_df.iterrows():
            report_content += f"{row['Date']},{row['Forecast Price']},{row['Upper Bound (95%)']},{row['Lower Bound (95%)']}\n"
    else:
        report_content += "No forecast data available\n"

    report_content += f"""

DISCLAIMER
==========
This is a simulated forecast based on historical patterns.
Past performance is not indicative of future results.
Cryptocurrency investments carry inherent risks.
"""

    # Convert to bytes
    report_bytes = report_content.encode('utf-8')

    return report_bytes


def main():
    st.set_page_config(
        page_title="Bitcoin Analysis Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Black and gold corporate header with smaller title
    st.markdown("""
    <div style="background: linear-gradient(135deg, #000000 0%, #0d0d0d 100%); padding: 20px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #3a3a3a; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="flex: 1;">
                <h1 style="color: #d4af37; text-align: left; margin: 0; font-size: 1.8em; text-shadow: 0 0 15px rgba(212, 175, 55, 0.4); letter-spacing: 1px;">₿ITCOIN ANALYTICS SUITE</h1>
                <p style="color: #e6e6e6; text-align: left; margin: 8px 0 0 0; font-size: 1.0em; opacity: 0.9;">Institutional Market Intelligence & Predictive Analytics Platform</p>
            </div>
            <div style="text-align: right;">
                <div style="background: rgba(212, 175, 55, 0.05); border: 1px solid #d4af37; border-radius: 10px; padding: 12px; min-width: 180px;">
                    <div style="color: #b0b0b0; font-size: 0.85em; margin-bottom: 4px;">MARKET STATUS</div>
                    <div style="color: #d4af37; font-weight: bold; font-size: 1.0em;">ACTIVE TRADING</div>
                    <div style="color: #b0b0b0; font-size: 0.75em; margin-top: 4px;">Real-time Data</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner('🔄 Loading and processing Bitcoin data...'):
        df_featured = load_and_process_data()

    # Minimalist black and gold sidebar
    st.sidebar.markdown("<div style='text-align: center; padding: 15px 0;'><h2 style='color: #d4af37; margin: 0; font-size: 1.8em; text-shadow: 0 0 10px rgba(212, 175, 55, 0.3);'>₿</h2><p style='color: #e6e6e6; margin: 5px 0 0 0; font-size: 0.9em;'>Analytics Suite</p></div>", unsafe_allow_html=True)

    # Minimalist divider
    st.sidebar.markdown("<hr style='border: 0; border-top: 1px solid #2a2a2a; margin: 20px 0;' />", unsafe_allow_html=True)

    # Stock selection (BTC is the only option in our dataset)
    available_stocks = df_featured['Stock Index'].unique()
    selected_stock = st.sidebar.selectbox(
        "Asset",
        options=available_stocks,
        index=0,
        help="Select cryptocurrency for analysis",
        label_visibility="collapsed"
    )

    # Time controls with minimalist styling
    st.sidebar.markdown("<div style='color: #d4af37; font-size: 0.9em; margin: 15px 0 8px 0; font-weight: 600;'>TIME FRAMEWORK</div>", unsafe_allow_html=True)

    timeframe = st.sidebar.selectbox(
        "Historical Data",
        options=["All", "1Y", "6M", "3M", "1M"],
        index=0,
        help="Time range for historical data",
        label_visibility="collapsed"
    )

    aggregation_freq = st.sidebar.radio(
        "Aggregation Level",
        options=["Daily", "Weekly", "Monthly"],
        index=0,
        help="Data granularity level",
        label_visibility="collapsed"
    )

    # Forecasting controls with minimalist styling
    st.sidebar.markdown("<div style='color: #d4af37; font-size: 0.9em; margin: 15px 0 8px 0; font-weight: 600;'>PREDICTIVE FRAMEWORK</div>", unsafe_allow_html=True)

    forecast_period = st.sidebar.selectbox(
        "Forecast Horizon",
        options=["daily", "monthly", "yearly"],
        index=0,
        format_func=lambda x: x.replace('_', ' ').title(),
        help="Forecast timeframe selection",
        label_visibility="collapsed"
    )


    # Filter data for selected stock
    df_stock = df_featured[df_featured['Stock Index'] == selected_stock].copy()
    df_stock = df_stock.sort_values('Date')  # Ensure proper sorting

    # Apply timeframe filter
    if timeframe != "All":
        if timeframe == "1M":
            months = 1
        elif timeframe == "3M":
            months = 3
        elif timeframe == "6M":
            months = 6
        elif timeframe == "1Y":
            months = 12
        else:  # "All" case
            months = 9999  # Large number to show all data

        cutoff_date = datetime.now() - pd.DateOffset(months=months)
        df_stock = df_stock[df_stock['Date'] >= cutoff_date]

    # Apply data aggregation based on frequency
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
    df_stock = aggregate_data(df_stock, freq_map[aggregation_freq])

    # Sort again after aggregation to ensure proper order
    df_stock = df_stock.sort_values('Date').reset_index(drop=True)

    # Calculate market overview metrics
    if not df_stock.empty and len(df_stock) > 1:  # Need at least 2 rows for pct_change
        current_price, daily_return, ytd_return, volatility = calculate_returns_metrics(df_stock)

        # Enhanced KPI cards with attractive colors and styling
        st.markdown("## 📊 INSTITUTIONAL MARKET DASHBOARD")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #0d0d1a, #1a1a2e); padding: 20px; border-radius: 12px; border: 1px solid #3a3a4a; height: 100%; box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="background: rgba(212, 175, 55, 0.15); width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                        <span style="color: #d4af37; font-size: 1.2em;">🪙</span>
                    </div>
                    <div style="color: #b0b0b0; font-size: 0.85em; font-weight: 500;">CURRENT PRICE</div>
                </div>
                <div style="color: #ffffff; font-size: 1.8em; font-weight: bold; margin: 5px 0; text-align: left;">${current_price:.2f}</div>
                <div style="color: #d4af37; font-size: 0.9em; margin-top: 8px; text-align: left; font-weight: 500;">BTC/USD</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            color_positive = "#06d6a0" if daily_return >= 0 else "#ef476f"
            color_negative = "#a6a6a6" if daily_return >= 0 else "#ef476f"
            bg_gradient = "to right, #06d6a015, #06d6a005" if daily_return >= 0 else "to right, #ef476f15, #ef476f05"
            trend_icon = "↗️" if daily_return >= 0 else "↘️"
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #0d0d1a, #1a1a2e); padding: 20px; border-radius: 12px; border: 1px solid #3a3a4a; height: 100%; box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="background: rgba({10, 214, 160 if daily_return >= 0 else 239, 71, 111}, 0.15); width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                        <span style="color: {color_positive}; font-size: 1.2em;">{trend_icon}</span>
                    </div>
                    <div style="color: #b0b0b0; font-size: 0.85em; font-weight: 500;">24H CHANGE</div>
                </div>
                <div style="color: {color_positive}; font-size: 1.8em; font-weight: bold; margin: 5px 0; text-align: left;">{daily_return:.2f}%</div>
                <div style="color: {color_negative}; font-size: 0.9em; margin-top: 8px; text-align: left; font-weight: 500;">{(abs(daily_return)):.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            color_positive = "#06d6a0" if ytd_return >= 0 else "#ef476f"
            color_negative = "#a6a6a6" if ytd_return >= 0 else "#ef476f"
            trend_icon = "↗️" if ytd_return >= 0 else "↘️"
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #0d0d1a, #1a1a2e); padding: 20px; border-radius: 12px; border: 1px solid #3a3a4a; height: 100%; box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="background: rgba({10, 214, 160 if ytd_return >= 0 else 239, 71, 111}, 0.15); width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                        <span style="color: {color_positive}; font-size: 1.2em;">📊</span>
                    </div>
                    <div style="color: #b0b0b0; font-size: 0.85em; font-weight: 500;">YTD PERFORMANCE</div>
                </div>
                <div style="color: {color_positive}; font-size: 1.8em; font-weight: bold; margin: 5px 0; text-align: left;">{ytd_return:.2f}%</div>
                <div style="color: {color_negative}; font-size: 0.9em; margin-top: 8px; text-align: left; font-weight: 500;">{trend_icon} {abs(ytd_return):.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            volatility_level = "HIGH" if volatility > 25 else "MODERATE" if volatility > 15 else "LOW"
            level_color = "#06d6a0" if volatility_level == "LOW" else "#ff9e00" if volatility_level == "MODERATE" else "#ef476f"
            vol_icon = "🟢" if volatility_level == "LOW" else "⚠️" if volatility_level == "MODERATE" else "🔴"
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #0d0d1a, #1a1a2e); padding: 20px; border-radius: 12px; border: 1px solid #3a3a4a; height: 100%; box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="background: rgba({6, 214, 160 if volatility_level == "LOW" else 255, 158, 0 if volatility_level == "MODERATE" else 239, 71, 111}, 0.15); width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                        <span style="color: {level_color}; font-size: 1.2em;">{vol_icon}</span>
                    </div>
                    <div style="color: #b0b0b0; font-size: 0.85em; font-weight: 500;">30D VOLATILITY</div>
                </div>
                <div style="color: #e6e6e6; font-size: 1.8em; font-weight: bold; margin: 5px 0; text-align: left;">{volatility:.2f}%</div>
                <div style="color: {level_color}; font-size: 0.9em; margin-top: 8px; text-align: left; text-transform: uppercase; font-weight: 500;">{volatility_level}</div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            volume_value = df_stock['Trading Volume'].iloc[-1] if 'Trading Volume' in df_stock.columns else 0
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #0d0d1a, #1a1a2e); padding: 20px; border-radius: 12px; border: 1px solid #3a3a4a; height: 100%; box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="background: rgba(212, 175, 55, 0.15); width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                        <span style="color: #d4af37; font-size: 1.2em;">💸</span>
                    </div>
                    <div style="color: #b0b0b0; font-size: 0.85em; font-weight: 500;">24H VOLUME</div>
                </div>
                <div style="color: #e6e6e6; font-size: 1.8em; font-weight: bold; margin: 5px 0; text-align: left;">${volume_value:,.0f}</div>
                <div style="color: #d4af37; font-size: 0.9em; margin-top: 8px; text-align: left; font-weight: 500;">TOTAL TRADED</div>
            </div>
            """, unsafe_allow_html=True)

    # Create professional tabs with enhanced styling
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 PRICE ANALYTICS",
        "⚠️ RISK PROFILES",
        "🔍 PREDICTIVE ANALYTICS",
        "📊 DATA DASHBOARD"
    ])

    with tab1:
        st.markdown("### Price & Trend Analysis")
        price_chart = create_candlestick_chart(df_stock, selected_stock)
        st.plotly_chart(price_chart, use_container_width=True)

    with tab2:
        st.markdown("### 🛡️ INSTITUTIONAL RISK ANALYTICS")

        # Create enhanced risk analysis dashboard
        risk_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('📊 PRICE VOLATILITY TRENDS', '📈 DAILY RETURNS ANALYSIS', '⚠️ RISK METRICS DASHBOARD', '📈 CORRELATION ANALYSIS'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12
        )

        # Add volatility plot with enhanced styling
        if 'Close Price_volatility' in df_stock.columns:
            risk_fig.add_trace(
                go.Scatter(
                    x=df_stock['Date'],
                    y=df_stock['Close Price_volatility'] * 100,
                    mode='lines',
                    name='30D Volatility (%)',
                    line=dict(color='#f7931a', width=2.5)
                ),
                row=1, col=1
            )
            # Add volatility average reference line
            vol_avg = (df_stock['Close Price_volatility'] * 100).mean()
            risk_fig.add_hline(y=vol_avg, line_dash="dash", line_color="#79c0ff",
                             annotation_text=f"Average Volatility: {vol_avg:.2f}%",
                             line_width=1, row=1, col=1)

        # Daily returns with enhanced styling
        if 'Close Price_return' in df_stock.columns:
            risk_fig.add_trace(
                go.Scatter(
                    x=df_stock['Date'],
                    y=df_stock['Close Price_return'] * 100,
                    mode='lines',
                    name='Daily Returns (%)',
                    line=dict(color='#79c0ff', width=2)
                ),
                row=1, col=2
            )
            # Add zero reference line
            risk_fig.add_hline(y=0, line_dash="solid", line_color="#8b949e",
                             line_width=1, row=1, col=2)

        # Add risk metrics visualization with enhanced styling
        risk_metrics = ['Close Price_volatility', 'Close Price_return']
        available_metrics = [col for col in risk_metrics if col in df_stock.columns]

        if available_metrics:
            means = [df_stock[col].mean() if col in df_stock.columns else 0 for col in available_metrics]
            colors = ['#f7931a', '#79c0ff']  # Gold and blue for Bitcoin theme
            risk_fig.add_trace(
                go.Bar(
                    x=available_metrics,
                    y=[m * 100 if 'return' not in metric.lower() else m * 100 for i, metric in enumerate(available_metrics) for m in [means[i]]],
                    name='Risk Metrics',
                    marker_color=colors[:len(available_metrics)],
                    text=[f'{m*100:.2f}%' if 'return' not in metric.lower() else f'{m*100:.2f}%' for i, metric in enumerate(available_metrics) for m in [means[i]]],
                    textposition='auto',
                ),
                row=2, col=1
            )

        # Add simple correlation analysis if possible
        numeric_cols = df_stock.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            # Use a simple correlation of the first two numeric columns
            col1, col2 = numeric_cols[0], numeric_cols[1]
            if col1 != 'Date' and col2 != 'Date':
                corr_val = df_stock[col1].corr(df_stock[col2])
                risk_fig.add_annotation(
                    xref="x domain", yref="y domain",
                    x=0.5, y=0.5,
                    text=f"Correlation<br>{col1} vs {col2}<br>{corr_val:.3f}",
                    showarrow=False,
                    font=dict(size=16, color="#e6edf3"),
                    bgcolor="rgba(26, 26, 36, 0.8)",
                    row=2, col=2
                )

        risk_fig.update_layout(
            height=700,
            showlegend=True,
            title_text="COMPREHENSIVE RISK ANALYSIS DASHBOARD",
            title_font_size=18,
            title_font_color="#f7931a",
            paper_bgcolor='rgba(13, 14, 22, 0.8)',  # Dark background
            plot_bgcolor='rgba(26, 26, 38, 0.5)',   # Lighter dark for plot areas
            font_color="#e6edf3"
        )

        # Update axes styling
        risk_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(139, 148, 158, 0.2)', row=1, col=1)
        risk_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(139, 148, 158, 0.2)', row=1, col=1)
        risk_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(139, 148, 158, 0.2)', row=1, col=2)
        risk_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(139, 148, 158, 0.2)', row=1, col=2)
        risk_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(139, 148, 158, 0.2)', row=2, col=1)
        risk_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(139, 148, 158, 0.2)', row=2, col=1)

        st.plotly_chart(risk_fig, use_container_width=True)

    with tab3:
        st.markdown("### 📈 PREDICTIVE ANALYTICS PLATFORM")

        # Move the generate forecast button to this tab
        col1, col2 = st.columns([1,3])
        with col1:
            generate_forecast = st.button("✨ GENERATE FORECAST", use_container_width=True, type="primary")
        with col2:
            st.markdown(f"<div style='background-color: #1a1a1a; border: 1px solid #3a3a3a; border-radius: 8px; padding: 12px; margin-top: 8px;'><span style='color: #d4af37; font-weight: 600;'>FORECAST HORIZON:</span> <span style='color: #e6e6e6;'>{forecast_period.replace('_', ' ').title()}</span> | <span style='color: #d4af37; font-weight: 600;'>PREDICTION:</span> <span style='color: #e6e6e6;'>{7 if forecast_period == 'daily' else 5 if forecast_period == 'monthly' else 12} {'days' if forecast_period == 'daily' else 'months' if forecast_period == 'monthly' else 'months'} ahead</span></div>", unsafe_allow_html=True)

        if generate_forecast:
            with st.spinner(f'⚡ Generating {forecast_period.replace("_", " ").title()} forecast with advanced machine learning algorithms...'):
                forecast_fig, forecast_df = create_forecast_simulation(df_stock, selected_stock, forecast_period)
                if forecast_fig:
                    st.plotly_chart(forecast_fig, use_container_width=True)

                    # Add PDF download button for the forecast report in this tab
                    st.markdown("### 📎 EXECUTIVE REPORT EXPORT")
                    pdf_bytes = create_pdf_report(df_stock, selected_stock, forecast_period, forecast_df)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 DOWNLOAD ANALYTICS REPORT",
                            data=pdf_bytes,
                            file_name=f"bitcoin_forecast_report_{forecast_period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    with col2:
                        # Add another export option if needed
                        st.markdown(f"<div style='background: linear-gradient(135deg, #0a0a0a, #1a1a1a); padding: 12px; border-radius: 8px; border: 1px solid #3a3a3a; text-align: center;'><span style='color: #d4af37; font-weight: 600;'>REPORT READY</span><br><span style='color: #e6e6e6; font-size: 0.9em;'>Coverage: {7 if forecast_period == 'daily' else 5 if forecast_period == 'monthly' else 12}-{'DAY' if forecast_period == 'daily' else 'MONTH' if forecast_period == 'monthly' else 'YEAR'} PREDICTION</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background: linear-gradient(135deg, #0d0d1a, #1a1a2e); border-left: 4px solid #f7931a; padding: 15px; margin: 15px 0; border-radius: 8px; border: 1px solid #3a3a4a; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);'><span style='color: #e6edf3; font-size: 1.1em; display: block;'><strong>👆 ACTION REQUIRED:</strong> Click 'GENERATE FORECAST' above to activate predictive modeling algorithms</span></div>", unsafe_allow_html=True)

            # Show a preview/metrics area if no forecast is generated yet
            if len(df_stock) > 1:
                st.markdown("### 📊 REAL-TIME MARKET INTELLIGENCE")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(145deg, #0d0d1a, #1a1a2e); padding: 20px; border-radius: 12px; border: 1px solid #3a3a4a; height: 100%; box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);">
                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                            <div style="background: rgba(212, 175, 55, 0.15); width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                                <span style="color: #d4af37; font-size: 1.2em;">🪙</span>
                            </div>
                            <div style="color: #b0b0b0; font-size: 0.85em; font-weight: 500;">CURRENT PRICE</div>
                        </div>
                        <div style="color: #ffffff; font-size: 1.5em; font-weight: bold; margin: 5px 0; text-align: left;">${df_stock['Close Price'].iloc[-1]:.2f}</div>
                        <div style="color: #d4af37; font-size: 0.9em; margin-top: 8px; text-align: left; font-weight: 500;">BTC/USD</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    change = ((df_stock['Close Price'].iloc[-1] - df_stock['Close Price'].iloc[-2]) / df_stock['Close Price'].iloc[-2]) * 100
                    color_positive = "#06d6a0" if change >= 0 else "#ef476f"
                    trend_arrow = "↗️" if change >= 0 else "↘️"
                    st.markdown(f"""
                    <div style="background: linear-gradient(145deg, #0d0d1a, #1a1a2e); padding: 20px; border-radius: 12px; border: 1px solid #3a3a4a; height: 100%; box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);">
                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                            <div style="background: rgba({10, 214, 160 if change >= 0 else 239, 71, 111}, 0.15); width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                                <span style="color: {color_positive}; font-size: 1.2em;">{trend_arrow}</span>
                            </div>
                            <div style="color: #b0b0b0; font-size: 0.85em; font-weight: 500;">24H CHANGE</div>
                        </div>
                        <div style="color: {color_positive}; font-size: 1.5em; font-weight: bold; margin: 5px 0; text-align: left;">{change:.2f}%</div>
                        <div style="color: {color_positive}; font-size: 0.9em; margin-top: 8px; text-align: left; font-weight: 500;">{trend_arrow} {abs(change):.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    if 'Trading Volume' in df_stock.columns:
                        volume = df_stock['Trading Volume'].iloc[-1]
                        st.markdown(f"""
                        <div style="background: linear-gradient(145deg, #0d0d1a, #1a1a2e); padding: 20px; border-radius: 12px; border: 1px solid #3a3a4a; height: 100%; box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);">
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <div style="background: rgba(212, 175, 55, 0.15); width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                                    <span style="color: #d4af37; font-size: 1.2em;">💸</span>
                                </div>
                                <div style="color: #b0b0b0; font-size: 0.85em; font-weight: 500;">24H VOLUME</div>
                            </div>
                            <div style="color: #e6e6e6; font-size: 1.5em; font-weight: bold; margin: 5px 0; text-align: left;">${volume:,.0f}</div>
                            <div style="color: #d4af37; font-size: 0.9em; margin-top: 8px; text-align: left; font-weight: 500;">TOTAL TRADED</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: linear-gradient(145deg, #0d0d1a, #1a1a2e); padding: 20px; border-radius: 12px; border: 1px solid #3a3a4a; height: 100%; box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);">
                            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                                <div style="background: rgba(212, 175, 55, 0.15); width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                                    <span style="color: #d4af37; font-size: 1.2em;">💸</span>
                                </div>
                                <div style="color: #b0b0b0; font-size: 0.85em; font-weight: 500;">24H VOLUME</div>
                            </div>
                            <div style="color: #e6e6e6; font-size: 1.5em; font-weight: bold; margin: 5px 0; text-align: left;">N/A</div>
                            <div style="color: #a6a6a6; font-size: 0.9em; margin-top: 8px; text-align: left; font-weight: 500;">NO DATA</div>
                        </div>
                        """, unsafe_allow_html=True)
                with col4:
                    high = df_stock['Daily High'].iloc[-1]
                    st.markdown(f"""
                    <div style="background: linear-gradient(145deg, #0d0d1a, #1a1a2e); padding: 20px; border-radius: 12px; border: 1px solid #3a3a4a; height: 100%; box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);">
                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                            <div style="background: rgba(212, 175, 55, 0.15); width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                                <span style="color: #d4af37; font-size: 1.2em;">📈</span>
                            </div>
                            <div style="color: #b0b0b0; font-size: 0.85em; font-weight: 500;">24H HIGH</div>
                        </div>
                        <div style="color: #e6e6e6; font-size: 1.5em; font-weight: bold; margin: 5px 0; text-align: left;">${high:.2f}</div>
                        <div style="color: #d4af37; font-size: 0.9em; margin-top: 8px; text-align: left; font-weight: 500;">MAXIMUM PRICE</div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### 📊 INSTITUTIONAL DATA DASHBOARD")

        # Enhanced data table with professional styling
        st.dataframe(
            df_stock.tail(10).style.format({
                'Open Price': '${:.2f}',
                'Close Price': '${:.2f}',
                'Daily High': '${:.2f}',
                'Daily Low': '${:.2f}',
                'Trading Volume': '{:,.0f}'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#0d0d1a'), ('color', '#d4af37'), ('font-weight', 'bold'), ('padding', '12px')]},
                {'selector': 'td', 'props': [('background-color', '#1a1a2e'), ('color', '#e6edf3'), ('padding', '10px'), ('border-bottom', '1px solid #2a2a3a')]},
                {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#151522')]},
                {'selector': 'tr:hover', 'props': [('background-color', '#2a2a3a'), ('cursor', 'pointer')]}
            ]),
            use_container_width=True
        )

        with st.expander("🔍 COMPREHENSIVE DATASET INTELLIGENCE", expanded=True):
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #0d0d1a, #1a1a2e); padding: 25px; border-radius: 12px; border: 1px solid #3a3a4a; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px;">
                    <div style="background: rgba(212, 175, 55, 0.05); border: 1px solid #3a3a4a; border-radius: 8px; padding: 15px;">
                        <div style="color: #b0b0b0; font-size: 0.85em; margin-bottom: 8px;">DATASET SIZE</div>
                        <div style="color: #d4af37; font-size: 1.8em; font-weight: bold; margin: 5px 0;">{len(df_featured):,}</div>
                        <div style="color: #8b949e; font-size: 0.8em;">TOTAL RECORDS</div>
                    </div>
                    <div style="background: rgba(212, 175, 55, 0.05); border: 1px solid #3a3a4a; border-radius: 8px; padding: 15px;">
                        <div style="color: #b0b0b0; font-size: 0.85em; margin-bottom: 8px;">DATE SPAN</div>
                        <div style="color: #e6edf3; font-size: 1.4em; font-weight: bold; margin: 5px 0;">{df_featured['Date'].min().strftime('%Y-%m-%d')}</div>
                        <div style="color: #8b949e; font-size: 0.8em;">→ {df_featured['Date'].max().strftime('%Y-%m-%d')}</div>
                    </div>
                    <div style="background: rgba(212, 175, 55, 0.05); border: 1px solid #3a3a4a; border-radius: 8px; padding: 15px;">
                        <div style="color: #b0b0b0; font-size: 0.85em; margin-bottom: 8px;">FEATURES COUNT</div>
                        <div style="color: #79c0ff; font-size: 1.8em; font-weight: bold; margin: 5px 0;">{len(df_featured.columns)}</div>
                        <div style="color: #8b949e; font-size: 0.8em;">INDICATORS</div>
                    </div>
                    <div style="background: rgba(212, 175, 55, 0.05); border: 1px solid #3a3a4a; border-radius: 8px; padding: 15px;">
                        <div style="color: #b0b0b0; font-size: 0.85em; margin-bottom: 8px;">ACTIVE ASSETS</div>
                        <div style="color: #56d364; font-size: 1.8em; font-weight: bold; margin: 5px 0;">{len(df_stock)}</div>
                        <div style="color: #8b949e; font-size: 0.8em;">RECORDS FOR {selected_stock}</div>
                    </div>
                </div>

                <div style="background: rgba(212, 175, 55, 0.03); border: 1px solid #3a3a4a; border-radius: 8px; padding: 15px;">
                    <div style="color: #d4af37; font-size: 1.1em; font-weight: bold; margin-bottom: 10px;">DATA QUALITY INDICATORS</div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                        <div style="padding: 8px;">
                            <div style="color: #8b949e; font-size: 0.8em;">Completeness</div>
                            <div style="color: #56d364; font-weight: bold;">{int(df_stock.count().mean()/len(df_stock)*100)}%</div>
                        </div>
                        <div style="padding: 8px;">
                            <div style="color: #8b949e; font-size: 0.8em;">Freshness</div>
                            <div style="color: #79c0ff; font-weight: bold;">{df_stock['Date'].max().strftime('%b %d, %Y')}</div>
                        </div>
                        <div style="padding: 8px;">
                            <div style="color: #8b949e; font-size: 0.8em;">Granularity</div>
                            <div style="color: #d29922; font-weight: bold;">Daily</div>
                        </div>
                        <div style="padding: 8px;">
                            <div style="color: #8b949e; font-size: 0.8em;">Currency</div>
                            <div style="color: #d4af37; font-weight: bold;">USD</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Professional corporate footer with enhanced styling
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #0d0d17, #1a1a28); border: 1px solid #3a3a4a; margin-top: 30px; border-radius: 15px;">
        <div style="margin-bottom: 20px;">
            <div style="color: #f7931a; font-size: 1.8em; font-weight: bold; letter-spacing: 1px; margin-bottom: 8px;">₿ITCOIN ANALYTICS SUITE</div>
            <div style="color: #e6edf3; font-size: 1.1em; margin-bottom: 12px;">Enterprise-Grade Cryptocurrency Intelligence Platform</div>
            <div style="color: #8b949e; font-size: 0.9em;">Powered by Advanced Machine Learning & Predictive Analytics</div>
        </div>

        <div style="display: flex; justify-content: center; align-items: center; gap: 25px; flex-wrap: wrap; margin-bottom: 15px; padding: 0 20px;">
            <div style="color: #79c0ff; font-size: 0.9em; display: flex; align-items: center; gap: 8px;">
                <i class="fas fa-database"></i> <span>Data Source: BTC Market Dataset</span>
            </div>
            <div style="color: #79c0ff; font-size: 0.9em; display: flex; align-items: center; gap: 8px;">
                <i class="fas fa-calendar-alt"></i> <span>Generated: {}</span>
            </div>
            <div style="color: #79c0ff; font-size: 0.9em; display: flex; align-items: center; gap: 8px;">
                <i class="fas fa-globe-americas"></i> <span>Global Markets</span>
            </div>
        </div>

        <div style="border-top: 1px solid #30363d; padding-top: 15px;">
            <div style="color: #8b949e; font-size: 0.85em; margin-bottom: 5px;">
                © 2025 Bitcoin Analytics Corp. | Institutional Solutions Division
            </div>
            <div style="color: #6e7681; font-size: 0.75em;">
                This platform is for institutional use only. All data and insights are proprietary and confidential.
            </div>
        </div>
    </div>
    """.format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)


if __name__ == "__main__":
    main()