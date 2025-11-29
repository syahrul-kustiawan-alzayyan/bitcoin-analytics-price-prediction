"""
Main execution script for financial analysis and forecasting project
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_loading import load_financial_dataset
from src.data_cleaning import clean_financial_data
from src.feature_engineering import engineer_features
from src.modeling import (
    random_forest_forecast, linear_regression_forecast, 
    arima_forecast, prophet_forecast, evaluate_model_performance
)
from src.visualization import create_interactive_dashboard


def main():
    print("="*70)
    print("         FINANCIAL ANALYSIS AND FORECASTING PROJECT")
    print("="*70)
    
    # Step 1: Load the dataset
    print("\n🔍 LOADING DATASET...")
    df = load_financial_dataset()
    print(f"   📊 Dataset shape: {df.shape}")
    print(f"   📅 Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"   🏢 Stock indices: {', '.join(df['Stock Index'].unique())}")
    
    # Step 2: Clean the data
    print("\n🧹 CLEANING DATASET...")
    df_clean = clean_financial_data(df)
    print(f"   ✅ Cleaned dataset shape: {df_clean.shape}")
    
    # Step 3: Engineer features
    print("\n⚙️  ENGINEERING FEATURES...")
    df_features = engineer_features(df_clean)
    num_new_features = len(df_features.columns) - len(df_clean.columns)
    print(f"   ✅ Feature-engineered dataset: {df_features.shape}")
    print(f"   ✅ New features created: {num_new_features}")
    
    # Step 4: Focus on a specific stock for modeling
    # Check what data we have - if single asset (like BTC), analyze it
    if df_features['Stock Index'].nunique() == 1:
        stock_to_analyze = df_features['Stock Index'].iloc[0]
    else:
        stock_to_analyze = 'S&P 500'  # You can change this to any stock in your dataset

    df_stock = df_features[df_features['Stock Index'] == stock_to_analyze].copy()

    print(f"\n📋 ANALYZING {stock_to_analyze}...")
    print(f"   📊 {stock_to_analyze} data shape: {df_stock.shape}")
    
    # Basic data exploration without plotting
    print(f"\n📈 {stock_to_analyze} STATISTICS:")
    print(f"   📅 Date range: {df_stock['Date'].min().strftime('%Y-%m-%d')} to {df_stock['Date'].max().strftime('%Y-%m-%d')}")
    
    # Show summary statistics for key financial metrics
    key_metrics = ['Close Price', 'Open Price', 'Daily High', 'Daily Low', 'Trading Volume']
    available_metrics = [col for col in key_metrics if col in df_stock.columns]
    
    if available_metrics:
        stats = df_stock[available_metrics].describe()
        print(f"   📊 Key metrics summary:")
        for metric in available_metrics:
            mean_val = df_stock[metric].mean()
            min_val = df_stock[metric].min()
            max_val = df_stock[metric].max()
            print(f"     • {metric}: Mean=${mean_val:.2f}, Min=${min_val:.2f}, Max=${max_val:.2f}")
    
    # Step 5: Model Development
    print("\n🤖 DEVELOPING FORECASTING MODELS...")
    
    # Prepare data for modeling - remove rows with NaN values for training
    df_stock_modeling = df_stock.dropna()
    print(f"   📊 Available data for modeling after cleaning NaNs: {df_stock_modeling.shape}")
    
    # Random Forest model
    print("\n   🌲 Training Random Forest model...")
    rf_results = None
    try:
        rf_results = random_forest_forecast(df_stock_modeling, target_col='Close Price')
        print(f"   ✅ Random Forest - MSE: {rf_results['mse']:.4f}, MAE: {rf_results['mae']:.4f}, R²: {rf_results['r2']:.4f}")
    except Exception as e:
        print(f"   ❌ Random Forest model failed: {str(e)[:100]}...")  # Truncate error message
    
    # Linear Regression model
    print("\n   ↔️  Training Linear Regression model...")
    lr_results = None
    try:
        lr_results = linear_regression_forecast(df_stock_modeling, target_col='Close Price')
        print(f"   ✅ Linear Regression - MSE: {lr_results['mse']:.4f}, MAE: {lr_results['mae']:.4f}, R²: {lr_results['r2']:.4f}")
    except Exception as e:
        print(f"   ❌ Linear Regression model failed: {str(e)[:100]}...")  # Truncate error message
    
    # Time series models
    target_series = df_stock['Close Price'].dropna()
    
    # ARIMA model
    print("\n   📈 Training ARIMA model...")
    arima_forecast_vals = None
    try:
        arima_forecast_vals, arima_conf_int, arima_model = arima_forecast(target_series, order=(1,1,1), forecast_steps=30)
        print(f"   ✅ ARIMA model trained successfully")
    except Exception as e:
        print(f"   ❌ ARIMA model failed: {str(e)[:100]}...")  # Truncate error message
    
    # Prophet model
    print("\n   🔮 Training Prophet model...")
    prophet_forecast_df = None
    try:
        prophet_forecast_df, prophet_model = prophet_forecast(df_stock[['Date', 'Close Price']], target_col='Close Price', forecast_periods=30)
        print(f"   ✅ Prophet model trained successfully")
    except Exception as e:
        print(f"   ❌ Prophet model failed: {str(e)[:100]}...")  # Truncate error message
    
    # Step 6: Results Summary
    print("\n🏆 MODEL PERFORMANCE SUMMARY:")
    
    # Determine best model
    best_model = None
    best_r2 = -float('inf')
    
    if rf_results and rf_results['r2'] > best_r2:
        best_model = "Random Forest"
        best_r2 = rf_results['r2']
    
    if lr_results and lr_results['r2'] > best_r2:
        best_model = "Linear Regression"
        best_r2 = lr_results['r2']
    
    if rf_results:
        print(f"   🌲 Random Forest: R² = {rf_results['r2']:.4f}, MSE = {rf_results['mse']:.4f}")
    if lr_results:
        print(f"   ↔️  Linear Regression: R² = {lr_results['r2']:.4f}, MSE = {lr_results['mse']:.4f}")
    
    if best_model:
        print(f"\n   🏆 BEST MODEL: {best_model} (R² = {best_r2:.4f})")
    
    print("\n   📝 Note: Models with R² closer to 1.0 are better, closer to 0 or negative are poor")
    
    # Step 7: Create interactive dashboard
    print("\n🌐 CREATING INTERACTIVE DASHBOARD...")
    try:
        dashboard = create_interactive_dashboard(df_stock)
        
        # Save the dashboard as an HTML file
        reports_dir = os.path.join('reports')
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir, exist_ok=True)
        
        dashboard_path = os.path.join(reports_dir, 'interactive_dashboard.html')
        dashboard.write_html(dashboard_path)
        print(f"   ✅ Dashboard saved to: {dashboard_path}")
    except Exception as e:
        print(f"   ❌ Dashboard creation failed: {str(e)[:100]}...")
    
    # Step 8: Save processed data
    print("\n💾 SAVING PROCESSED DATA...")
    try:
        processed_path = os.path.join('data', 'processed', 'featured_data.csv')
        df_features.to_csv(processed_path, index=False)
        print(f"   ✅ Processed data saved to: {processed_path}")
    except Exception as e:
        print(f"   ❌ Data saving failed: {str(e)[:100]}...")
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 FINANCIAL ANALYSIS AND FORECASTING PROJECT COMPLETE!")
    print("="*70)
    print(f"📊 Analyzed: {len(df['Stock Index'].unique())} stock indices")
    print(f"⚙️  Features created: {num_new_features}")
    print(f"🤖 Models trained: Random Forest, Linear Regression, ARIMA, Prophet")
    print(f"🎯 Forecasting horizon: 30 days")
    print(f"📋 Dashboard available: reports/interactive_dashboard.html")
    print("="*70)
    
    print("\n📋 NEXT STEPS:")
    print("   1. Open reports/interactive_dashboard.html in your browser")
    print("   2. Run 'streamlit run dashboard/streamlit_app.py' for live dashboard") 
    print("   3. Check data/processed/featured_data.csv for model-ready data")
    print("   4. Explore Jupyter notebooks in the notebooks/ directory")
    if best_model:
        print(f"   5. {best_model} performed best (R² = {best_r2:.4f})")


if __name__ == "__main__":
    main()