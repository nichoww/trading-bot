#!/usr/bin/env python3
"""
Main script to backtest trading strategies using trained models
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
import sqlite3
from datetime import datetime

from src.backtesting import Backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
DB_PATH = Path('data/market_data.db')
MODELS_DIR = Path('models')
PROCESSED_DATA_DIR = Path('data/processed')
BACKTEST_RESULTS_DIR = Path('backtest_results')
TICKERS = ['AAPL', 'AMD', 'AMZN', 'BA', 'COST', 'DIS', 'F', 'GE', 'GM', 
           'GOOGL', 'INTC', 'JPM', 'META', 'MSFT', 'NFLX', 'NVDA', 'PYPL', 
           'TSLA', 'V', 'WMT']

# Create results directory
BACKTEST_RESULTS_DIR.mkdir(exist_ok=True)

def load_market_data_from_db(ticker):
    """Load original OHLCV data from SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f"SELECT * FROM ohlcv WHERE ticker = '{ticker}' ORDER BY date"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            logger.warning(f"No data found for {ticker}")
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    except Exception as e:
        logger.error(f"Error loading market data for {ticker}: {str(e)}")
        return None


def load_features(ticker):
    """Load engineered features for a ticker"""
    feature_file = PROCESSED_DATA_DIR / f"{ticker}_features.csv"
    
    if not feature_file.exists():
        logger.warning(f"Feature file not found: {feature_file}")
        return None
    
    try:
        df = pd.read_csv(feature_file)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        logger.error(f"Error loading features for {ticker}: {str(e)}")
        return None


def load_models():
    """Load all trained models and scaler"""
    try:
        rf_model = joblib.load(MODELS_DIR / 'random_forest_model.joblib')
        gb_model = joblib.load(MODELS_DIR / 'gradient_boosting_model.joblib')
        lr_model = joblib.load(MODELS_DIR / 'logistic_regression_model.joblib')
        scaler = joblib.load(MODELS_DIR / 'feature_scaler.joblib')
        
        logger.info(f"Loaded models from {MODELS_DIR}")
        return rf_model, gb_model, lr_model, scaler
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def generate_signals(features_df, rf_model, gb_model, lr_model, scaler):
    """
    Generate trading signals using trained models
    
    Args:
        features_df: DataFrame with engineered features
        rf_model: Random Forest model
        gb_model: Gradient Boosting model
        lr_model: Logistic Regression model
        scaler: Feature scaler
        
    Returns:
        DataFrame with trading signals
    """
    try:
        # Prepare features for prediction (same as training)
        feature_cols = [c for c in features_df.columns if c not in ['ticker', 'date', 'Target_1d']]
        
        # Encode categorical features
        df = features_df.copy()
        
        if 'RSI_Category' in df.columns:
            rsi_mapping = {'oversold': 0, 'neutral': 1, 'overbought': 2}
            df['RSI_Category'] = df['RSI_Category'].map(rsi_mapping)
        
        if 'MACD_Crossover' in df.columns:
            macd_mapping = {'bullish': 1, 'no_cross': 0, 'bearish': -1}
            df['MACD_Crossover'] = df['MACD_Crossover'].map(macd_mapping)
        
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)
        
        # Get predictions from all models
        rf_pred = rf_model.predict(X_scaled)
        gb_pred = gb_model.predict(X_scaled)
        lr_pred = lr_model.predict(X_scaled)
        
        # Get probabilities for voting/confidence
        rf_proba = rf_model.predict_proba(X_scaled)[:, 1]
        gb_proba = gb_model.predict_proba(X_scaled)[:, 1]
        lr_proba = lr_model.predict_proba(X_scaled)[:, 1]
        
        # Ensemble: vote with probability weighting
        avg_proba = (rf_proba + gb_proba + lr_proba) / 3
        ensemble_pred = (avg_proba > 0.5).astype(int)
        
        # Generate signals: 1=buy (prediction is 1), -1=sell (prediction is 0), 0=hold
        df['Signal'] = np.where(ensemble_pred == 1, 1, -1)
        df['Confidence'] = np.where(ensemble_pred == 1, avg_proba, 1 - avg_proba)
        
        # Only buy if confidence > 55%, otherwise hold
        df['Signal'] = np.where((df['Confidence'] > 0.55) & (ensemble_pred == 1), 1,
                                np.where((df['Confidence'] > 0.55) & (ensemble_pred == 0), -1, 0))
        
        return df
        
    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        raise


def run_backtest_for_ticker(ticker, market_data, features_df, rf_model, gb_model, lr_model, scaler):
    """Run backtest for a single ticker"""
    try:
        # Generate signals
        signals_df = generate_signals(features_df, rf_model, gb_model, lr_model, scaler)
        
        # Merge with market data (use Close prices)
        backtest_data = market_data[['date', 'close']].copy()
        backtest_data.columns = ['date', 'Close']
        backtest_data['date'] = pd.to_datetime(backtest_data['date'])
        
        # Merge signals with price data
        signals_df['date'] = pd.to_datetime(signals_df['date'])
        merged = pd.merge(backtest_data, signals_df[['date', 'Signal', 'Confidence']], 
                         on='date', how='inner')
        
        if merged.empty:
            logger.warning(f"No merged data for {ticker}")
            return None
        
        # Run backtest
        backtest = Backtest(merged, initial_capital=10000, commission=0.001)
        backtest.apply_signals('Signal')
        results = backtest.calculate_returns()
        
        # Add ticker to results
        results['Ticker'] = ticker
        results['Total_Trades'] = len(backtest.trades)
        
        logger.info(f"{ticker}: Return={results['total_return_pct']:.2f}%, "
                   f"Sharpe={results['sharpe_ratio']:.2f}, "
                   f"MaxDD={results['max_drawdown']:.2f}%, "
                   f"Trades={results['Total_Trades']}")
        
        return results, backtest
        
    except Exception as e:
        logger.error(f"Error running backtest for {ticker}: {str(e)}")
        return None, None


def main():
    """Main backtesting pipeline"""
    
    print("\n" + "="*80)
    print("BACKTESTING TRADING STRATEGIES")
    print("="*80 + "\n")
    
    # Load models
    logger.info("Loading trained models...")
    rf_model, gb_model, lr_model, scaler = load_models()
    
    # Run backtests
    all_results = []
    backtest_engines = {}
    
    logger.info(f"\nRunning backtests for {len(TICKERS)} tickers...\n")
    
    for ticker in TICKERS:
        # Load data
        market_data = load_market_data_from_db(ticker)
        features_df = load_features(ticker)
        
        if market_data is None or features_df is None:
            logger.warning(f"Skipping {ticker} - missing data")
            continue
        
        # Run backtest
        results, backtest = run_backtest_for_ticker(
            ticker, market_data, features_df, 
            rf_model, gb_model, lr_model, scaler
        )
        
        if results is not None:
            all_results.append(results)
            backtest_engines[ticker] = backtest
    
    if not all_results:
        logger.error("No backtests completed successfully")
        return
    
    # Create results summary
    results_df = pd.DataFrame(all_results)
    
    # Summary statistics
    print("\n" + "="*80)
    print("BACKTEST RESULTS SUMMARY")
    print("="*80 + "\n")
    
    print(f"Backtests Completed: {len(results_df)}/{len(TICKERS)}")
    print(f"\nPerformance Statistics:")
    print(f"  Average Return: {results_df['total_return_pct'].mean():.2f}%")
    print(f"  Median Return:  {results_df['total_return_pct'].median():.2f}%")
    print(f"  Best Return:    {results_df['total_return_pct'].max():.2f}% ({results_df.loc[results_df['total_return_pct'].idxmax(), 'Ticker']})")
    print(f"  Worst Return:   {results_df['total_return_pct'].min():.2f}% ({results_df.loc[results_df['total_return_pct'].idxmin(), 'Ticker']})")
    
    print(f"\nRisk Metrics:")
    print(f"  Average Sharpe:  {results_df['sharpe_ratio'].mean():.2f}")
    print(f"  Average Max DD:  {results_df['max_drawdown'].mean():.2f}%")
    print(f"  Win Rate:        {(results_df['total_return_pct'] > 0).sum()} / {len(results_df)} tickers ({(results_df['total_return_pct'] > 0).sum()/len(results_df)*100:.1f}%)")
    
    print(f"\nTrade Statistics:")
    print(f"  Average Trades:  {results_df['Total_Trades'].mean():.0f}")
    print(f"  Total Trades:    {results_df['Total_Trades'].sum()}")
    
    # Detailed results table
    print("\n" + "-"*80)
    print("DETAILED RESULTS BY TICKER")
    print("-"*80 + "\n")
    
    display_df = results_df[['Ticker', 'total_return_pct', 'sharpe_ratio', 'max_drawdown', 'Total_Trades']].copy()
    display_df.columns = ['Ticker', 'Return %', 'Sharpe', 'Max DD %', 'Trades']
    display_df = display_df.sort_values('Return %', ascending=False)
    
    print(display_df.to_string(index=False))
    
    # Save detailed results
    results_df.to_csv(BACKTEST_RESULTS_DIR / 'backtest_summary.csv', index=False)
    display_df.to_csv(BACKTEST_RESULTS_DIR / 'backtest_summary_display.csv', index=False)
    
    print(f"\n\nResults saved to: {BACKTEST_RESULTS_DIR.absolute()}")
    
    # Portfolio level backtest (combined signals)
    print("\n" + "="*80)
    print("PORTFOLIO-LEVEL ANALYSIS")
    print("="*80 + "\n")
    
    try:
        # Combine all signals
        portfolio_returns = []
        total_portfolio_value = 0
        
        for ticker in TICKERS:
            if ticker in backtest_engines:
                backtest = backtest_engines[ticker]
                if 'Portfolio_Value' in backtest.data.columns:
                    final_value = backtest.data['Portfolio_Value'].iloc[-1]
                    ret_pct = ((final_value - 10000) / 10000) * 100
                    portfolio_returns.append(ret_pct)
                    total_portfolio_value += final_value
        
        if portfolio_returns:
            print(f"Portfolio Summary (Equal-Weighted, $10k per ticker):")
            print(f"  Initial Capital:    ${len(TICKERS) * 10000:,.0f}")
            print(f"  Final Value:        ${total_portfolio_value:,.0f}")
            print(f"  Total Return:       {np.mean(portfolio_returns):.2f}%")
            print(f"  Return Volatility:  {np.std(portfolio_returns):.2f}%")
    except Exception as e:
        logger.warning(f"Could not calculate portfolio metrics: {str(e)}")
    
    print("\n" + "="*80)
    print("BACKTESTING COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
