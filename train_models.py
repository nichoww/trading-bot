#!/usr/bin/env python3
"""
Main script to train all models on engineered features
"""

import os
import pandas as pd
import logging
from pathlib import Path
from src.model_training import (
    train_random_forest,
    train_gradient_boosting,
    train_logistic_regression,
    prepare_training_data,
    save_model
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
PROCESSED_DATA_DIR = Path('data/processed')
MODELS_DIR = Path('models')
TICKERS = ['AAPL', 'AMD', 'AMZN', 'BA', 'COST', 'DIS', 'F', 'GE', 'GM', 
           'GOOGL', 'INTC', 'JPM', 'META', 'MSFT', 'NFLX', 'NVDA', 'PYPL', 
           'TSLA', 'V', 'WMT']

# Create models directory
MODELS_DIR.mkdir(exist_ok=True)

def load_all_features():
    """Load and combine all feature files into one dataset"""
    dfs = []
    
    logger.info(f"Loading features from {PROCESSED_DATA_DIR}...")
    
    for ticker in TICKERS:
        feature_file = PROCESSED_DATA_DIR / f"{ticker}_features.csv"
        if feature_file.exists():
            df = pd.read_csv(feature_file)
            dfs.append(df)
            logger.info(f"Loaded {len(df)} rows for {ticker}")
        else:
            logger.warning(f"Feature file not found: {feature_file}")
    
    if not dfs:
        logger.error("No feature files found!")
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(combined_df)} total rows from {len(dfs)} tickers")
    
    return combined_df

def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("TRAINING MACHINE LEARNING MODELS")
    print("="*80 + "\n")
    
    # Load combined features
    df = load_all_features()
    if df is None:
        logger.error("Failed to load feature data")
        return
    
    # Check for Target_1d column
    if 'Target_1d' not in df.columns:
        logger.error("Target_1d column not found in features!")
        logger.info(f"Available columns: {df.columns.tolist()}")
        return
    
    # Prepare training data
    logger.info("\nPreparing training data...")
    
    # Drop non-numeric columns for training (ticker, date)
    df_train = df.drop(columns=['ticker', 'date'], errors='ignore')
    
    # Verify no NaN in target
    nan_targets = df_train['Target_1d'].isna().sum()
    if nan_targets > 0:
        logger.warning(f"Dropping {nan_targets} rows with NaN targets")
        df_train = df_train.dropna(subset=['Target_1d'])
    
    # Encode categorical features
    logger.info("Encoding categorical features...")
    
    # RSI_Category: 'oversold' (0), 'neutral' (1), 'overbought' (2)
    if 'RSI_Category' in df_train.columns:
        rsi_mapping = {'oversold': 0, 'neutral': 1, 'overbought': 2}
        df_train['RSI_Category'] = df_train['RSI_Category'].map(rsi_mapping)
        logger.info("Encoded RSI_Category: oversold=0, neutral=1, overbought=2")
    
    # MACD_Crossover: 'bullish' (1), 'no_cross' (0), 'bearish' (-1)
    if 'MACD_Crossover' in df_train.columns:
        macd_mapping = {'bullish': 1, 'no_cross': 0, 'bearish': -1}
        df_train['MACD_Crossover'] = df_train['MACD_Crossover'].map(macd_mapping)
        logger.info("Encoded MACD_Crossover: bullish=1, no_cross=0, bearish=-1")
    
    logger.info(f"Training on {len(df_train)} samples")
    logger.info(f"Feature columns: {[c for c in df_train.columns if c != 'Target_1d']}")
    
    # Split and scale data
    X_train, X_test, y_train, y_test, scaler = prepare_training_data(
        df_train, 
        target_column='Target_1d',
        test_size=0.2,
        random_state=42
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Positive class: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    
    # Train Random Forest
    print("\n" + "-"*80)
    print("Training Random Forest Classifier...")
    print("-"*80)
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    save_model(rf_model, MODELS_DIR / 'random_forest_model.joblib')
    
    # Train Gradient Boosting
    print("\n" + "-"*80)
    print("Training Gradient Boosting Classifier...")
    print("-"*80)
    gb_model, gb_metrics = train_gradient_boosting(X_train, X_test, y_train, y_test)
    save_model(gb_model, MODELS_DIR / 'gradient_boosting_model.joblib')
    
    # Train Logistic Regression
    print("\n" + "-"*80)
    print("Training Logistic Regression...")
    print("-"*80)
    lr_model, lr_metrics = train_logistic_regression(X_train, X_test, y_train, y_test)
    save_model(lr_model, MODELS_DIR / 'logistic_regression_model.joblib')
    
    # Save scaler
    save_model(scaler, MODELS_DIR / 'feature_scaler.joblib')
    
    # Summary
    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETE")
    print("="*80)
    print(f"\nModels saved to: {MODELS_DIR.absolute()}")
    print("\nPerformance Summary:")
    print(f"\nRandom Forest:")
    print(f"  Train Accuracy: {rf_metrics['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {rf_metrics['test_accuracy']:.4f}")
    print(f"  Precision:      {rf_metrics['precision']:.4f}")
    print(f"  Recall:         {rf_metrics['recall']:.4f}")
    print(f"  F1-Score:       {rf_metrics['f1_score']:.4f}")
    
    print(f"\nGradient Boosting:")
    print(f"  Train Accuracy: {gb_metrics['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {gb_metrics['test_accuracy']:.4f}")
    print(f"  Precision:      {gb_metrics['precision']:.4f}")
    print(f"  Recall:         {gb_metrics['recall']:.4f}")
    print(f"  F1-Score:       {gb_metrics['f1_score']:.4f}")
    
    print(f"\nLogistic Regression:")
    print(f"  Train Accuracy: {lr_metrics['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {lr_metrics['test_accuracy']:.4f}")
    print(f"  Precision:      {lr_metrics['precision']:.4f}")
    print(f"  Recall:         {lr_metrics['recall']:.4f}")
    print(f"  F1-Score:       {lr_metrics['f1_score']:.4f}")
    
    print("\n" + "="*80)
    print("Ready for backtesting!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
