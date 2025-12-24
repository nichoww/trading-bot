import logging
import pickle
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = Path('data/processed')
MODEL_PATH = Path('models')
FEATURES_FILE = DATA_PATH / 'features_multiday.csv'
MODEL_FILE = MODEL_PATH / 'rf_model_1d_optimized.pkl'
SCALER_FILE = MODEL_PATH / 'scaler_1d_optimized.pkl'

def main():
    logger.info("\n" + "="*80)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*80)
    
    # Load features
    logger.info("\nLoading features...")
    features = pd.read_csv(FEATURES_FILE)
    logger.info(f"✓ Loaded features: {features.shape}")
    
    # Load model
    logger.info("\nLoading optimized model...")
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
    except:
        model = joblib.load(MODEL_FILE)
    logger.info(f"✓ Model loaded")
    
    # Load scaler to get feature names
    logger.info("\nLoading scaler...")
    try:
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = joblib.load(SCALER_FILE)
    logger.info(f"✓ Scaler loaded")
    
    # Get numeric columns
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'balance']
    
    # The model was trained on scaled features, use first n_estimators features
    n_features = model.n_features_in_
    numeric_cols = numeric_cols[:n_features]
    
    logger.info(f"✓ Selected {len(numeric_cols)} numeric features (model has {n_features})")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': numeric_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Calculate cumulative importance
    importance_df['cumsum'] = importance_df['importance'].cumsum()
    importance_df['cumsum_pct'] = (importance_df['cumsum'] / importance_df['importance'].sum()) * 100
    
    # Categorize features
    feature_categories = {
        'price_change': ['price_change_1d', 'price_change_3d', 'price_change_5d', 'price_change_7d'],
        'momentum_features': ['momentum_5d', 'momentum_10d', 'momentum_20d'],
        'volatility_features': ['volatility_5d', 'volatility_10d', 'volatility_ratio', 
                               'rolling_std_5d', 'rolling_std_10d'],
        'volume_features': ['volume_avg_5d', 'volume_trend', 'money_flow_index'],
        'trend_features': ['adx', 'consecutive_up_days', 'consecutive_down_days'],
        'market_regime': ['above_sma_20', 'above_sma_50', 'sma_20_50_distance', 
                         'sma_crossover', 'volatility_regime'],
        'basic_features': ['open', 'high', 'low', 'close', 'volume', 'sma_5', 'sma_10', 
                          'sma_20', 'sma_50', 'rsi_14', 'upper_band', 'lower_band', 
                          'macd', 'signal_line', 'histogram']
    }
    
    # Add category to importance_df
    def get_category(feature):
        for cat, features_list in feature_categories.items():
            if feature in features_list:
                return cat
        return 'unknown'
    
    importance_df['category'] = importance_df['feature'].apply(get_category)
    
    # Print top features
    logger.info("\n" + "="*80)
    logger.info("TOP 20 MOST IMPORTANT FEATURES")
    logger.info("="*80)
    
    for idx, row in importance_df.head(20).iterrows():
        logger.info(f"{idx+1:2d}. {row['feature']:30s} | Importance: {row['importance']:.6f} | Category: {row['category']}")
    
    # Print category summary
    logger.info("\n" + "="*80)
    logger.info("FEATURE IMPORTANCE BY CATEGORY")
    logger.info("="*80)
    
    category_summary = importance_df.groupby('category')['importance'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
    category_summary['sum_pct'] = (category_summary['sum'] / category_summary['sum'].sum()) * 100
    
    logger.info(f"\n{'Category':<20} {'Total':>12} {'Avg':>12} {'Count':>8} {'% of Total':>12}")
    logger.info("-" * 65)
    for cat, row in category_summary.iterrows():
        logger.info(f"{cat:<20} {row['sum']:>12.6f} {row['mean']:>12.6f} {int(row['count']):>8d} {row['sum_pct']:>11.2f}%")
    
    # Print feature improvement
    logger.info("\n" + "="*80)
    logger.info("NEW FEATURES PERFORMANCE (Added in Feature Engineering v2)")
    logger.info("="*80)
    
    new_features = {
        'Momentum (5d, 10d, 20d)': importance_df[importance_df['feature'].isin(['momentum_5d', 'momentum_10d', 'momentum_20d'])],
        'Volatility (5d, 10d, ratio, std)': importance_df[importance_df['feature'].isin(['volatility_5d', 'volatility_10d', 'volatility_ratio', 'rolling_std_5d', 'rolling_std_10d'])],
        'Volume Features': importance_df[importance_df['feature'].isin(['volume_avg_5d', 'volume_trend', 'money_flow_index'])],
        'Trend Strength': importance_df[importance_df['feature'].isin(['adx', 'consecutive_up_days', 'consecutive_down_days'])],
        'Market Regime': importance_df[importance_df['feature'].isin(['above_sma_20', 'above_sma_50', 'sma_20_50_distance', 'sma_crossover', 'volatility_regime'])],
    }
    
    logger.info("\n")
    for group_name, group_df in new_features.items():
        if len(group_df) > 0:
            total_importance = group_df['importance'].sum()
            logger.info(f"{group_name}:")
            logger.info(f"  Total Importance: {total_importance:.6f}")
            logger.info(f"  Features ({len(group_df)}):")
            for _, row in group_df.sort_values('importance', ascending=False).iterrows():
                logger.info(f"    • {row['feature']:30s}: {row['importance']:.6f} (Rank: {row.name + 1})")
            logger.info("")
    
    # Save to file
    importance_df.to_csv(MODEL_PATH / 'feature_importances.csv', index=False)
    logger.info(f"\n✓ Feature importances saved to {MODEL_PATH / 'feature_importances.csv'}")
    
    # Calculate statistics
    top_10_pct = importance_df.head(int(len(importance_df) * 0.1))
    logger.info("\n" + "="*80)
    logger.info("STATISTICS")
    logger.info("="*80)
    logger.info(f"Total Features: {len(importance_df)}")
    logger.info(f"Top 10% ({len(top_10_pct)} features) account for: {top_10_pct['importance'].sum():.2%} of importance")
    logger.info(f"Top 5 features account for: {importance_df.head(5)['importance'].sum():.2%} of importance")
    
    logger.info("\n✓ Analysis complete!")

if __name__ == '__main__':
    main()
