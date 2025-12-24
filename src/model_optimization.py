"""
Model Optimization Script
==========================
Performs GridSearchCV hyperparameter tuning on the best-performing timeframe model.
Uses TimeSeriesSplit for proper time-series cross-validation.

Results from multi-day comparison:
- 1d: 48.04% ✓ BEST
- 3d: 46.88%
- 5d: 46.14%
- 7d: 47.02%

Therefore optimizing 1-day model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, classification_report)
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================================
# SETUP LOGGING
# ===========================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===========================================================================================
# CONSTANTS
# ===========================================================================================

PROCESSED_DATA_DIR = Path('./data/processed')
MODELS_DIR = Path('./models')
LOGS_DIR = Path('./logs')

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BEST_TIMEFRAME = '1d'
TARGET_COLUMN = 'Target_1d'

# ===========================================================================================
# BASELINE HYPERPARAMETERS
# ===========================================================================================

BASELINE_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 50,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

# Grid Search Parameters (reduced for faster execution)
GRID_PARAMS = {
    'n_estimators': [100, 150, 200],
    'max_depth': [8, 12, 16],
    'min_samples_split': [30, 50],
    'min_samples_leaf': [1, 2]
}

# ===========================================================================================
# CLASS: ModelOptimizer
# ===========================================================================================

class ModelOptimizer:
    """Handles hyperparameter optimization for Random Forest model."""
    
    def __init__(self):
        self.features_df = None
        self.baseline_results = {}
        self.optimization_results = {}
        self.best_model = None
        self.best_params = None
        self.scaler = None
        
    def load_features(self):
        """Load multi-day features CSV."""
        logger.info("="*80)
        logger.info("LOADING FEATURES")
        logger.info("="*80)
        
        feature_file = PROCESSED_DATA_DIR / 'features_multiday.csv'
        if not feature_file.exists():
            logger.error(f"Feature file not found: {feature_file}")
            raise FileNotFoundError(f"{feature_file}")
        
        self.features_df = pd.read_csv(feature_file)
        self.features_df['date'] = pd.to_datetime(self.features_df['date'])
        
        logger.info(f"✓ Loaded features: {self.features_df.shape[0]:,} rows × {self.features_df.shape[1]} cols")
        logger.info(f"  Date range: {self.features_df['date'].min()} to {self.features_df['date'].max()}")
        logger.info(f"  Target balance: {self.features_df[TARGET_COLUMN].sum():,} positive")
    
    def get_feature_columns(self):
        """Extract numeric features, excluding categorical columns."""
        logger.info("\nExtracting numeric features...")
        
        exclude_cols = {'date', 'ticker', TARGET_COLUMN, 'Target_1d', 'Target_1d_1pct',
                       'Target_3d', 'Target_3d_1pct', 'Target_5d', 'Target_5d_1pct',
                       'Target_7d', 'Target_7d_1pct', 'Return_1d', 'Return_3d', 'Return_5d',
                       'Return_7d', 'RSI_Category', 'MACD_Crossover'}
        
        features = [col for col in self.features_df.columns
                   if col not in exclude_cols and 
                   pd.api.types.is_numeric_dtype(self.features_df[col])]
        
        logger.info(f"✓ Selected {len(features)} numeric features")
        return features
    
    def create_time_series_split(self):
        """Create temporal train/val/test split (70/15/15)."""
        logger.info("\nCreating time-series split...")
        
        # Sort by date
        df = self.features_df.sort_values('date').reset_index(drop=True)
        n = len(df)
        
        train_size = int(n * 0.70)
        val_size = int(n * 0.15)
        
        split_info = {
            'train_end_idx': train_size,
            'val_end_idx': train_size + val_size,
            'train_end_date': df.iloc[train_size - 1]['date'],
            'val_end_date': df.iloc[train_size + val_size - 1]['date'],
            'test_end_date': df.iloc[-1]['date']
        }
        
        logger.info(f"  Train: rows 0-{train_size-1} ({train_size:,} samples)")
        logger.info(f"    Date range: {df.iloc[0]['date'].date()} to {split_info['train_end_date'].date()}")
        logger.info(f"  Val: rows {train_size}-{train_size+val_size-1} ({val_size:,} samples)")
        logger.info(f"    Date range: {split_info['train_end_date'].date()} to {split_info['val_end_date'].date()}")
        logger.info(f"  Test: rows {train_size+val_size}-{n-1} ({n-train_size-val_size:,} samples)")
        logger.info(f"    Date range: {split_info['val_end_date'].date()} to {split_info['test_end_date'].date()}")
        
        return df, split_info
    
    def train_baseline(self, features):
        """Train baseline model with original hyperparameters."""
        logger.info("\n" + "="*80)
        logger.info("BASELINE MODEL TRAINING")
        logger.info("="*80)
        
        df, split_info = self.create_time_series_split()
        
        # Extract data
        train_df = df.iloc[:split_info['train_end_idx']]
        val_df = df.iloc[split_info['train_end_idx']:split_info['val_end_idx']]
        test_df = df.iloc[split_info['val_end_idx']:]
        
        X_train = train_df[features].values
        y_train = train_df[TARGET_COLUMN].values
        X_val = val_df[features].values
        y_val = val_df[TARGET_COLUMN].values
        X_test = test_df[features].values
        y_test = test_df[TARGET_COLUMN].values
        
        # Scale features
        logger.info("\nScaling features (fit on training data)...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        logger.info("✓ Feature scaling complete")
        
        # Train baseline
        logger.info("\nTraining baseline Random Forest...")
        baseline_model = RandomForestClassifier(**BASELINE_PARAMS)
        baseline_model.fit(X_train_scaled, y_train)
        logger.info("✓ Baseline model training complete")
        
        # Evaluate on test set
        logger.info("\nEvaluating baseline on test set...")
        y_pred = baseline_model.predict(X_test_scaled)
        y_pred_proba = baseline_model.predict_proba(X_test_scaled)[:, 1]
        
        self.baseline_results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'params': BASELINE_PARAMS.copy()
        }
        
        logger.info(f"  Baseline Accuracy: {self.baseline_results['accuracy']:.4f}")
        logger.info(f"  Baseline Precision: {self.baseline_results['precision']:.4f}")
        logger.info(f"  Baseline Recall: {self.baseline_results['recall']:.4f}")
        logger.info(f"  Baseline F1-Score: {self.baseline_results['f1']:.4f}")
        logger.info(f"  Baseline ROC-AUC: {self.baseline_results['roc_auc']:.4f}")
        
        return df, split_info, features, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
    
    def optimize_hyperparameters(self, df, split_info, features):
        """Perform GridSearchCV for hyperparameter optimization."""
        logger.info("\n" + "="*80)
        logger.info("HYPERPARAMETER OPTIMIZATION - GRID SEARCH")
        logger.info("="*80)
        
        # Prepare data
        train_df = df.iloc[:split_info['train_end_idx']]
        val_df = df.iloc[split_info['train_end_idx']:split_info['val_end_idx']]
        test_df = df.iloc[split_info['val_end_idx']:]
        
        X_train = train_df[features].values
        y_train = train_df[TARGET_COLUMN].values
        X_val = val_df[features].values
        y_val = val_df[TARGET_COLUMN].values
        X_test = test_df[features].values
        y_test = test_df[TARGET_COLUMN].values
        
        # Scale using same scaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("\nGrid Search Parameters:")
        for param, values in GRID_PARAMS.items():
            logger.info(f"  {param}: {values}")
        
        total_combinations = np.prod([len(v) for v in GRID_PARAMS.values()])
        logger.info(f"\nTotal combinations to evaluate: {total_combinations}")
        
        # Create TimeSeriesSplit for CV
        tscv = TimeSeriesSplit(n_splits=5)
        
        logger.info("\nLaunching GridSearchCV with TimeSeriesSplit (5 splits)...")
        logger.info("This may take 30-60 minutes. Progress will be logged every 10% of combinations.\n")
        
        # Base estimator
        base_rf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_rf,
            param_grid=GRID_PARAMS,
            cv=tscv,
            scoring='accuracy',
            n_jobs=1,  # Don't parallelize outer CV
            verbose=2  # Print progress
        )
        
        import time
        start_time = time.time()
        
        grid_search.fit(X_train_scaled, y_train)
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"\n✓ GridSearchCV complete (took {elapsed_time/60:.1f} minutes)")
        
        # Best model and parameters
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        logger.info(f"\nBest parameters found:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        
        logger.info(f"\nCross-validation score (best): {grid_search.best_score_:.4f}")
        
        # Evaluate on validation set
        logger.info("\nEvaluating optimized model on validation set...")
        y_val_pred = self.best_model.predict(X_val_scaled)
        y_val_pred_proba = self.best_model.predict_proba(X_val_scaled)[:, 1]
        
        val_accuracy = accuracy_score(y_val, y_val_pred)
        logger.info(f"  Val Accuracy: {val_accuracy:.4f}")
        
        # Evaluate on test set
        logger.info("\nEvaluating optimized model on test set...")
        y_test_pred = self.best_model.predict(X_test_scaled)
        y_test_pred_proba = self.best_model.predict_proba(X_test_scaled)[:, 1]
        
        self.optimization_results = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'params': self.best_params.copy(),
            'cv_score': grid_search.best_score_,
            'elapsed_time': elapsed_time
        }
        
        logger.info(f"  Optimized Accuracy: {self.optimization_results['accuracy']:.4f}")
        logger.info(f"  Optimized Precision: {self.optimization_results['precision']:.4f}")
        logger.info(f"  Optimized Recall: {self.optimization_results['recall']:.4f}")
        logger.info(f"  Optimized F1-Score: {self.optimization_results['f1']:.4f}")
        logger.info(f"  Optimized ROC-AUC: {self.optimization_results['roc_auc']:.4f}")
        
        return X_test_scaled, y_test
    
    def compare_results(self):
        """Compare baseline vs optimized model."""
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON: BASELINE vs OPTIMIZED")
        logger.info("="*80)
        
        baseline_acc = self.baseline_results['accuracy']
        optimized_acc = self.optimization_results['accuracy']
        improvement = (optimized_acc - baseline_acc) * 100
        improvement_pct = (improvement / baseline_acc) * 100
        
        logger.info("\n" + "-"*80)
        logger.info("Comparison Table:")
        logger.info("-"*80)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        print("\n")
        print(f"{'Metric':<15} {'Baseline':<15} {'Optimized':<15} {'Improvement':<15}")
        print("-" * 60)
        
        for metric, key in zip(metrics, metric_keys):
            baseline_val = self.baseline_results[key]
            optimized_val = self.optimization_results[key]
            imp = (optimized_val - baseline_val) * 100
            
            print(f"{metric:<15} {baseline_val:.4f}{'':<9} {optimized_val:.4f}{'':<9} {imp:+.4f}%")
            logger.info(f"{metric}: {baseline_val:.4f} → {optimized_val:.4f} ({imp:+.4f}%)")
        
        print("-" * 60)
        print(f"{'ACCURACY':<15} {baseline_acc:.4f}{'':<9} {optimized_acc:.4f}{'':<9} {improvement:+.4f}% ({improvement_pct:+.2f}%)")
        
        logger.info(f"\nBaseline Accuracy: {baseline_acc:.4f}")
        logger.info(f"Optimized Accuracy: {optimized_acc:.4f}")
        logger.info(f"Absolute Improvement: {improvement:+.4f}%")
        logger.info(f"Relative Improvement: {improvement_pct:+.2f}%")
        
        # Recommendation
        logger.info("\n" + "-"*80)
        logger.info("RECOMMENDATION:")
        logger.info("-"*80)
        
        if improvement < 0.01:  # Less than 0.01% improvement
            logger.info("\n⚠️  BASELINE MODEL IS BETTER")
            logger.info("   The optimization didn't find better parameters.")
            logger.info("   Recommendation: Use BASELINE model (simpler is better)")
            logger.info("   Reason: Simpler models generalize better and are faster")
            recommendation = "BASELINE"
        elif improvement < 0.005:  # Less than 0.005% improvement
            logger.info("\n⚠️  MINIMAL IMPROVEMENT (<0.005%)")
            logger.info("   The optimization provided negligible improvement.")
            logger.info("   Recommendation: Use BASELINE model (not worth added complexity)")
            recommendation = "BASELINE"
        else:
            logger.info(f"\n✓ OPTIMIZATION SUCCESSFUL (+{improvement:.4f}%)")
            logger.info("   The optimized model shows measurable improvement.")
            logger.info("   Recommendation: Use OPTIMIZED model")
            recommendation = "OPTIMIZED"
        
        logger.info(f"\nFinal Decision: {recommendation}")
        logger.info("-"*80)
        
        return recommendation
    
    def save_results(self, recommendation, X_test_scaled, y_test):
        """Save models, parameters, and results."""
        logger.info("\n" + "="*80)
        logger.info("SAVING RESULTS")
        logger.info("="*80)
        
        # Save best model
        if recommendation == "OPTIMIZED":
            model_path = MODELS_DIR / f'rf_model_{BEST_TIMEFRAME}_optimized.pkl'
            joblib.dump(self.best_model, model_path)
            logger.info(f"✓ Saved optimized model to {model_path}")
            
            # Save best parameters
            params_path = MODELS_DIR / 'best_params.json'
            with open(params_path, 'w') as f:
                json.dump(self.best_params, f, indent=2)
            logger.info(f"✓ Saved best parameters to {params_path}")
        else:
            logger.info("ℹ Using baseline model - no optimization changes needed")
        
        # Save scaler
        scaler_path = MODELS_DIR / f'scaler_{BEST_TIMEFRAME}_optimized.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"✓ Saved scaler to {scaler_path}")
        
        # Save optimization results
        results_path = LOGS_DIR / 'optimization_results.txt'
        with open(results_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL OPTIMIZATION RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Timeframe: {BEST_TIMEFRAME}-day\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("BASELINE MODEL\n")
            f.write("-"*80 + "\n")
            for key, value in self.baseline_results.items():
                if key != 'confusion_matrix':
                    f.write(f"{key}: {value}\n")
            f.write(f"Confusion Matrix:\n{self.baseline_results['confusion_matrix']}\n\n")
            
            f.write("OPTIMIZED MODEL\n")
            f.write("-"*80 + "\n")
            for key, value in self.optimization_results.items():
                if key != 'confusion_matrix':
                    f.write(f"{key}: {value}\n")
            f.write(f"Confusion Matrix:\n{self.optimization_results['confusion_matrix']}\n\n")
            
            f.write("COMPARISON TABLE\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Metric':<20} {'Baseline':<20} {'Optimized':<20} {'Change':<20}\n")
            f.write("-"*80 + "\n")
            
            for metric, key in zip(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                                  ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
                baseline_val = self.baseline_results[key]
                optimized_val = self.optimization_results[key]
                change = optimized_val - baseline_val
                f.write(f"{metric:<20} {baseline_val:<20.4f} {optimized_val:<20.4f} {change:+.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATION\n")
            f.write("="*80 + "\n")
            if self.optimization_results['accuracy'] > self.baseline_results['accuracy']:
                improvement_pct = ((self.optimization_results['accuracy'] - self.baseline_results['accuracy']) 
                                  / self.baseline_results['accuracy'] * 100)
                f.write(f"Use OPTIMIZED model (accuracy improved by {improvement_pct:.2f}%)\n")
                f.write(f"Best Parameters: {json.dumps(self.best_params, indent=2)}\n")
            else:
                f.write("Use BASELINE model (simpler is better)\n")
        
        logger.info(f"✓ Saved results to {results_path}")
    
    def run_full_pipeline(self):
        """Run complete optimization pipeline."""
        logger.info("\n")
        logger.info("╔" + "="*78 + "╗")
        logger.info("║" + " "*20 + "MODEL OPTIMIZATION PIPELINE" + " "*31 + "║")
        logger.info("║" + " "*20 + f"Timeframe: {BEST_TIMEFRAME}-day (best performer)" + " "*13 + "║")
        logger.info("╚" + "="*78 + "╝")
        
        try:
            # Load features
            self.load_features()
            features = self.get_feature_columns()
            
            # Train baseline
            df, split_info, features, X_train, y_train, X_val, y_val, X_test, y_test = self.train_baseline(features)
            
            # Optimize hyperparameters
            X_test_scaled, y_test = self.optimize_hyperparameters(df, split_info, features)
            
            # Compare results
            recommendation = self.compare_results()
            
            # Save results
            self.save_results(recommendation, X_test_scaled, y_test)
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETE ✓")
            logger.info("="*80)
            logger.info("\nNext Steps:")
            logger.info("  1. Review logs/optimization_results.txt")
            logger.info("  2. Check if optimization helped (expected: 0-3% improvement)")
            logger.info("  3. Use best model for final testing and live trading")
            
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {str(e)}", exc_info=True)
            raise

# ===========================================================================================
# MAIN
# ===========================================================================================

if __name__ == "__main__":
    optimizer = ModelOptimizer()
    optimizer.run_full_pipeline()
