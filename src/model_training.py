"""
Model Training Module - Time-Series Aware
Trains Random Forest with proper train/val/test splits and evaluation
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve, auc)
from sklearn.model_selection import GridSearchCV

# Configure logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
DB_PATH = Path('data/market_data.db')
MODELS_DIR = Path('models')
PROCESSED_DATA_DIR = Path('data/processed')
CHARTS_DIR = Path('data/processed/model_charts')

MODELS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)



class TimeSeriesAwareTrainer:
    """
    Trains models with time-series aware splits to prevent data leakage
    """
    
    def __init__(self, db_path=DB_PATH, test_size=0.15, val_size=0.15):
        """
        Initialize trainer
        
        Args:
            db_path: Path to SQLite database
            test_size: Fraction for test set (e.g., 0.15 = last 15%)
            val_size: Fraction for validation set (e.g., 0.15 = middle 15%)
            Training set gets: 1 - test_size - val_size = 0.70
        """
        self.db_path = db_path
        self.test_size = test_size
        self.val_size = val_size
        self.train_size = 1 - test_size - val_size
        
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.date_train = None
        self.date_val = None
        self.date_test = None
        
        self.scaler = None
        self.model = None
        self.model_history = None
        self.best_params = None
        self.evaluation_results = {}
        
        logger.info(f"Trainer initialized: Train={self.train_size:.0%}, Val={self.val_size:.0%}, Test={self.test_size:.0%}")
    
    
    def load_data_from_db(self):
        """
        Load all OHLCV data from SQLite database
        
        Returns:
            pd.DataFrame with columns: ticker, date, open, high, low, close, volume
        """
        try:
            logger.info(f"Loading data from {self.db_path}")
            conn = sqlite3.connect(self.db_path)
            query = "SELECT ticker, date, open, high, low, close, volume FROM ohlcv ORDER BY ticker, date"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"Loaded {len(df):,} rows from {df['ticker'].nunique()} tickers")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data from database: {str(e)}")
            raise
    
    
    def load_features(self, market_data):
        """
        Load engineered features and merge with market data
        
        Args:
            market_data: DataFrame from load_data_from_db()
            
        Returns:
            pd.DataFrame with features and target
        """
        try:
            logger.info("Loading engineered features")
            tickers = market_data['ticker'].unique()
            
            feature_dfs = []
            for ticker in tickers:
                feature_file = PROCESSED_DATA_DIR / f"{ticker}_features.csv"
                if feature_file.exists():
                    df = pd.read_csv(feature_file)
                    df['date'] = pd.to_datetime(df['date'])
                    feature_dfs.append(df)
                else:
                    logger.warning(f"Feature file not found: {ticker}")
            
            if not feature_dfs:
                raise FileNotFoundError("No feature files found")
            
            combined_features = pd.concat(feature_dfs, ignore_index=True)
            logger.info(f"Loaded features for {len(combined_features):,} records")
            
            return combined_features
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            raise
    
    
    def create_time_series_splits(self, df):
        """
        Create time-series aware train/val/test splits
        Split by DATE, not randomly, to prevent data leakage
        
        Args:
            df: DataFrame with 'date' column (sorted by date)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            logger.info("Creating time-series aware splits")
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Calculate split points
            n_samples = len(df)
            train_end = int(n_samples * self.train_size)
            val_end = train_end + int(n_samples * self.val_size)
            
            # Extract dates for logging
            date_train = df.iloc[:train_end]['date']
            date_val = df.iloc[train_end:val_end]['date']
            date_test = df.iloc[val_end:]['date']
            
            logger.info(f"Train dates: {date_train.min()} to {date_train.max()}")
            logger.info(f"Val dates:   {date_val.min()} to {date_val.max()}")
            logger.info(f"Test dates:  {date_test.min()} to {date_test.max()}")
            logger.info(f"Samples - Train: {len(date_train):,}, Val: {len(date_val):,}, Test: {len(date_test):,}")
            
            # Get feature columns (exclude date, ticker, target)
            feature_cols = [c for c in df.columns if c not in ['date', 'ticker', 'Target_1d']]
            
            # Encode categorical features BEFORE converting to array
            df_encoded = df.copy()
            
            if 'RSI_Category' in df_encoded.columns:
                rsi_mapping = {'oversold': 0, 'neutral': 1, 'overbought': 2}
                df_encoded['RSI_Category'] = df_encoded['RSI_Category'].map(rsi_mapping)
                logger.info("Encoded RSI_Category")
            
            if 'MACD_Crossover' in df_encoded.columns:
                macd_mapping = {'bearish': -1, 'no_cross': 0, 'bullish': 1}
                df_encoded['MACD_Crossover'] = df_encoded['MACD_Crossover'].map(macd_mapping)
                logger.info("Encoded MACD_Crossover")
            
            # Separate features and target
            X = df_encoded[feature_cols].values
            y = df_encoded['Target_1d'].values
            
            # Split data
            self.X_train = X[:train_end]
            self.X_val = X[train_end:val_end]
            self.X_test = X[val_end:]
            
            self.y_train = y[:train_end]
            self.y_val = y[train_end:val_end]
            self.y_test = y[val_end:]
            
            self.date_train = date_train
            self.date_val = date_val
            self.date_test = date_test
            
            # Check class distribution
            for name, y_set in [('train', self.y_train), ('val', self.y_val), ('test', self.y_test)]:
                n_pos = (y_set == 1).sum()
                n_neg = (y_set == 0).sum()
                pct_pos = (n_pos / len(y_set)) * 100
                logger.info(f"{name.upper()}: Positive={n_pos:,} ({pct_pos:.1f}%), Negative={n_neg:,} ({100-pct_pos:.1f}%)")
            
            return feature_cols
            
        except Exception as e:
            logger.error(f"Error creating splits: {str(e)}")
            raise
    
    
    def scale_features(self):
        """
        Fit scaler on training data, apply to all sets
        """
        try:
            logger.info("Scaling features")
            
            self.scaler = StandardScaler()
            self.scaler.fit(self.X_train)
            
            self.X_train = self.scaler.transform(self.X_train)
            self.X_val = self.scaler.transform(self.X_val)
            self.X_test = self.scaler.transform(self.X_test)
            
            logger.info("Feature scaling complete")
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise
    
    
    def train_random_forest(self, n_estimators=100, max_depth=10, min_samples_split=50):
        """
        Train Random Forest with class weight balancing
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
        """
        try:
            logger.info(f"Training Random Forest: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
            
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                class_weight='balanced',  # Handle class imbalance
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            
            self.model.fit(self.X_train, self.y_train)
            logger.info("Model training complete")
            
        except Exception as e:
            logger.error(f"Error training Random Forest: {str(e)}")
            raise
    
    
    def train_with_time_series_cv(self, n_splits=5):
        """
        Train with time-series cross-validation to validate on unseen future data
        
        Args:
            n_splits: Number of CV folds
        """
        try:
            logger.info(f"Training with {n_splits}-fold Time Series Cross-Validation")
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            fold_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(self.X_train)):
                X_fold_train = self.X_train[train_idx]
                X_fold_test = self.X_train[test_idx]
                y_fold_train = self.y_train[train_idx]
                y_fold_test = self.y_train[test_idx]
                
                fold_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=50,
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42
                )
                
                fold_model.fit(X_fold_train, y_fold_train)
                fold_pred = fold_model.predict(X_fold_test)
                fold_acc = accuracy_score(y_fold_test, fold_pred)
                fold_scores.append(fold_acc)
                
                logger.info(f"  Fold {fold+1}/{n_splits}: Accuracy = {fold_acc:.4f}")
            
            mean_cv_score = np.mean(fold_scores)
            std_cv_score = np.std(fold_scores)
            logger.info(f"CV Results: Mean Accuracy = {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
            
            self.evaluation_results['cv_mean'] = mean_cv_score
            self.evaluation_results['cv_std'] = std_cv_score
            
        except Exception as e:
            logger.error(f"Error in time series CV: {str(e)}")
            raise
    
    
    def hyperparameter_tuning(self):
        """
        Grid search for best hyperparameters
        """
        try:
            logger.info("Starting hyperparameter tuning (Grid Search)")
            
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [30, 50, 100]
            }
            
            grid_search = GridSearchCV(
                RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42),
                param_grid,
                cv=3,  # 3-fold CV
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            logger.info("Grid search running (this may take a few minutes)...")
            grid_search.fit(self.X_train, self.y_train)
            
            self.best_params = grid_search.best_params_
            self.model = grid_search.best_estimator_
            
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")
            
            self.evaluation_results['grid_search_best_score'] = grid_search.best_score_
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            logger.warning("Continuing with default parameters")
    
    
    def evaluate_model(self):
        """
        Evaluate model on validation and test sets
        """
        try:
            logger.info("Evaluating model on validation and test sets")
            
            for set_name, X, y in [('Validation', self.X_val, self.y_val), 
                                    ('Test', self.X_test, self.y_test)]:
                
                # Predictions
                y_pred = self.model.predict(X)
                y_pred_proba = self.model.predict_proba(X)[:, 1]
                
                # Metrics
                acc = accuracy_score(y, y_pred)
                prec = precision_score(y, y_pred)
                rec = recall_score(y, y_pred)
                f1 = f1_score(y, y_pred)
                roc_auc = roc_auc_score(y, y_pred_proba)
                
                logger.info(f"\n{set_name} Set Evaluation:")
                logger.info(f"  Accuracy:  {acc:.4f}")
                logger.info(f"  Precision: {prec:.4f}")
                logger.info(f"  Recall:    {rec:.4f}")
                logger.info(f"  F1-Score:  {f1:.4f}")
                logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
                
                # Store results
                self.evaluation_results[f'{set_name.lower()}_accuracy'] = acc
                self.evaluation_results[f'{set_name.lower()}_precision'] = prec
                self.evaluation_results[f'{set_name.lower()}_recall'] = rec
                self.evaluation_results[f'{set_name.lower()}_f1'] = f1
                self.evaluation_results[f'{set_name.lower()}_roc_auc'] = roc_auc
                
                # Confusion matrix
                cm = confusion_matrix(y, y_pred)
                logger.info(f"  Confusion Matrix:\n{cm}")
                
                self.evaluation_results[f'{set_name.lower()}_cm'] = cm
                self.evaluation_results[f'{set_name.lower()}_y_pred'] = y_pred
                self.evaluation_results[f'{set_name.lower()}_y_pred_proba'] = y_pred_proba
                self.evaluation_results[f'{set_name.lower()}_y'] = y
        
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    
    def get_feature_importance(self, top_n=10, feature_cols=None):
        """
        Extract top N most important features
        
        Args:
            top_n: Number of top features to show
            feature_cols: List of feature names
        """
        try:
            logger.info(f"Extracting top {top_n} important features")
            
            importances = self.model.feature_importances_
            
            if feature_cols is None:
                feature_cols = [f"Feature_{i}" for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            top_features = importance_df.head(top_n)
            logger.info(f"Top {top_n} features:")
            for idx, row in top_features.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            self.evaluation_results['feature_importance'] = importance_df
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise
    
    
    def save_model(self):
        """
        Save trained model and scaler
        """
        try:
            logger.info("Saving model artifacts")
            
            model_path = MODELS_DIR / 'rf_model.pkl'
            scaler_path = MODELS_DIR / 'feature_scaler.pkl'
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"Model saved: {model_path}")
            logger.info(f"Scaler saved: {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    
    def save_evaluation_report(self, feature_cols):
        """
        Save evaluation metrics to text file
        """
        try:
            logger.info("Saving evaluation report")
            
            report_path = log_dir / 'model_evaluation.txt'
            
            with open(report_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("MODEL EVALUATION REPORT\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                
                # Dataset info
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Training samples:   {len(self.X_train):,} ({self.train_size:.0%})\n")
                f.write(f"Validation samples: {len(self.X_val):,} ({self.val_size:.0%})\n")
                f.write(f"Test samples:       {len(self.X_test):,} ({self.test_size:.0%})\n")
                f.write(f"Total features:     {len(feature_cols)}\n\n")
                
                # Cross-validation results
                if 'cv_mean' in self.evaluation_results:
                    f.write("CROSS-VALIDATION RESULTS:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Mean CV Accuracy: {self.evaluation_results['cv_mean']:.4f}\n")
                    f.write(f"Std CV Accuracy:  {self.evaluation_results['cv_std']:.4f}\n\n")
                
                # Validation results
                f.write("VALIDATION SET EVALUATION:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Accuracy:  {self.evaluation_results.get('validation_accuracy', 0):.4f}\n")
                f.write(f"Precision: {self.evaluation_results.get('validation_precision', 0):.4f}\n")
                f.write(f"Recall:    {self.evaluation_results.get('validation_recall', 0):.4f}\n")
                f.write(f"F1-Score:  {self.evaluation_results.get('validation_f1', 0):.4f}\n")
                f.write(f"ROC-AUC:   {self.evaluation_results.get('validation_roc_auc', 0):.4f}\n\n")
                
                # Test results
                f.write("TEST SET EVALUATION (FINAL PERFORMANCE):\n")
                f.write("-" * 80 + "\n")
                f.write(f"Accuracy:  {self.evaluation_results.get('test_accuracy', 0):.4f}\n")
                f.write(f"Precision: {self.evaluation_results.get('test_precision', 0):.4f}\n")
                f.write(f"Recall:    {self.evaluation_results.get('test_recall', 0):.4f}\n")
                f.write(f"F1-Score:  {self.evaluation_results.get('test_f1', 0):.4f}\n")
                f.write(f"ROC-AUC:   {self.evaluation_results.get('test_roc_auc', 0):.4f}\n\n")
                
                # Model parameters
                if self.best_params:
                    f.write("BEST HYPERPARAMETERS (after tuning):\n")
                    f.write("-" * 80 + "\n")
                    for param, value in self.best_params.items():
                        f.write(f"{param}: {value}\n")
                    f.write("\n")
            
            logger.info(f"Evaluation report saved: {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation report: {str(e)}")
    
    
    def save_feature_importance_csv(self, feature_cols):
        """
        Save feature importance to CSV
        """
        try:
            logger.info("Saving feature importance to CSV")
            
            importance_df = self.evaluation_results.get('feature_importance')
            if importance_df is not None:
                csv_path = MODELS_DIR / 'feature_importance.csv'
                importance_df.to_csv(csv_path, index=False)
                logger.info(f"Feature importance saved: {csv_path}")
        
        except Exception as e:
            logger.error(f"Error saving feature importance CSV: {str(e)}")
    
    
    def plot_confusion_matrix(self):
        """
        Plot confusion matrices for validation and test sets
        """
        try:
            logger.info("Creating confusion matrix visualizations")
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            for idx, set_name in enumerate(['Validation', 'Test']):
                cm = self.evaluation_results[f'{set_name.lower()}_cm']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                           xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
                axes[idx].set_title(f'{set_name} Set - Confusion Matrix')
                axes[idx].set_ylabel('True Label')
                axes[idx].set_xlabel('Predicted Label')
            
            plt.tight_layout()
            plt.savefig(CHARTS_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            logger.info("Saved: confusion_matrix.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
    
    
    def plot_feature_importance(self, top_n=10):
        """
        Plot top N feature importances
        """
        try:
            logger.info("Creating feature importance visualization")
            
            importance_df = self.evaluation_results.get('feature_importance')
            if importance_df is None:
                logger.warning("Feature importance not available")
                return
            
            top_df = importance_df.head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(top_df)), top_df['importance'].values)
            ax.set_yticks(range(len(top_df)))
            ax.set_yticklabels(top_df['feature'].values)
            ax.set_xlabel('Importance')
            ax.set_title(f'Top {top_n} Feature Importances')
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(CHARTS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
            logger.info("Saved: feature_importance.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
    
    
    def plot_roc_curve(self):
        """
        Plot ROC curves for validation and test sets
        """
        try:
            logger.info("Creating ROC curve visualization")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for set_name in ['Validation', 'Test']:
                y = self.evaluation_results[f'{set_name.lower()}_y']
                y_pred_proba = self.evaluation_results[f'{set_name.lower()}_y_pred_proba']
                
                fpr, tpr, _ = roc_curve(y, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, label=f'{set_name} (AUC={roc_auc:.4f})', linewidth=2)
            
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves')
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(CHARTS_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
            logger.info("Saved: roc_curve.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {str(e)}")
    
    
    def run_full_pipeline(self, use_hyperparameter_tuning=False):
        """
        Run complete training pipeline
        """
        logger.info("="*80)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("="*80)
        
        try:
            # Load data
            market_data = self.load_data_from_db()
            features_df = self.load_features(market_data)
            
            # Create time-series aware splits
            feature_cols = self.create_time_series_splits(features_df)
            
            # Scale features
            self.scale_features()
            
            # Train with time-series CV
            self.train_with_time_series_cv(n_splits=5)
            
            # Train initial model
            self.train_random_forest()
            
            # Hyperparameter tuning (optional)
            if use_hyperparameter_tuning:
                self.hyperparameter_tuning()
                self.train_random_forest(**self.best_params)
            
            # Evaluate
            self.evaluate_model()
            
            # Feature importance
            self.get_feature_importance(top_n=10, feature_cols=feature_cols)
            
            # Save artifacts
            self.save_model()
            self.save_evaluation_report(feature_cols)
            self.save_feature_importance_csv(feature_cols)
            
            # Create visualizations
            self.plot_confusion_matrix()
            self.plot_feature_importance()
            self.plot_roc_curve()
            
            logger.info("="*80)
            logger.info("MODEL TRAINING COMPLETE")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """
    Main entry point
    """
    trainer = TimeSeriesAwareTrainer()
    trainer.run_full_pipeline(use_hyperparameter_tuning=False)  # Set to True for tuning


if __name__ == '__main__':
    main()

def load_model(filepath):
    """
    Load trained model from file
    
    Args:
        filepath (str): Path to model file
        
    Returns:
        Trained model
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


if __name__ == "__main__":
    print("Model Training Module")
    print("Use train_random_forest() or train_gradient_boosting() to train models")
