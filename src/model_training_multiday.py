"""
Multi-Day Model Training and Comparison
Trains separate Random Forest models for different prediction timeframes (1d, 3d, 5d, 7d)
and compares their performance to identify the best prediction window.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve, confusion_matrix)
import joblib

# ============================================================================
# LOGGING SETUP
# ============================================================================

log_dir = 'logs'
Path(log_dir).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'model_training_multiday.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'
CHARTS_DIR = 'data/processed/model_charts'

Path(MODELS_DIR).mkdir(exist_ok=True)
Path(CHARTS_DIR).mkdir(exist_ok=True)


# ============================================================================
# MULTI-DAY MODEL TRAINER
# ============================================================================

class MultiDayModelTrainer:
    """
    Train and evaluate Random Forest models for different prediction timeframes
    """
    
    def __init__(self, test_size=0.15, val_size=0.15):
        """
        Initialize trainer with time-series split configuration
        
        Args:
            test_size (float): Proportion of data for test set
            val_size (float): Proportion of data for validation set
        """
        self.test_size = test_size
        self.val_size = val_size
        self.train_size = 1 - test_size - val_size
        
        self.models = {}  # Store trained models for each timeframe
        self.scalers = {}  # Store scalers for each timeframe
        self.results = {}  # Store results for comparison
        self.feature_importance_dict = {}
        
        logger.info(f"MultiDayModelTrainer initialized: Train={self.train_size:.0%}, Val={self.val_size:.0%}, Test={self.test_size:.0%}")
    
    
    def load_multiday_features(self):
        """
        Load features with all multi-day targets from CSV
        
        Returns:
            pd.DataFrame: Features with all targets
        """
        try:
            filepath = os.path.join(PROCESSED_DIR, 'features_multiday.csv')
            df = pd.read_csv(filepath)
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"Loaded features from {filepath}: {len(df)} rows × {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            raise
    
    
    def get_feature_columns(self, df):
        """
        Extract feature columns (exclude targets, date, ticker, and categorical features)
        
        Args:
            df (pd.DataFrame): Full dataset
            
        Returns:
            list: Feature column names
        """
        exclude_cols = [
            'date', 'ticker',
            'Target_1d', 'Target_1d_1pct',
            'Target_3d', 'Target_3d_1pct',
            'Target_5d', 'Target_5d_1pct',
            'Target_7d', 'Target_7d_1pct',
            'Return_1d', 'Return_3d', 'Return_5d', 'Return_7d',
            'RSI_Category', 'MACD_Crossover'  # Categorical features
        ]
        
        # Get only numeric columns
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        logger.info(f"Using {len(feature_cols)} feature columns")
        return feature_cols
    
    
    def create_time_series_split(self, df):
        """
        Create time-series aware train/val/test split
        
        Args:
            df (pd.DataFrame): Full dataset sorted by date
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test, dates)
        """
        try:
            n = len(df)
            train_end = int(n * self.train_size)
            val_end = train_end + int(n * self.val_size)
            
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
            
            logger.info(f"Time-series split:")
            logger.info(f"  Train: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} samples)")
            logger.info(f"  Val:   {val_df['date'].min()} to {val_df['date'].max()} ({len(val_df)} samples)")
            logger.info(f"  Test:  {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} samples)")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Error creating time-series split: {str(e)}")
            raise
    
    
    def train_model_for_target(self, timeframe, df, feature_cols):
        """
        Train a Random Forest model for a specific timeframe target
        
        Args:
            timeframe (str): '1d', '3d', '5d', or '7d'
            df (pd.DataFrame): Full dataset with targets
            feature_cols (list): Feature column names
            
        Returns:
            dict: Results including model, scaler, metrics, etc.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING {timeframe.upper()} MODEL")
        logger.info(f"{'='*80}")
        
        target_col = f'Target_{timeframe}'
        
        # Check if target exists
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found in data")
            raise ValueError(f"Target column {target_col} not found")
        
        # Create time-series split
        train_df, val_df, test_df = self.create_time_series_split(df)
        
        # Prepare data
        X_train = train_df[feature_cols].copy()
        y_train = train_df[target_col].copy()
        
        X_val = val_df[feature_cols].copy()
        y_val = val_df[target_col].copy()
        
        X_test = test_df[feature_cols].copy()
        y_test = test_df[target_col].copy()
        
        # Log class distribution
        for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            pos_count = y_split.sum()
            pos_pct = (pos_count / len(y_split)) * 100
            logger.info(f"{split_name}: Positive={int(pos_count):,} ({pos_pct:.1f}%), Negative={int(len(y_split)-pos_count):,} ({100-pos_pct:.1f}%)")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info("Feature scaling complete (fit on train, applied to val/test)")
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        logger.info("Model training complete")
        
        # 5-fold time-series cross-validation
        logger.info("Running 5-fold time-series cross-validation...")
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        fold_num = 0
        for train_idx, test_idx in tscv.split(X_train_scaled):
            fold_num += 1
            X_cv_train = X_train_scaled[train_idx]
            y_cv_train = y_train.iloc[train_idx]
            X_cv_test = X_train_scaled[test_idx]
            y_cv_test = y_train.iloc[test_idx]
            
            cv_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=50,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            cv_model.fit(X_cv_train, y_cv_train)
            cv_acc = cv_model.score(X_cv_test, y_cv_test)
            cv_scores.append(cv_acc)
            logger.info(f"  Fold {fold_num}/5: Accuracy = {cv_acc:.4f}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        logger.info(f"CV Results: Mean Accuracy = {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        # Evaluate on validation set
        logger.info("\nValidation Set Evaluation:")
        y_val_pred = model.predict(X_val_scaled)
        y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred, zero_division=0)
        val_rec = recall_score(y_val, y_val_pred, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
        val_auc = roc_auc_score(y_val, y_val_proba)
        
        logger.info(f"  Accuracy:  {val_acc:.4f}")
        logger.info(f"  Precision: {val_prec:.4f}")
        logger.info(f"  Recall:    {val_rec:.4f}")
        logger.info(f"  F1-Score:  {val_f1:.4f}")
        logger.info(f"  ROC-AUC:   {val_auc:.4f}")
        
        # Evaluate on test set
        logger.info("\nTest Set Evaluation (FINAL PERFORMANCE):")
        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec = precision_score(y_test, y_test_pred, zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        logger.info(f"  Accuracy:  {test_acc:.4f}")
        logger.info(f"  Precision: {test_prec:.4f}")
        logger.info(f"  Recall:    {test_rec:.4f}")
        logger.info(f"  F1-Score:  {test_f1:.4f}")
        logger.info(f"  ROC-AUC:   {test_auc:.4f}")
        
        # Confusion matrices
        val_cm = confusion_matrix(y_val, y_val_pred)
        test_cm = confusion_matrix(y_test, y_test_pred)
        
        logger.info(f"\nValidation Confusion Matrix:\n{val_cm}")
        logger.info(f"Test Confusion Matrix:\n{test_cm}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        logger.info(f"\nTop 10 Features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        # Store results
        results = {
            'timeframe': timeframe,
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'val_accuracy': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1': val_f1,
            'val_auc': val_auc,
            'test_accuracy': test_acc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'val_cm': val_cm,
            'test_cm': test_cm,
            'y_test': y_test,
            'y_test_proba': y_test_proba,
            'feature_importance': feature_importance,
            'train_dates': (train_df['date'].min(), train_df['date'].max()),
            'val_dates': (val_df['date'].min(), val_df['date'].max()),
            'test_dates': (test_df['date'].min(), test_df['date'].max())
        }
        
        logger.info(f"{'='*80}\n")
        
        return results
    
    
    def save_models(self):
        """Save all trained models to disk"""
        for timeframe, results in self.results.items():
            model_path = os.path.join(MODELS_DIR, f'rf_model_{timeframe}.pkl')
            scaler_path = os.path.join(MODELS_DIR, f'scaler_{timeframe}.pkl')
            
            joblib.dump(results['model'], model_path)
            joblib.dump(results['scaler'], scaler_path)
            
            logger.info(f"Saved {timeframe} model to {model_path}")
            logger.info(f"Saved {timeframe} scaler to {scaler_path}")
    
    
    def save_comparison_table(self):
        """Save comparison table of all models to text file"""
        filepath = os.path.join(log_dir, 'model_comparison.txt')
        
        with open(filepath, 'w') as f:
            f.write("\n" + "="*100 + "\n")
            f.write("MULTI-DAY MODEL COMPARISON\n")
            f.write("="*100 + "\n\n")
            
            # Create comparison table
            timeframes = ['1d', '3d', '5d', '7d']
            table_data = []
            
            for tf in timeframes:
                if tf in self.results:
                    r = self.results[tf]
                    table_data.append({
                        'Timeframe': tf,
                        'CV Acc': f"{r['cv_mean']:.4f}",
                        'Val Acc': f"{r['val_accuracy']:.4f}",
                        'Test Acc': f"{r['test_accuracy']:.4f}",
                        'Precision': f"{r['test_precision']:.4f}",
                        'Recall': f"{r['test_recall']:.4f}",
                        'F1': f"{r['test_f1']:.4f}",
                        'ROC-AUC': f"{r['test_auc']:.4f}"
                    })
            
            table_df = pd.DataFrame(table_data)
            f.write(table_df.to_string(index=False))
            f.write("\n\n")
            
            # Detailed results for each model
            f.write("="*100 + "\n")
            f.write("DETAILED RESULTS FOR EACH TIMEFRAME\n")
            f.write("="*100 + "\n\n")
            
            for tf in timeframes:
                if tf in self.results:
                    r = self.results[tf]
                    f.write(f"\n{tf.upper()} MODEL\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Train dates: {r['train_dates'][0]} to {r['train_dates'][1]}\n")
                    f.write(f"Val dates:   {r['val_dates'][0]} to {r['val_dates'][1]}\n")
                    f.write(f"Test dates:  {r['test_dates'][0]} to {r['test_dates'][1]}\n")
                    f.write(f"\nCross-Validation:\n")
                    f.write(f"  Mean Accuracy: {r['cv_mean']:.4f} (+/- {r['cv_std']:.4f})\n")
                    f.write(f"\nValidation Set:\n")
                    f.write(f"  Accuracy:  {r['val_accuracy']:.4f}\n")
                    f.write(f"  Precision: {r['val_precision']:.4f}\n")
                    f.write(f"  Recall:    {r['val_recall']:.4f}\n")
                    f.write(f"  F1-Score:  {r['val_f1']:.4f}\n")
                    f.write(f"  ROC-AUC:   {r['val_auc']:.4f}\n")
                    f.write(f"\nTest Set:\n")
                    f.write(f"  Accuracy:  {r['test_accuracy']:.4f}\n")
                    f.write(f"  Precision: {r['test_precision']:.4f}\n")
                    f.write(f"  Recall:    {r['test_recall']:.4f}\n")
                    f.write(f"  F1-Score:  {r['test_f1']:.4f}\n")
                    f.write(f"  ROC-AUC:   {r['test_auc']:.4f}\n")
                    f.write(f"\nTop 10 Features:\n")
                    for idx, row in r['feature_importance'].head(10).iterrows():
                        f.write(f"  {row['Feature']}: {row['Importance']:.4f}\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("RECOMMENDATION\n")
            f.write("="*100 + "\n\n")
            
            # Find best model
            best_tf = max(self.results.keys(), key=lambda x: self.results[x]['test_accuracy'])
            best_acc = self.results[best_tf]['test_accuracy']
            
            f.write(f"Based on test accuracy, the {best_tf.upper()} model performs best with {best_acc:.2%} accuracy.\n\n")
            
            # Comparison to 1-day model
            if '1d' in self.results:
                acc_1d = self.results['1d']['test_accuracy']
                improvement = best_acc - acc_1d
                improvement_pct = (improvement / acc_1d) * 100
                f.write(f"Improvement over 1-day model: {improvement:+.4f} ({improvement_pct:+.2f}%)\n\n")
            
            # Recommendation for trading
            f.write("TRADING RECOMMENDATION:\n")
            f.write(f"  Use {best_tf.upper()} prediction horizon for trading strategy.\n")
            f.write(f"  Expected accuracy: {best_acc:.2%}\n")
            f.write(f"  Hold period: {best_tf}\n")
        
        logger.info(f"Saved comparison table to {filepath}")
    
    
    def plot_accuracy_comparison(self):
        """Create bar chart comparing accuracy across timeframes"""
        timeframes = []
        accuracies = []
        
        for tf in ['1d', '3d', '5d', '7d']:
            if tf in self.results:
                timeframes.append(tf)
                accuracies.append(self.results[tf]['test_accuracy'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#FF6B6B' if acc < 0.50 else '#FFA500' if acc < 0.53 else '#4CAF50' for acc in accuracies]
        bars = ax.bar(timeframes, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.2%}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.axhline(0.50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Random (50%)')
        ax.set_ylim([0.45, max(accuracies) + 0.03])
        ax.set_xlabel('Prediction Timeframe', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison Across Timeframes', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(CHARTS_DIR, 'accuracy_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved accuracy comparison chart to {filepath}")
        plt.close()
    
    
    def plot_confusion_matrices(self):
        """Create 2x2 grid of confusion matrices for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Confusion Matrices - Test Set Performance', fontsize=16, fontweight='bold', y=1.00)
        
        timeframes = ['1d', '3d', '5d', '7d']
        for idx, (ax, tf) in enumerate(zip(axes.flat, timeframes)):
            if tf in self.results:
                cm = self.results[tf]['test_cm']
                
                # Plot confusion matrix
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                           cbar=False, square=True, 
                           xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
                
                acc = self.results[tf]['test_accuracy']
                ax.set_title(f'{tf.upper()} Model\nAccuracy: {acc:.2%}', fontweight='bold', fontsize=12)
                ax.set_ylabel('True Label', fontweight='bold')
                ax.set_xlabel('Predicted Label', fontweight='bold')
        
        plt.tight_layout()
        filepath = os.path.join(CHARTS_DIR, 'confusion_matrices_multiday.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrices to {filepath}")
        plt.close()
    
    
    def plot_roc_curves(self):
        """Create overlaid ROC curves for all models"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = {'1d': '#FF6B6B', '3d': '#FFA500', '5d': '#4CAF50', '7d': '#2196F3'}
        
        for tf in ['1d', '3d', '5d', '7d']:
            if tf in self.results:
                y_test = self.results[tf]['y_test']
                y_proba = self.results[tf]['y_test_proba']
                auc = self.results[tf]['test_auc']
                
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                ax.plot(fpr, tpr, color=colors[tf], linewidth=2.5, 
                       label=f'{tf.upper()} (AUC={auc:.3f})')
        
        # Random classifier line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random (AUC=0.5)')
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Multi-Day Model Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(CHARTS_DIR, 'roc_curves_multiday.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curves to {filepath}")
        plt.close()
    
    
    def run_full_pipeline(self):
        """Execute complete training pipeline for all timeframes"""
        logger.info("\n" + "="*80)
        logger.info("STARTING MULTI-DAY MODEL TRAINING PIPELINE")
        logger.info("="*80 + "\n")
        
        # Load data
        df = self.load_multiday_features()
        feature_cols = self.get_feature_columns(df)
        
        # Train models for each timeframe
        for timeframe in ['1d', '3d', '5d', '7d']:
            try:
                results = self.train_model_for_target(timeframe, df, feature_cols)
                self.results[timeframe] = results
            except Exception as e:
                logger.error(f"Failed to train {timeframe} model: {str(e)}")
                continue
        
        # Save models and results
        self.save_models()
        self.save_comparison_table()
        
        # Create visualizations
        self.plot_accuracy_comparison()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        
        # Print final summary
        self.print_summary()
        
        logger.info("="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
    
    
    def print_summary(self):
        """Print final summary of all models"""
        print("\n" + "="*100)
        print("MULTI-DAY MODEL TRAINING SUMMARY")
        print("="*100)
        
        # Summary table
        print("\nModel Performance Comparison:")
        print("-"*100)
        print(f"{'Timeframe':<12} {'CV Accuracy':<15} {'Val Accuracy':<15} {'Test Accuracy':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
        print("-"*100)
        
        for tf in ['1d', '3d', '5d', '7d']:
            if tf in self.results:
                r = self.results[tf]
                print(f"{tf:<12} {r['cv_mean']:<15.4f} {r['val_accuracy']:<15.4f} {r['test_accuracy']:<15.4f} {r['test_precision']:<12.4f} {r['test_recall']:<12.4f} {r['test_f1']:<12.4f} {r['test_auc']:<12.4f}")
        
        print("-"*100)
        
        # Find best model
        if self.results:
            best_tf = max(self.results.keys(), key=lambda x: self.results[x]['test_accuracy'])
            best_acc = self.results[best_tf]['test_accuracy']
            
            print(f"\n🏆 RECOMMENDATION:")
            print(f"   Based on test accuracy, the {best_tf.upper()} model performs best with {best_acc:.2%} accuracy.\n")
            
            # Improvement over 1-day
            if '1d' in self.results:
                acc_1d = self.results['1d']['test_accuracy']
                improvement = best_acc - acc_1d
                improvement_pct = (improvement / acc_1d) * 100
                print(f"   Improvement over 1-day model: {improvement:+.4f} ({improvement_pct:+.2f}%)")
                print(f"   1-day accuracy: {acc_1d:.2%}")
                print(f"   {best_tf.upper()} accuracy: {best_acc:.2%}")
        
        print("\n📁 Files saved:")
        print(f"   - Models: models/rf_model_1d.pkl, rf_model_3d.pkl, rf_model_5d.pkl, rf_model_7d.pkl")
        print(f"   - Comparison: logs/model_comparison.txt")
        print(f"   - Charts: data/processed/model_charts/")
        
        print("\n" + "="*100 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    trainer = MultiDayModelTrainer()
    trainer.run_full_pipeline()
