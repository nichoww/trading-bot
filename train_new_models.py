#!/usr/bin/env python3
"""
Main training script entry point
Run time-series aware model training with proper splits and evaluation
"""

from pathlib import Path
from src.model_training import TimeSeriesAwareTrainer
import logging

logger = logging.getLogger(__name__)

def main():
    """
    Execute full training pipeline
    """
    print("\n" + "="*80)
    print("TRADING BOT - TIME-SERIES AWARE MODEL TRAINING")
    print("="*80 + "\n")
    
    trainer = TimeSeriesAwareTrainer(
        test_size=0.15,  # Last 15% as test set
        val_size=0.15    # Middle 15% as validation set
                         # First 70% as training set
    )
    
    # Run full pipeline WITHOUT hyperparameter tuning
    # Set to True if you want grid search (slower but may improve accuracy)
    trainer.run_full_pipeline(use_hyperparameter_tuning=False)
    
    print("\n" + "="*80)
    print("Training complete! Check logs/ and models/ directories for results")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
