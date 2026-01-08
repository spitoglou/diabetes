#!/usr/bin/env python3
"""
Script to train the best model for diabetes prediction using PyCaret
Subject: 559
Window Steps: 12 (60 minutes of data)
Prediction Horizon: 6 steps (30 minutes ahead)

This script uses the existing Experiment framework to:
1. Create featurized datasets for training and testing
2. Setup PyCaret regression environment
3. Compare multiple models to find the best performing ones
4. Save the best models for deployment
5. Evaluate performance on holdout and unseen data
"""

import sys
import os
from pathlib import Path
from loguru import logger
import time

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.helpers.experiment import Experiment
from src.helpers.dataframe import read_df, save_df

def main():
    """
    Main function to train the best model for subject 559
    """
    # Configuration parameters
    PATIENT_ID = 559
    WINDOW_STEPS = 12  # 12 * 5 minutes = 60 minutes of historical data
    PREDICTION_HORIZON = 6  # 6 * 5 minutes = 30 minutes ahead prediction
    
    logger.info("Starting diabetes prediction model training")
    logger.info(f"Patient ID: {PATIENT_ID}")
    logger.info(f"Window Steps: {WINDOW_STEPS} (representing {WINDOW_STEPS * 5} minutes of data)")
    logger.info(f"Prediction Horizon: {PREDICTION_HORIZON} steps (representing {PREDICTION_HORIZON * 5} minutes ahead)")
    
    try:
        # Create experiment instance
        # Disable Neptune tracking for local training
        # Enable feature selection and use speed level 1 for comprehensive model comparison
        experiment = Experiment(
            patient=PATIENT_ID,
            window=WINDOW_STEPS, 
            horizon=PREDICTION_HORIZON,
            min_per_measure=5,  # 5 minutes between measurements
            best_models_no=3,   # Save top 3 models
            speed=1,            # Comprehensive model comparison (speed=1 includes all models)
            log_type="standard", # Log to stderr
            perform_gap_corrections=True,  # Clean data gaps
            minimal_features=False,  # Use full feature set for better accuracy
            enable_neptune=False  # Disable Neptune for local training
        )
        
        logger.info("Experiment configuration completed")
        
        # Check if datasets already exist
        train_ds_name = f"dataframes/{PATIENT_ID}_train_0_{WINDOW_STEPS}_{PREDICTION_HORIZON}.pkl"
        test_ds_name = f"dataframes/{PATIENT_ID}_test_0_{WINDOW_STEPS}_{PREDICTION_HORIZON}.pkl"
        
        if not os.path.exists(train_ds_name) or not os.path.exists(test_ds_name):
            logger.info("Creating feature datasets...")
            # Create datasets if they don't exist
            from src.helpers.experiment import create_tsfresh_dataframe
            
            # Create training dataset
            train_params = {
                'ohio_no': PATIENT_ID,
                'scope': 'train',
                'train_ds_size': 0,  # Use all available training data
                'window_size': WINDOW_STEPS,
                'prediction_horizon': PREDICTION_HORIZON,
                'minimal_features': False,
            }
            logger.info("Creating training dataset...")
            create_tsfresh_dataframe(train_params)
            
            # Create test dataset
            test_params = {
                'ohio_no': PATIENT_ID,
                'scope': 'test',
                'train_ds_size': 0,  # Use all available test data
                'window_size': WINDOW_STEPS,
                'prediction_horizon': PREDICTION_HORIZON,
                'minimal_features': False,
            }
            logger.info("Creating test dataset...")
            create_tsfresh_dataframe(test_params)
        else:
            logger.info("Found existing datasets, proceeding with training...")
        
        # Run the complete experiment
        logger.info("Starting model training and evaluation...")
        start_time = time.time()
        
        experiment.run_experiment()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.success(f"Training completed successfully!")
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        
        # Print results summary
        print("\n" + "="*80)
        print("TRAINING RESULTS SUMMARY")
        print("="*80)
        
        print(f"\nBest Models Comparison:")
        print(experiment.models_comparison_df.to_string())
        
        print(f"\nHoldout Performance:")
        print(f"  RMSE: {experiment.holdout_rmse:.4f}")
        print(f"  RMADEX: {experiment.holdout_rmadex:.4f}")
        print(f"  Clarke Error Grid: {experiment.holdout_cega_res}")
        
        print(f"\nUnseen Data Performance:")
        print(f"  RMSE: {experiment.unseen_rmse:.4f}")
        print(f"  RMADEX: {experiment.unseen_rmadex:.4f}")
        print(f"  Clarke Error Grid: {experiment.unseen_cega_res}")
        
        # List saved models
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob(f"{PATIENT_ID}_{WINDOW_STEPS}_{PREDICTION_HORIZON}_*.pkl"))
            print(f"\nSaved Models ({len(model_files)} files):")
            for model_file in sorted(model_files):
                print(f"  {model_file}")
        
        print("\n" + "="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def verify_environment():
    """
    Verify that the required environment and dependencies are available
    """
    try:
        import pycaret
        import pandas
        import numpy
        import sklearn
        import tsfresh
        
        logger.info("All required dependencies are available")
        
        # Check if data directories exist
        dataframes_dir = Path("dataframes")
        models_dir = Path("models")
        
        if not dataframes_dir.exists():
            dataframes_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created dataframes directory")
            
        if not models_dir.exists():
            models_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created models directory")
            
        return True
        
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        return False

if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Verify environment
    if not verify_environment():
        logger.error("Environment verification failed. Please install missing dependencies.")
        sys.exit(1)
    
    # Run training
    success = main()
    
    if success:
        logger.success("Model training completed successfully!")
        sys.exit(0)
    else:
        logger.error("Model training failed!")
        sys.exit(1)