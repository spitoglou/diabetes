#!/usr/bin/env python3
"""
Simple PyCaret Model Training Script for Diabetes Prediction
Subject: 559, Window: 12 steps, Horizon: 6 steps (30 minutes)

This script directly uses PyCaret to:
1. Load or create the required dataset
2. Setup the regression environment 
3. Compare all available models
4. Select and save the best performing model
5. Evaluate performance metrics
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import time
import uuid

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.helpers.experiment import create_tsfresh_dataframe, create_ds_name
from src.helpers.dataframe import read_df
from src.helpers.diabetes.madex import madex, rmadex
from src.helpers.diabetes.cega import clarke_error_grid

# Import PyCaret
from pycaret.regression import (
    setup, 
    compare_models, 
    finalize_model,
    predict_model,
    save_model,
    pull,
    add_metric,
    get_config
)

def get_available_models():
    """
    Get list of available models in the current PyCaret installation
    """
    from pycaret.regression import models
    
    # Get all available models
    available_models = models()
    available_ids = available_models.index.tolist()
    
    # Define preferred models in order of preference
    preferred_models = [
        'lr',        # Linear Regression
        'ridge',     # Ridge Regression  
        'lasso',     # Lasso Regression
        'en',        # Elastic Net
        'rf',        # Random Forest
        'et',        # Extra Trees
        'gbr',       # Gradient Boosting
        'xgboost',   # XGBoost
        'lightgbm',  # LightGBM
        'catboost',  # CatBoost
        'ada',       # AdaBoost
        'dt',        # Decision Tree
        'knn',       # K-Nearest Neighbors
        'svm',       # Support Vector Machine
    ]
    
    # Filter to only include available models
    available_preferred = [model for model in preferred_models if model in available_ids]
    
    logger.info(f"Available models: {available_preferred}")
    return available_preferred

def clean_dataframe_for_pycaret(df):
    """
    Clean and prepare dataframe for PyCaret following project specifications
    """
    # Remove columns with NaN values
    logger.info(f"Original dataframe shape: {df.shape}")
    df = df.dropna(axis=1)
    logger.info(f"After removing NaN columns: {df.shape}")
    
    # Replace infinite values and remove those columns
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1)
    logger.info(f"After removing infinite value columns: {df.shape}")
    
    # Fix column names for LightGBM compatibility (PyCaret specification)
    import re
    new_names = {col: re.sub(r"[^A-Za-z0-9_]+", "", col) for col in df.columns}
    new_n_list = list(new_names.values())
    
    # Handle duplicate column names
    new_names = {
        col: f"{new_col}_{i}" if new_col in new_n_list[:i] else new_col
        for i, (col, new_col) in enumerate(new_names.items())
    }
    df = df.rename(columns=new_names)
    
    # Replace spaces with underscores in column names (PyCaret specification)
    df.columns = df.columns.str.replace(' ', '_')
    
    # Replace spaces with underscores in categorical/string values (PyCaret specification)
    # This ensures feature name consistency throughout the ML pipeline
    logger.info("Replacing spaces with underscores in categorical values...")
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # Replace spaces with underscores in string/categorical values
            df[col] = df[col].astype(str).str.replace(' ', '_')
        elif df[col].dtype == 'string':
            # Handle pandas string dtype
            df[col] = df[col].str.replace(' ', '_')
    
    logger.info(f"Final cleaned dataframe shape: {df.shape}")
    return df

def load_or_create_dataset(patient_id, window_steps, prediction_horizon):
    """
    Load existing dataset or create new one with tsfresh features
    """
    # Define parameters for training dataset
    train_params = {
        'ohio_no': patient_id,
        'scope': 'train',
        'train_ds_size': 0,
        'window_size': window_steps,
        'prediction_horizon': prediction_horizon,
        'minimal_features': False,
    }
    
    # Define parameters for test dataset
    test_params = {
        'ohio_no': patient_id,
        'scope': 'test',
        'train_ds_size': 0,
        'window_size': window_steps,
        'prediction_horizon': prediction_horizon,
        'minimal_features': False,
    }
    
    logger.info("Loading/creating training dataset...")
    train_df = create_tsfresh_dataframe(train_params)
    train_df = clean_dataframe_for_pycaret(train_df)
    
    logger.info("Loading/creating test dataset...")
    test_df = create_tsfresh_dataframe(test_params)  
    test_df = clean_dataframe_for_pycaret(test_df)
    
    # Align columns between train and test datasets
    train_columns = set(train_df.columns)
    test_columns = set(test_df.columns)
    common_columns = train_columns.intersection(test_columns)
    
    logger.info(f"Train columns: {len(train_columns)}, Test columns: {len(test_columns)}")
    logger.info(f"Common columns: {len(common_columns)}")
    
    # Keep only common columns
    train_df = train_df[list(common_columns)]
    test_df = test_df[list(common_columns)]
    
    return train_df, test_df

def run_pycaret_comparison(train_df, test_df, patient_id, window_steps, prediction_horizon):
    """
    Run PyCaret model comparison and return best models
    """
    logger.info("Setting up PyCaret regression environment...")
    
    # Setup PyCaret environment with migration-compatible parameters
    reg = setup(
        data=train_df,
        target='label',
        train_size=0.8,  # 80% for training, 20% for validation
        feature_selection=True,
        low_variance_threshold=0.0,  # PyCaret 3.x compatible (replaces ignore_low_variance=True)
        ignore_features=[
            'start', 'end', 'start_time', 'end_time', 
            'start_time_of_day', 'end_time_of_day'
        ],
        session_id=1974,  # For reproducibility
        html=False,
        verbose=True,
        use_gpu=False  # Set to True if GPU available and desired
    )
    
    # Custom diabetes-specific metrics will be calculated manually after training
    logger.info("Custom diabetes metrics (MADEX, RMADEX) will be calculated after model comparison")
    
    # Compare all models
    logger.info("Comparing all available models...")
    logger.info("This may take several minutes depending on dataset size...")
    
    start_time = time.time()
    
    # Get available models dynamically
    available_models = get_available_models()
    
    if not available_models:
        raise RuntimeError("No models available for comparison")
    
    logger.info(f"Comparing {len(available_models)} available models...")
    
    # Compare available models
    best_models = compare_models(
        include=available_models,
        sort='RMSE',  # Use standard RMSE for sorting
        n_select=min(3, len(available_models)),  # Return top 3 or all available if less than 3
        verbose=True
    )
    
    comparison_time = time.time() - start_time
    logger.info(f"Model comparison completed in {comparison_time:.2f} seconds")
    
    # Get comparison results
    results_df = pull()
    logger.info("Model comparison results:")
    print(results_df)
    
    # Finalize the best model (trains on full dataset)
    logger.info("Finalizing the best model...")
    if isinstance(best_models, list):
        best_model = best_models[0]
    else:
        best_model = best_models
        
    final_model = finalize_model(best_model)
    
    # Save the best model
    model_name = str(best_model).split('(')[0]
    model_id = str(uuid.uuid4())[:8]
    model_filename = f"models/{patient_id}_{window_steps}_{prediction_horizon}_best_{model_name}_{model_id}"
    
    save_model(final_model, model_filename)
    logger.success(f"Best model saved as: {model_filename}.pkl")
    
    # Evaluate on test data
    logger.info("Evaluating model on test dataset...")
    test_predictions = predict_model(final_model, data=test_df)
    
    # Calculate performance metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    y_true = test_predictions['label']
    y_pred = test_predictions['prediction_label']
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate diabetes-specific metrics
    madex_score = madex(y_true, y_pred)
    rmadex_score = rmadex(y_true, y_pred)
    
    # Clarke Error Grid Analysis
    fig, cega_results = clarke_error_grid(y_true, y_pred, f"Subject {patient_id}")
    cega_dict = dict(zip(['A', 'B', 'C', 'D', 'E'], cega_results))
    
    return {
        'model': final_model,
        'model_name': model_name,
        'model_filename': model_filename,
        'comparison_results': results_df,
        'test_predictions': test_predictions,
        'metrics': {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MADEX': madex_score,
            'RMADEX': rmadex_score,
            'CEGA': cega_dict
        },
        'execution_time': comparison_time
    }

def main():
    """
    Main function to train the best model
    """
    # Configuration
    PATIENT_ID = 559
    WINDOW_STEPS = 12
    PREDICTION_HORIZON = 6
    
    logger.info("="*80)
    logger.info("DIABETES PREDICTION MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Patient ID: {PATIENT_ID}")
    logger.info(f"Window Steps: {WINDOW_STEPS} ({WINDOW_STEPS * 5} minutes of data)")
    logger.info(f"Prediction Horizon: {PREDICTION_HORIZON} steps ({PREDICTION_HORIZON * 5} minutes ahead)")
    logger.info("="*80)
    
    try:
        # Ensure directories exist
        Path("dataframes").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        
        # Load/create datasets
        logger.info("Step 1: Loading datasets...")
        train_df, test_df = load_or_create_dataset(PATIENT_ID, WINDOW_STEPS, PREDICTION_HORIZON)
        
        # Run PyCaret comparison
        logger.info("Step 2: Running PyCaret model comparison...")
        results = run_pycaret_comparison(train_df, test_df, PATIENT_ID, WINDOW_STEPS, PREDICTION_HORIZON)
        
        # Print final results
        logger.info("Step 3: Training completed!")
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"Best Model: {results['model_name']}")
        print(f"Model File: {results['model_filename']}.pkl")
        print(f"Training Time: {results['execution_time']:.2f} seconds")
        print("\nPerformance Metrics on Test Data:")
        for metric, value in results['metrics'].items():
            if metric != 'CEGA':
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  Clarke Error Grid: {value}")
        
        print("\nModel Comparison Summary:")
        print(results['comparison_results'].head().to_string())
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    success = main()
    sys.exit(0 if success else 1)