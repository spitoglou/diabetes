#!/usr/bin/env python3
"""
Dataset Check Script for Diabetes Prediction
Check if required datasets exist and show basic information
"""

import sys
import os
import pandas as pd
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_dataset_availability():
    """
    Check if the required datasets for subject 559 with 12 window and 6 horizon exist
    """
    PATIENT_ID = 559
    WINDOW_STEPS = 12
    PREDICTION_HORIZON = 6
    
    # Expected dataset paths
    train_path = f"dataframes/{PATIENT_ID}_train_0_{WINDOW_STEPS}_{PREDICTION_HORIZON}.pkl"
    test_path = f"dataframes/{PATIENT_ID}_test_0_{WINDOW_STEPS}_{PREDICTION_HORIZON}.pkl"
    
    logger.info(f"Checking datasets for Patient {PATIENT_ID}, Window {WINDOW_STEPS}, Horizon {PREDICTION_HORIZON}")
    logger.info(f"Expected training dataset: {train_path}")
    logger.info(f"Expected test dataset: {test_path}")
    
    train_exists = os.path.exists(train_path)
    test_exists = os.path.exists(test_path)
    
    print("\n" + "="*60)
    print("DATASET AVAILABILITY CHECK")
    print("="*60)
    print(f"Training dataset: {'✓ EXISTS' if train_exists else '✗ NOT FOUND'}")
    print(f"Test dataset: {'✓ EXISTS' if test_exists else '✗ NOT FOUND'}")
    
    if train_exists and test_exists:
        try:
            from src.helpers.dataframe import read_df
            
            # Load and examine datasets
            logger.info("Loading datasets to check structure...")
            train_df = read_df(train_path)
            test_df = read_df(test_path)
            
            print(f"\nDATASET INFORMATION:")
            print(f"Training dataset shape: {train_df.shape}")
            print(f"Test dataset shape: {test_df.shape}")
            print(f"Target column (label) in training: {'label' in train_df.columns}")
            print(f"Target column (label) in test: {'label' in test_df.columns}")
            
            if 'label' in train_df.columns:
                print(f"Training label statistics:")
                print(f"  Min: {train_df['label'].min():.2f}")
                print(f"  Max: {train_df['label'].max():.2f}")
                print(f"  Mean: {train_df['label'].mean():.2f}")
                print(f"  Std: {train_df['label'].std():.2f}")
                
            print(f"\nSample columns (first 10):")
            for i, col in enumerate(train_df.columns[:10]):
                print(f"  {i+1}. {col}")
            if len(train_df.columns) > 10:
                print(f"  ... and {len(train_df.columns) - 10} more columns")
                
            print("\nDatasets are ready for training!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            return False
    else:
        print("\nDatasets need to be created before training.")
        print("You can create them by running one of the training scripts.")
        return False

def list_all_available_datasets():
    """
    List all available datasets in the dataframes directory
    """
    dataframes_dir = Path("dataframes")
    
    if not dataframes_dir.exists():
        print("No dataframes directory found.")
        return
        
    pkl_files = list(dataframes_dir.glob("*.pkl"))
    
    if not pkl_files:
        print("No dataset files found in dataframes directory.")
        return
        
    print("\n" + "="*60)
    print("ALL AVAILABLE DATASETS")
    print("="*60)
    
    # Group by patient
    datasets_by_patient = {}
    for file_path in pkl_files:
        filename = file_path.name
        try:
            # Parse filename: patient_scope_trainsize_window_horizon.pkl
            parts = filename.replace('.pkl', '').split('_')
            if len(parts) >= 5:
                patient = parts[0]
                scope = parts[1]
                window = parts[3]
                horizon = parts[4]
                
                if patient not in datasets_by_patient:
                    datasets_by_patient[patient] = []
                    
                datasets_by_patient[patient].append({
                    'scope': scope,
                    'window': window,
                    'horizon': horizon,
                    'file': filename
                })
        except:
            continue
    
    for patient, datasets in sorted(datasets_by_patient.items()):
        print(f"\nPatient {patient}:")
        for dataset in datasets:
            print(f"  {dataset['scope']} | Window: {dataset['window']} | Horizon: {dataset['horizon']} | {dataset['file']}")

def show_models():
    """
    Show available trained models
    """
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("No models directory found.")
        return
        
    model_files = list(models_dir.glob("*.pkl"))
    
    if not model_files:
        print("No trained models found.")
        return
        
    print("\n" + "="*60)
    print("AVAILABLE TRAINED MODELS")
    print("="*60)
    
    for model_file in sorted(model_files):
        try:
            # Get file size
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"{model_file.name} ({size_mb:.1f} MB)")
        except:
            print(f"{model_file.name}")

def main():
    """
    Main function to run all checks
    """
    logger.info("Starting dataset and model availability check...")
    
    # Check specific dataset
    dataset_ready = check_dataset_availability()
    
    # List all datasets
    list_all_available_datasets()
    
    # Show available models
    show_models()
    
    print("\n" + "="*60)
    if dataset_ready:
        print("✓ Ready to train! The required datasets are available.")
        print("Run 'python simple_train_559.py' or 'python train_best_model_559.py' to start training.")
    else:
        print("⚠ Datasets need to be created first.")
        print("The training scripts will automatically create them if they don't exist.")
    print("="*60)

if __name__ == "__main__":
    # Setup simple logging
    logger.remove()
    logger.add(sys.stderr, format="<level>{level}</level>: {message}", level="INFO")
    
    main()