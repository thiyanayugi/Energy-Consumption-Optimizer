"""
Predictive modeling utilities.
Implements XGBoost and LSTM models for energy consumption prediction.

This module provides machine learning models for time-series energy forecasting,
including gradient boosting (XGBoost) for tabular data and recurrent neural networks
(LSTM) for capturing long-term temporal dependencies.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

# Deep learning imports (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM model will not be available.")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for regression models.
    
    Computes RMSE (Root Mean Squared Error), MAE (Mean Absolute Error),
    and MAPE (Mean Absolute Percentage Error) for model evaluation.
    
    Args:
        y_true: True values (ground truth)
        y_pred: Predicted values (model output)
    
    Returns:
        Dictionary with RMSE, MAE, and MAPE metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    # Filter out zero values to avoid division by zero errors
    # MAPE is expressed as a percentage for interpretability (0-100%)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def train_xgboost_model(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: Optional[pd.DataFrame] = None,
                       y_val: Optional[pd.Series] = None,
                       params: Optional[Dict] = None) -> xgb.XGBRegressor:
    """
    Train XGBoost regression model for energy consumption prediction.
    
    XGBoost is a gradient boosting algorithm that builds an ensemble of decision trees
    to make predictions. It's particularly effective for tabular data with complex patterns.
    
    Args:
        X_train: Training features (preprocessed and engineered)
        y_train: Training target (energy consumption values)
        X_val: Validation features for monitoring training progress (optional)
        y_val: Validation target (optional)
        params: XGBoost hyperparameters (uses defaults if None)
    
    Returns:
        Trained XGBoost model ready for predictions
    """
    print("\n" + "="*60)
    print("TRAINING XGBOOST MODEL")
    print("="*60 + "\n")
    
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    print(f"Model parameters: {params}")
    
    # Create model
    model = xgb.XGBRegressor(**params)
    
    # Prepare evaluation set for monitoring training progress
    # Allows tracking performance on both training and validation data during training
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))
    
    # Train model
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train.values, y_train_pred)
    
    print(f"\nTraining Metrics:")
    print(f"  RMSE: {train_metrics['RMSE']:.4f}")
    print(f"  MAE: {train_metrics['MAE']:.4f}")
    print(f"  MAPE: {train_metrics['MAPE']:.2f}%")
    
    # Evaluate on validation set if provided
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        val_metrics = calculate_metrics(y_val.values, y_val_pred)
        
        print(f"\nValidation Metrics:")
        print(f"  RMSE: {val_metrics['RMSE']:.4f}")
        print(f"  MAE: {val_metrics['MAE']:.4f}")
        print(f"  MAPE: {val_metrics['MAPE']:.2f}%")
    
    print("\n" + "="*60)
    print("XGBOOST TRAINING COMPLETE")
    print("="*60)
    
    return model


def prepare_lstm_data(X: pd.DataFrame, 
                     y: pd.Series, 
                     sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for LSTM model (create sequences).
    
    LSTM models require input data to be in 3D format: (samples, time_steps, features).
    This function creates sliding windows of historical data for sequence learning.
    
    Args:
        X: Features DataFrame
        y: Target Series
        sequence_length: Number of time steps in each sequence
    
    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    X_values = X.values
    y_values = y.values
    
    X_sequences = []
    y_sequences = []
    
    # Create sliding windows of data
    # Each sequence uses 'sequence_length' previous time steps to predict the next value
    # Example: if sequence_length=24, use hours 0-23 to predict hour 24
    for i in range(len(X_values) - sequence_length):
        X_sequences.append(X_values[i:i+sequence_length])
        y_sequences.append(y_values[i+sequence_length])
    
    return np.array(X_sequences), np.array(y_sequences)


def train_lstm_model(X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_val: Optional[pd.DataFrame] = None,
                    y_val: Optional[pd.Series] = None,
                    params: Optional[Dict] = None):
    """
    Train LSTM model for time series prediction.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        params: LSTM parameters
    
    Returns:
        Trained LSTM model or None if TensorFlow not available
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Skipping LSTM model.")
        return None
    
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60 + "\n")
    
    if params is None:
        params = {
            'units': 64,
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'sequence_length': 24
        }
    
    print(f"Model parameters: {params}")
    
    # Prepare sequences
    sequence_length = params['sequence_length']
    X_train_seq, y_train_seq = prepare_lstm_data(X_train, y_train, sequence_length)
    
    print(f"Training sequences shape: {X_train_seq.shape}")
    
    validation_data = None
    if X_val is not None and y_val is not None:
        X_val_seq, y_val_seq = prepare_lstm_data(X_val, y_val, sequence_length)
        validation_data = (X_val_seq, y_val_seq)
        print(f"Validation sequences shape: {X_val_seq.shape}")
    
    # Build LSTM model
    model = Sequential([
        LSTM(params['units'], activation='relu', return_sequences=True, 
             input_shape=(sequence_length, X_train.shape[1])),
        Dropout(params['dropout']),
        LSTM(params['units'] // 2, activation='relu'),
        Dropout(params['dropout']),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("\nModel architecture:")
    model.summary()
    
    
    # Early stopping to prevent overfitting
    # Monitors validation loss and stops training if no improvement for 'patience' epochs
    # Restores the best weights found during training to ensure optimal model performance
    early_stop = EarlyStopping(monitor='val_loss' if validation_data else 'loss', 
                              patience=10, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_data=validation_data,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    y_train_pred = model.predict(X_train_seq, verbose=0).flatten()
    train_metrics = calculate_metrics(y_train_seq, y_train_pred)
    
    print(f"\nTraining Metrics:")
    print(f"  RMSE: {train_metrics['RMSE']:.4f}")
    print(f"  MAE: {train_metrics['MAE']:.4f}")
    print(f"  MAPE: {train_metrics['MAPE']:.2f}%")
    
    if validation_data:
        y_val_pred = model.predict(X_val_seq, verbose=0).flatten()
        val_metrics = calculate_metrics(y_val_seq, y_val_pred)
        
        print(f"\nValidation Metrics:")
        print(f"  RMSE: {val_metrics['RMSE']:.4f}")
        print(f"  MAE: {val_metrics['MAE']:.4f}")
        print(f"  MAPE: {val_metrics['MAPE']:.2f}%")
    
    print("\n" + "="*60)
    print("LSTM TRAINING COMPLETE")
    print("="*60)
    
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, 
                   model_type: str = 'xgboost') -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_type: Type of model ('xgboost' or 'lstm')
    
    Returns:
        Tuple of (predictions, metrics dictionary)
    """
    print(f"\nEvaluating {model_type.upper()} model on test set...")
    
    if model_type == 'lstm' and TENSORFLOW_AVAILABLE:
        # Prepare sequences for LSTM
        sequence_length = 24  # Should match training
        X_test_seq, y_test_seq = prepare_lstm_data(X_test, y_test, sequence_length)
        y_pred = model.predict(X_test_seq, verbose=0).flatten()
        y_true = y_test_seq
    else:
        # XGBoost or other sklearn-compatible model
        y_pred = model.predict(X_test)
        y_true = y_test.values
    
    metrics = calculate_metrics(y_true, y_pred)
    
    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    return y_pred, metrics


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src import config
    from src.data_loader import load_and_prepare_data
    from src.preprocessor import preprocess_pipeline
    from src.feature_engineer import prepare_features_and_target, split_train_val_test
    
    # Load and preprocess
    df = load_and_prepare_data(config.RAW_DATA_DIR, home_id=1)
    df_processed, scaler = preprocess_pipeline(df, config.RESAMPLE_INTERVAL)
    
    # Feature engineering
    appliance_cols = [col for col in df_processed.columns if 'Appliance' in col or 'Aggregate' in col]
    appliance_cols = [col for col in appliance_cols if not any(x in col for x in ['lag', 'rolling'])]
    
    if appliance_cols:
        features, target = prepare_features_and_target(
            df_processed, appliance_cols[0], appliance_cols[:3]
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
            features, target
        )
        
        # Train XGBoost
        xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val, config.XGBOOST_PARAMS)
        
        # Evaluate
        y_pred, metrics = evaluate_model(xgb_model, X_test, y_test, 'xgboost')
