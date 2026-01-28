# Updated: 2026-01-28
"""
Feature engineering utilities.
Creates lag features, rolling statistics, and prepares data for modeling.

This module provides comprehensive feature engineering capabilities for time-series
energy consumption data, including lag features for temporal dependencies,
rolling window statistics for trend analysis, and proper train/val/test splitting
while maintaining temporal ordering.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import warnings
# TODO: Add unit tests
warnings.filterwarnings('ignore')


def create_lag_features(df: pd.DataFrame, 
                       columns: List[str], 
                       lag_intervals: List[int]) -> pd.DataFrame:
    """
    Create lag features for specified columns.
    
    Lag features capture temporal dependencies by including past values as predictors.
    For example, lag_1 represents the value from the previous time step.
    
    Args:
        df: DataFrame with time series data
        columns: List of columns to create lag features for
        lag_intervals: List of lag intervals (e.g., [1, 2, 3, 4] for 1-4 steps back)
    
    Returns:
        DataFrame with additional lag feature columns named '{column}_lag_{interval}'
    """
    print(f"Creating lag features for {len(columns)} columns with lags: {lag_intervals}")
    
    df_with_lags = df.copy()
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
            continue
        
        # Create lag features to capture temporal dependencies
        # Lag features help the model learn patterns from previous time steps
        for lag in lag_intervals:
            lag_col_name = f"{col}_lag_{lag}"
            df_with_lags[lag_col_name] = df[col].shift(lag)
    
    # Count how many lag features were created
    lag_feature_count = len(columns) * len(lag_intervals)
    print(f"Created {lag_feature_count} lag features")
    
    return df_with_lags


def create_rolling_features(df: pd.DataFrame,
                           columns: List[str],
                           window_sizes: List[int]) -> pd.DataFrame:
    """
    Create rolling statistics features.
    
    Computes rolling mean, sum, and standard deviation over specified window sizes
    to capture consumption trends, cumulative patterns, and variability.
    
    Args:
        df: DataFrame with time series data
        columns: List of columns to create rolling features for
        window_sizes: List of window sizes in intervals (e.g., [4, 8, 12])
    
    Returns:
        DataFrame with rolling mean, sum, and std features for each column/window combination
    """
    print(f"Creating rolling features for {len(columns)} columns with windows: {window_sizes}")
    
    df_with_rolling = df.copy()
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
            continue
        
        for window in window_sizes:
            # Rolling mean captures average consumption trends over time
            mean_col_name = f"{col}_rolling_mean_{window}"
            df_with_rolling[mean_col_name] = df[col].rolling(window=window, min_periods=1).mean()
            
            # Rolling sum shows cumulative consumption patterns
            sum_col_name = f"{col}_rolling_sum_{window}"
            df_with_rolling[sum_col_name] = df[col].rolling(window=window, min_periods=1).sum()
            
            # Rolling std captures consumption variability and volatility
            std_col_name = f"{col}_rolling_std_{window}"
            df_with_rolling[std_col_name] = df[col].rolling(window=window, min_periods=1).std()
    
    # Count how many rolling features were created
    rolling_feature_count = len(columns) * len(window_sizes) * 3  # mean, sum, std
    print(f"Created {rolling_feature_count} rolling features")
    
    return df_with_rolling


def create_target_variable(df: pd.DataFrame,
                          target_column: str,
                          forecast_horizon: int = 4) -> pd.DataFrame:
    """
    Create target variable for prediction (future value).
    
    Args:
        df: DataFrame with time series data
        target_column: Column to predict
        forecast_horizon: Number of intervals ahead to predict (default: 4 = 1 hour for 15min intervals)
    
    Returns:
        DataFrame with target variable column
    """
    print(f"Creating target variable: {target_column} at horizon +{forecast_horizon}")
    
    df_with_target = df.copy()
    
    # Shift target column backwards to create future value
    df_with_target['target'] = df[target_column].shift(-forecast_horizon)
    
    return df_with_target


def prepare_features_and_target(df: pd.DataFrame,
                                target_column: str,
                                appliance_columns: List[str],
                                lag_intervals: List[int] = [1, 2, 3, 4],
                                rolling_windows: List[int] = [4, 8, 12],
                                forecast_horizon: int = 4) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Preprocessed DataFrame
        target_column: Column to predict
        appliance_columns: List of appliance columns for feature engineering
        lag_intervals: Lag intervals to create
        rolling_windows: Rolling window sizes
        forecast_horizon: Prediction horizon
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60 + "\n")
    
    # Create lag features
    df_features = create_lag_features(df, appliance_columns, lag_intervals)
    
    # Create rolling features
    df_features = create_rolling_features(df_features, appliance_columns, rolling_windows)
    
    # Create target variable
    df_features = create_target_variable(df_features, target_column, forecast_horizon)
    
    # Separate features and target
    target = df_features['target']
    features = df_features.drop(columns=['target'])
    
    # Remove rows with NaN values (from lag/rolling/target creation)
    valid_indices = target.notna()
    features = features[valid_indices]
    target = target[valid_indices]
    
    # Also remove any remaining NaN in features
    valid_indices = features.notna().all(axis=1)
    features = features[valid_indices]
    target = target[valid_indices]
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Feature columns: {len(features.columns)}")
    
    return features, target


def split_train_val_test(features: pd.DataFrame,
                         target: pd.Series,
                         train_ratio: float = 0.70,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15) -> Tuple:
    """
    Split data into train, validation, and test sets chronologically.
    
    Important: For time series data, we must split chronologically (not randomly)
    to prevent data leakage and ensure realistic evaluation on future data.
    
    Args:
        features: Feature DataFrame
        target: Target Series
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train, val, and test ratios must sum to 1.0"
    
    print(f"\nSplitting data: {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test")
    
    n = len(features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train = features.iloc[:train_end]
    X_val = features.iloc[train_end:val_end]
    X_test = features.iloc[val_end:]
    
    y_train = target.iloc[:train_end]
    y_val = target.iloc[train_end:val_end]
    y_test = target.iloc[val_end:]
    
    print(f"Train set: {len(X_train)} samples ({X_train.index.min()} to {X_train.index.max()})")
    print(f"Val set: {len(X_val)} samples ({X_val.index.min()} to {X_val.index.max()})")
    print(f"Test set: {len(X_test)} samples ({X_test.index.min()} to {X_test.index.max()})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src import config
    from src.data_loader import load_and_prepare_data
    from src.preprocessor import preprocess_pipeline
    
    # Load and preprocess data
    df = load_and_prepare_data(config.RAW_DATA_DIR, home_id=1)
    df_processed, scaler = preprocess_pipeline(df, config.RESAMPLE_INTERVAL)
    
    # Feature engineering
    appliance_cols = [col for col in df_processed.columns if 'Appliance' in col or 'Aggregate' in col]
    appliance_cols = [col for col in appliance_cols if not any(x in col for x in ['lag', 'rolling'])]
    
    if appliance_cols:
        target_col = appliance_cols[0]  # Use first appliance as target
        
        features, target = prepare_features_and_target(
            df_processed,
            target_column=target_col,
            appliance_columns=appliance_cols[:3],  # Use first 3 appliances for features
            lag_intervals=config.LAG_INTERVALS,
            rolling_windows=config.ROLLING_WINDOW_SIZES
        )
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
            features, target,
            config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO
        )
        
        print("\nFeature engineering complete!")
