"""
Data preprocessing utilities.
Handles resampling, missing values, time features, normalization, and outlier removal.

This module implements a comprehensive preprocessing pipeline for time-series energy data,
including temporal resampling, missing value imputation, cyclical time feature encoding,
and data normalization for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def resample_data(df: pd.DataFrame, interval: str = '15min') -> pd.DataFrame:
    """
    Resample data to specified time interval.
    
    Args:
        df: DataFrame with datetime index
        interval: Resampling interval (e.g., '15min', '1H')
    
    Returns:
        Resampled DataFrame
    """
    print(f"Resampling data to {interval} intervals...")
    
    # Resample using mean for power values
    # Mean is appropriate for power consumption as it represents average demand over the interval
    df_resampled = df.resample(interval).mean()
    
    print(f"Resampled from {len(df)} to {len(df_resampled)} records")
    
    return df_resampled


def handle_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: DataFrame with potential missing values
        method: Method to use for imputation:
            - 'ffill': Forward fill (propagate last valid observation)
            - 'bfill': Backward fill (use next valid observation)
            - 'interpolate': Time-based linear interpolation
            - 'drop': Remove rows with missing values
    
    Returns:
        DataFrame with missing values handled
    """
    missing_count = df.isnull().sum().sum()
    
    if missing_count == 0:
        print("No missing values found.")
        return df
    
    print(f"Found {missing_count} missing values. Handling with method: {method}")
    
    df_filled = df.copy()
    
    if method == 'ffill':
        # Forward fill propagates last valid observation forward
        df_filled = df_filled.fillna(method='ffill')
        # Fill any remaining NaNs at the start with backward fill
        df_filled = df_filled.fillna(method='bfill')
    elif method == 'bfill':
        df_filled = df_filled.fillna(method='bfill')
        df_filled = df_filled.fillna(method='ffill')
    elif method == 'interpolate':
        # Time-based interpolation for smooth transitions
        df_filled = df_filled.interpolate(method='time')
    elif method == 'drop':
        df_filled = df_filled.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    remaining_missing = df_filled.isnull().sum().sum()
    print(f"Missing values after handling: {remaining_missing}")
    
    return df_filled


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to the dataset.
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        DataFrame with additional time features
    """
    print("Adding time-based features...")
    
    df_with_time = df.copy()
    
    # Extract time features
    df_with_time['hour'] = df_with_time.index.hour
    df_with_time['day_of_week'] = df_with_time.index.dayofweek
    df_with_time['day_of_month'] = df_with_time.index.day
    df_with_time['month'] = df_with_time.index.month
    df_with_time['is_weekend'] = (df_with_time['day_of_week'] >= 5).astype(int)
    
    
    # Cyclical encoding for hour (to capture that 23:00 and 00:00 are close)
    # Using sine and cosine ensures smooth transitions at day boundaries
    df_with_time['hour_sin'] = np.sin(2 * np.pi * df_with_time['hour'] / 24)
    df_with_time['hour_cos'] = np.cos(2 * np.pi * df_with_time['hour'] / 24)
    
    # Cyclical encoding for day of week
    df_with_time['day_sin'] = np.sin(2 * np.pi * df_with_time['day_of_week'] / 7)
    df_with_time['day_cos'] = np.cos(2 * np.pi * df_with_time['day_of_week'] / 7)
    
    print(f"Added {8} time features")
    
    return df_with_time


def normalize_data(df: pd.DataFrame, 
                   columns_to_normalize: Optional[list] = None,
                   scaler: Optional[MinMaxScaler] = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalize specified columns using MinMaxScaler.
    
    Args:
        df: DataFrame to normalize
        columns_to_normalize: List of columns to normalize (if None, normalize all numeric columns)
        scaler: Pre-fitted scaler (if None, fit a new one)
    
    Returns:
        Tuple of (normalized DataFrame, fitted scaler)
    """
    print("Normalizing data...")
    
    df_normalized = df.copy()
    
    if columns_to_normalize is None:
        # Normalize all numeric columns except time features
        time_features = ['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
                        'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        columns_to_normalize = [col for col in df.select_dtypes(include=[np.number]).columns 
                               if col not in time_features]
    
    if not columns_to_normalize:
        print("No columns to normalize.")
        return df_normalized, None
    
    if scaler is None:
        scaler = MinMaxScaler()
        df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
        print(f"Fitted scaler and normalized {len(columns_to_normalize)} columns")
    else:
        df_normalized[columns_to_normalize] = scaler.transform(df[columns_to_normalize])
        print(f"Normalized {len(columns_to_normalize)} columns using provided scaler")
    
    return df_normalized, scaler


def remove_outliers(df: pd.DataFrame, 
                    columns: Optional[list] = None,
                    threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers using z-score method.
    
    Args:
        df: DataFrame to process
        columns: Columns to check for outliers (if None, check all numeric columns)
        threshold: Z-score threshold (default: 3 standard deviations)
    
    Returns:
        DataFrame with outliers removed
    """
    print(f"Removing outliers with threshold {threshold} standard deviations...")
    
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate z-scores: number of standard deviations from the mean
    # Z-score = (value - mean) / std_deviation
    # Values with |z-score| > threshold are considered outliers
    z_scores = np.abs((df_clean[columns] - df_clean[columns].mean()) / df_clean[columns].std())
    
    # Keep rows where all columns are within threshold
    mask = (z_scores < threshold).all(axis=1)
    
    removed_count = len(df_clean) - mask.sum()
    df_clean = df_clean[mask]
    
    print(f"Removed {removed_count} outlier records ({removed_count/len(df)*100:.2f}%)")
    
    return df_clean


def preprocess_pipeline(df: pd.DataFrame,
                       resample_interval: str = '15min',
                       missing_method: str = 'ffill',
                       remove_outliers_flag: bool = True,
                       outlier_threshold: float = 3.0,
                       normalize: bool = True) -> Tuple[pd.DataFrame, Optional[MinMaxScaler]]:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Raw DataFrame
        resample_interval: Resampling interval
        missing_method: Method for handling missing values
        remove_outliers_flag: Whether to remove outliers
        outlier_threshold: Z-score threshold for outlier removal
        normalize: Whether to normalize data
    
    Returns:
        Tuple of (preprocessed DataFrame, scaler if normalization was applied)
    """
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Resample
    df_processed = resample_data(df, resample_interval)
    
    # Step 2: Handle missing values
    df_processed = handle_missing_values(df_processed, missing_method)
    
    # Step 3: Remove outliers (before adding time features)
    if remove_outliers_flag:
        df_processed = remove_outliers(df_processed, threshold=outlier_threshold)
    
    # Step 4: Add time features
    df_processed = add_time_features(df_processed)
    
    # Step 5: Normalize
    scaler = None
    if normalize:
        df_processed, scaler = normalize_data(df_processed)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Final dataset shape: {df_processed.shape}")
    print(f"Date range: {df_processed.index.min()} to {df_processed.index.max()}")
    
    return df_processed, scaler


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src import config
    from src.data_loader import load_and_prepare_data
    
    # Load data
    df = load_and_prepare_data(
        data_path=config.RAW_DATA_DIR,
        home_id=1
    )
    
    # Preprocess
    df_processed, scaler = preprocess_pipeline(
        df,
        resample_interval=config.RESAMPLE_INTERVAL,
        missing_method=config.MISSING_VALUE_METHOD,
        outlier_threshold=config.OUTLIER_THRESHOLD
    )
    
    print("\nProcessed data:")
    print(df_processed.head())
    print("\nColumns:")
    print(df_processed.columns.tolist())
