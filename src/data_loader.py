"""
Data loading utilities for REFIT Smart Home dataset.
Handles loading appliance data, weather data, and merging datasets.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_refit_data(data_path: str, home_id: int = 1) -> pd.DataFrame:
    """
    Load REFIT Smart Home dataset for a specific home.
    
    Args:
        data_path: Path to directory containing REFIT CSV files
        home_id: Home ID to load (1-21)
    
    Returns:
        DataFrame with timestamp index and appliance power columns
    """
    # Try different file formats
    filename = f"CLEAN_House{home_id}.csv"
    filepath = os.path.join(data_path, filename)
    
    # Check if data is in REFIT_TIME_SERIES_VALUES subdirectory
    if not os.path.exists(filepath):
        filepath = os.path.join(data_path, "REFIT_TIME_SERIES_VALUES", filename)
    
    # If individual house files don't exist, try the combined XML-based format
    if not os.path.exists(filepath):
        combined_file = os.path.join(data_path, "REFIT_TIME_SERIES_VALUES", "REFIT_TIME_SERIES_VALUES.csv")
        if os.path.exists(combined_file):
            print(f"Loading from combined REFIT file...")
            print(f"Note: This format contains all homes. Filtering for House {home_id}...")
            
            # Load the combined file
            # The format has columns: TimeSeriesVariable/@id, dateTime, data
            # We need to map TimeSeriesVariable IDs to houses and appliances
            
            print("Loading FULL dataset (this may take a few minutes)...")
            df = pd.read_csv(combined_file)  # Load ALL data
            
            print(f"Loaded {len(df)} total rows from combined file")
            
            # Filter to only first 10 TimeSeriesVariables (represents one house)
            # TimeSeriesVariable1-10 = House 1 (Aggregate + 9 appliances)
            df = df[df['TimeSeriesVariable/@id'].str.contains('TimeSeriesVariable[1-9]$|TimeSeriesVariable10$', regex=True)]
            
            print(f"Filtered to {len(df)} rows for House 1 appliances")
            
            # Convert to wide format
            df['timestamp'] = pd.to_datetime(df['dateTime'])
            df = df.pivot_table(
                index='timestamp',
                columns='TimeSeriesVariable/@id',
                values='data',
                aggfunc='first'
            )
            
            # Rename columns to be more descriptive
            # TimeSeriesVariable1 = Aggregate, TimeSeriesVariable2-10 = Appliances
            column_mapping = {
                'TimeSeriesVariable1': 'Aggregate',
                'TimeSeriesVariable2': 'Appliance1',
                'TimeSeriesVariable3': 'Appliance2',
                'TimeSeriesVariable4': 'Appliance3',
                'TimeSeriesVariable5': 'Appliance4',
                'TimeSeriesVariable6': 'Appliance5',
                'TimeSeriesVariable7': 'Appliance6',
                'TimeSeriesVariable8': 'Appliance7',
                'TimeSeriesVariable9': 'Appliance8',
                'TimeSeriesVariable10': 'Appliance9',
            }
            
            # Rename available columns
            df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
            
            df.sort_index(inplace=True)
            
            print(f"Processed {len(df)} records from {df.index.min()} to {df.index.max()}")
            print(f"Columns: {list(df.columns)}")
            
            return df
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"REFIT data file not found. Tried:\n"
            f"  - {os.path.join(data_path, filename)}\n"
            f"  - {os.path.join(data_path, 'REFIT_TIME_SERIES_VALUES', filename)}\n"
            f"  - {os.path.join(data_path, 'REFIT_TIME_SERIES_VALUES', 'REFIT_TIME_SERIES_VALUES.csv')}"
        )


def load_weather_data(weather_path: str) -> Optional[pd.DataFrame]:
    """
    Load weather/climate data if available.
    
    Args:
        weather_path: Path to weather data CSV file
    
    Returns:
        DataFrame with timestamp index and weather features, or None if not available
    """
    if not os.path.exists(weather_path):
        print(f"Weather data not found at {weather_path}. Proceeding without weather data.")
        return None
    
    print(f"Loading weather data from {weather_path}...")
    
    try:
        df = pd.read_csv(weather_path)
        
        # Assume weather data has a timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            # Try to find any datetime-like column
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                df['timestamp'] = pd.to_datetime(df[date_cols[0]])
            else:
                raise ValueError("No timestamp column found in weather data")
        
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"Loaded {len(df)} weather records")
        print(f"Weather features: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        print(f"Error loading weather data: {e}")
        print("Proceeding without weather data.")
        return None


def merge_datasets(appliance_df: pd.DataFrame, 
                   weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Merge appliance data with weather data.
    
    Args:
        appliance_df: DataFrame with appliance power data
        weather_df: Optional DataFrame with weather data
    
    Returns:
        Merged DataFrame with all features
    """
    if weather_df is None:
        print("No weather data to merge. Using appliance data only.")
        return appliance_df.copy()
    
    print("Merging appliance and weather data...")
    
    # Merge on timestamp index using nearest time matching
    merged_df = pd.merge_asof(
        appliance_df.reset_index().sort_values('timestamp'),
        weather_df.reset_index().sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('1H')  # Allow 1 hour tolerance for matching
    )
    
    merged_df.set_index('timestamp', inplace=True)
    
    print(f"Merged dataset has {len(merged_df)} records and {len(merged_df.columns)} features")
    
    return merged_df


def get_appliance_subset(df: pd.DataFrame, appliance_columns: list) -> pd.DataFrame:
    """
    Extract a subset of appliances from the dataset.
    
    Args:
        df: Full DataFrame with all appliances
        appliance_columns: List of appliance column names to keep
    
    Returns:
        DataFrame with only specified appliances
    """
    available_cols = [col for col in appliance_columns if col in df.columns]
    
    if not available_cols:
        raise ValueError(f"None of the specified appliances found in dataset. "
                        f"Available columns: {list(df.columns)}")
    
    return df[available_cols].copy()


def load_and_prepare_data(data_path: str, 
                          home_id: int = 1,
                          weather_path: Optional[str] = None,
                          appliance_subset: Optional[list] = None) -> pd.DataFrame:
    """
    Complete data loading pipeline.
    
    Args:
        data_path: Path to REFIT data directory
        home_id: Home ID to load
        weather_path: Optional path to weather data
        appliance_subset: Optional list of appliance columns to keep
    
    Returns:
        Prepared DataFrame ready for preprocessing
    """
    # Load appliance data
    appliance_df = load_refit_data(data_path, home_id)
    
    # Load weather data if provided
    weather_df = None
    if weather_path:
        weather_df = load_weather_data(weather_path)
    
    # Merge datasets
    df = merge_datasets(appliance_df, weather_df)
    
    # Filter appliances if specified
    if appliance_subset:
        df = get_appliance_subset(df, appliance_subset)
    
    return df


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src import config
    
    # Load data
    df = load_and_prepare_data(
        data_path=config.RAW_DATA_DIR,
        home_id=1
    )
    
    print("\nData summary:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nBasic statistics:")
    print(df.describe())
