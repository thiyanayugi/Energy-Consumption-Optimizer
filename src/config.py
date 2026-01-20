"""
Configuration file for Energy Consumption Optimizer.
Contains all hyperparameters, paths, and settings.

This module centralizes all configuration parameters for the energy optimization
system, including data paths, model hyperparameters, and optimization constraints.

Author: Energy Optimizer Team
Version: 1.0.0
"""

import os
from typing import Dict, List, Any

# ============================================================================
# DATA PATHS
# ============================================================================
# Base directory for all data files (relative to project root)
# Dynamically resolves to project_root/data regardless of execution context
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
# Directory containing raw REFIT dataset CSV files
# Expected to contain CLEAN_HouseX.csv files from REFIT dataset
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
# Directory for preprocessed and feature-engineered data
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
# Directory for saving optimization results and visualizations
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================
# Resample to 15-minute intervals for consistent time-series analysis
RESAMPLE_INTERVAL = '15min'
# Forward fill for missing values - propagates last valid observation forward
MISSING_VALUE_METHOD = 'ffill'
# Standard deviations for outlier removal - values beyond 3σ are considered outliers
OUTLIER_THRESHOLD = 3

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================
LAG_INTERVALS = [1, 2, 3, 4]  # Number of lag features to create
ROLLING_WINDOW_SIZES = [4, 8, 12]  # Rolling window sizes (in intervals)

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Train/Validation/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

# LSTM parameters
LSTM_PARAMS = {
    'units': 64,
    'dropout': 0.2,
    'epochs': 50,
    'batch_size': 32,
    'sequence_length': 24  # Number of time steps to look back
}

# ============================================================================
# APPLIANCE DEFINITIONS
# ============================================================================

# Appliance names in REFIT dataset (House 1)
APPLIANCE_COLUMNS = [
    'Aggregate',
    'Appliance1',  # Fridge
    'Appliance2',  # Freezer (garage)
    'Appliance3',  # Tumble Dryer
    'Appliance4',  # Washing Machine
    'Appliance5',  # Dishwasher
    'Appliance6',  # Computer
    'Appliance7',  # TV Site
    'Appliance8',  # Electric Heater
    'Appliance9',  # Kettle
]

# Flexible appliances for optimization (can be scheduled)
FLEXIBLE_APPLIANCES = {
    'Dishwasher': {
        'column': 'Appliance5',
        'runtime_hours': 2,  # Must run for 2 hours
        'earliest_start': 0,  # Can start from midnight
        'latest_finish': 24,  # Must finish by midnight
        'power_rating': 1200,  # Watts (approximate)
    },
    'Washing Machine': {
        'column': 'Appliance4',
        'runtime_hours': 2,
        'earliest_start': 6,  # Can start from 6 AM
        'latest_finish': 22,  # Must finish by 10 PM
        'power_rating': 2000,
    },
    'Tumble Dryer': {
        'column': 'Appliance3',
        'runtime_hours': 1.5,
        'earliest_start': 6,
        'latest_finish': 23,
        'power_rating': 2500,
    },
}

# ============================================================================
# ELECTRICITY PRICING (Time-of-Use)
# ============================================================================

# Hourly electricity prices (£/kWh) - Realistic UK Economy 7 / Time-of-Use tariff
# Peak hours (16:00-20:00): Very high price (demand surge)
# Off-peak hours (00:00-07:00, 23:00-24:00): Very low price (encourage night usage)
# Standard hours: Medium price
HOURLY_PRICES = [
    0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,  # 00:00-06:00 (off-peak - cheapest)
    0.12, 0.14, 0.16, 0.16, 0.16,              # 07:00-11:00 (morning ramp-up)
    0.18, 0.18, 0.18, 0.20,                    # 12:00-15:00 (daytime)
    0.35, 0.40, 0.40, 0.35,                    # 16:00-19:00 (peak - most expensive!)
    0.22, 0.18, 0.14,                          # 20:00-22:00 (evening wind-down)
    0.10                                        # 23:00-24:00 (late off-peak)
]

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================
ALLOW_SIMULTANEOUS_APPLIANCES = True  # Can multiple appliances run at once?
MAX_SIMULTANEOUS_APPLIANCES = 2  # Maximum number of appliances running simultaneously

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
FIGURE_SIZE = (14, 6)
DPI = 100
COLOR_PALETTE = 'Set2'

# ============================================================================
# RANDOM SEED
# ============================================================================
RANDOM_SEED = 42
