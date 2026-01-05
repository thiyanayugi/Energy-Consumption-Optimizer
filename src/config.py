"""
Configuration file for Energy Consumption Optimizer.
Contains all hyperparameters, paths, and settings.
"""

import os

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================
RESAMPLE_INTERVAL = '15min'  # Resample to 15-minute intervals
MISSING_VALUE_METHOD = 'ffill'  # Forward fill for missing values
OUTLIER_THRESHOLD = 3  # Standard deviations for outlier removal

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

# Hourly electricity prices (£/kWh) - Example UK tariff
# Peak hours (16:00-19:00): Higher price
# Off-peak hours (00:00-07:00, 23:00-24:00): Lower price
# Standard hours: Medium price
HOURLY_PRICES = [
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,  # 00:00-06:00 (off-peak)
    0.15, 0.15, 0.15, 0.15, 0.15,              # 07:00-11:00 (standard)
    0.15, 0.15, 0.15, 0.15,                    # 12:00-15:00 (standard)
    0.25, 0.25, 0.25,                          # 16:00-18:00 (peak)
    0.15, 0.15, 0.15, 0.15,                    # 19:00-22:00 (standard)
    0.10                                        # 23:00-24:00 (off-peak)
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
