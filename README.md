<!-- Updated: 2026-01-28 -->
# Energy Consumption Optimizer

An end-to-end machine learning system for predicting household energy consumption and optimizing appliance schedules to minimize electricity costs using the REFIT Smart Home dataset.

## Features

- **Data Processing**: Load and preprocess REFIT Smart Home dataset with 15-minute interval resampling
- **Feature Engineering**: Automatic creation of lag features, rolling statistics, and time-based features
- **Predictive Modeling**: XGBoost and LSTM models for accurate energy consumption forecasting
- **Schedule Optimization**: Cost-minimizing appliance scheduling using convex optimization (cvxpy)
# TODO: Add error handling
- **Comprehensive Visualization**: Interactive plots for predictions, schedules, and cost comparisons

## Installation

1. Clone this repository or navigate to the project directory
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This project uses the [REFIT Smart Home dataset](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements). 

The dataset should be placed in the `data/raw/` directory. The project expects CSV files for individual homes (e.g., `CLEAN_House1.csv`).

## Usage

### Jupyter Notebook (Recommended)

Run the interactive notebook for step-by-step execution:

```bash
jupyter notebook energy_optimizer.ipynb
```

### Python Script

Run the complete pipeline from command line:

```bash
python main.py --data_path data/raw --home_id 1
```

Optional arguments:
- `--data_path`: Path to REFIT dataset directory (default: `data/raw`)
- `--home_id`: Home ID to analyze (default: 1)
- `--use_lstm`: Include LSTM model in addition to XGBoost
- `--output_dir`: Directory for saving results (default: `results`)

## Project Structure

```
energy_optimizer/
├── data/
│   ├── raw/              # REFIT dataset files
│   └── processed/        # Processed data files
├── src/
│   ├── config.py         # Configuration parameters
│   ├── data_loader.py    # Data loading utilities
│   ├── preprocessor.py   # Data preprocessing
│   ├── feature_engineer.py  # Feature engineering
│   ├── predictive_model.py  # ML models
│   ├── optimizer.py      # Schedule optimization
│   └── visualizer.py     # Visualization utilities
├── examples/
│   └── example_usage.py  # Usage examples
├── results/              # Output directory
├── energy_optimizer.ipynb  # Main Jupyter notebook
├── main.py               # Standalone script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Key Components

### 1. Data Loading
- Loads REFIT appliance-level data
- Optional weather data integration
- Timestamp alignment and merging

### 2. Preprocessing
- 15-minute interval resampling
- Missing value imputation
- Time-based feature extraction (hour, day of week, weekend flag)
- MinMaxScaler normalization

### 3. Feature Engineering
- Lag features (1-4 intervals)
- Rolling statistics (mean, sum)
- Weather features integration

### 4. Predictive Modeling
- **XGBoost Regressor**: Fast and accurate baseline model
- **LSTM** (optional): Deep learning for time-series patterns
- Evaluation metrics: RMSE, MAE, MAPE

### 5. Schedule Optimization
- Define flexible appliances with constraints:
  - Runtime requirements
  - Time windows (earliest start, latest finish)
- Minimize electricity cost using cvxpy
- Respect appliance constraints and user preferences

### 6. Visualization
- Predicted vs actual consumption plots
- Appliance schedule heatmaps
- Cost comparison charts
- Summary tables with savings metrics

## Example Results

The optimizer typically achieves:
- **10-30% cost savings** through load shifting
- **Peak demand reduction** by avoiding high-price periods
- **Constraint satisfaction** for all appliances

## Advanced Features

- Multi-appliance optimization with conflict resolution
- Time-of-use electricity pricing
- Peak-to-average ratio reduction
- Customizable appliance constraints

## License

This project is provided for educational and research purposes.

## References

- REFIT Smart Home Dataset: Murray, D., Stankovic, L., & Stankovic, V. (2016)
- XGBoost: Chen & Guestrin (2016)
- cvxpy: Diamond & Boyd (2016)
