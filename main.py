# Updated: 2026-01-28
# Updated: 2026-01-28
# Updated: 2026-01-28
"""
Main script for Energy Consumption Optimizer.
Runs the complete pipeline from data loading to optimization.

This script orchestrates the entire energy optimization workflow including:
data loading, preprocessing, feature engineering, predictive modeling,
schedule optimization, and results visualization.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(__file__))

from src import config
from src.data_loader import load_and_prepare_data
from src.preprocessor import preprocess_pipeline
from src.feature_engineer import prepare_features_and_target, split_train_val_test
from src.predictive_model import train_xgboost_model, train_lstm_model, evaluate_model
from src.optimizer import optimize_schedule, generate_schedule_dataframe, create_summary_table
from src.visualizer import (plot_predictions, plot_schedule_heatmap, plot_cost_comparison,
                            plot_feature_importance, create_results_summary)


def main(data_path: str, home_id: int = 1, use_lstm: bool = False, output_dir: str = "results"):
    """
    Main pipeline for energy consumption optimization.

    Executes a 7-step pipeline: data loading, preprocessing, feature engineering,
    predictive modeling (XGBoost/LSTM), schedule optimization, visualization,
    and results summary generation.

    Args:
        data_path: Path to REFIT dataset directory
        home_id: Home ID to analyze (1-21)
        use_lstm: Whether to train LSTM model in addition to XGBoost
        output_dir: Directory to save results and visualizations
    """
    print("\n" + "="*80)
    print(" "*20 + "ENERGY CONSUMPTION OPTIMIZER")
    print("="*80 + "\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # ========================================================================
    # STEP 1: DATA LOADING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80 + "\n")

    df = load_and_prepare_data(data_path, home_id)

    # ========================================================================
    # STEP 2: DATA PREPROCESSING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: DATA PREPROCESSING")
    print("="*80 + "\n")

    df_processed, scaler = preprocess_pipeline(
        df,
        resample_interval=config.RESAMPLE_INTERVAL,
        missing_method=config.MISSING_VALUE_METHOD,
        remove_outliers_flag=True,
        outlier_threshold=config.OUTLIER_THRESHOLD,
        normalize=True
    )

    # ========================================================================
    # STEP 3: FEATURE ENGINEERING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*80 + "\n")


    # Identify appliance columns (exclude derived features like lag, rolling, time features)
    appliance_cols = [col for col in df_processed.columns
                     if ('Appliance' in col or 'Aggregate' in col)
                     and not any(x in col for x in ['lag', 'rolling', 'hour', 'day', 'sin', 'cos', 'weekend'])]

    print(f"Found {len(appliance_cols)} appliance columns: {appliance_cols}")

    # Use Aggregate as target (total household consumption)
    target_column = 'Aggregate' if 'Aggregate' in appliance_cols else appliance_cols[0]

    # Use subset of appliances for features (to keep model manageable and avoid overfitting)
    feature_appliances = appliance_cols[:5] if len(appliance_cols) > 5 else appliance_cols

    features, target = prepare_features_and_target(
        df_processed,
        target_column=target_column,
        appliance_columns=feature_appliances,
        lag_intervals=config.LAG_INTERVALS,
        rolling_windows=config.ROLLING_WINDOW_SIZES,
        forecast_horizon=4  # Predict 1 hour ahead (4 * 15min intervals)
    )

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        features, target,
        config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO
    )

    # ========================================================================
    # STEP 4: PREDICTIVE MODELING
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: PREDICTIVE MODELING")
    print("="*80 + "\n")

    # Train XGBoost model
    xgb_model = train_xgboost_model(
        X_train, y_train, X_val, y_val,
        params=config.XGBOOST_PARAMS
    )

    # Evaluate XGBoost
    y_pred_xgb, metrics_xgb = evaluate_model(xgb_model, X_test, y_test, 'xgboost')

    # Plot predictions
    plot_predictions(
        y_test.values, y_pred_xgb,
        timestamps=y_test.index,
        title="XGBoost: Energy Consumption Prediction",
        save_path=os.path.join(output_dir, "xgboost_predictions.png")
    )

    # Plot feature importance
    plot_feature_importance(
        xgb_model, X_train.columns.tolist(),
        top_n=20,
        save_path=os.path.join(output_dir, "feature_importance.png")
    )

    # Train LSTM model (optional)
    if use_lstm:
        lstm_model = train_lstm_model(
            X_train, y_train, X_val, y_val,
            params=config.LSTM_PARAMS
        )

        if lstm_model is not None:
            y_pred_lstm, metrics_lstm = evaluate_model(lstm_model, X_test, y_test, 'lstm')

            plot_predictions(
                y_test.values[-len(y_pred_lstm):], y_pred_lstm,
                timestamps=y_test.index[-len(y_pred_lstm):],
                title="LSTM: Energy Consumption Prediction",
                save_path=os.path.join(output_dir, "lstm_predictions.png")
            )

    # ========================================================================
    # STEP 5: APPLIANCE SCHEDULE OPTIMIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: APPLIANCE SCHEDULE OPTIMIZATION")
    print("="*80 + "\n")

    # Calculate original schedule (baseline: run at earliest time)
    original_schedule = {}
    for name, config_app in config.FLEXIBLE_APPLIANCES.items():
        runtime_hours = int(np.ceil(config_app['runtime_hours']))
        earliest = config_app['earliest_start']
        schedule = np.zeros(24)
        schedule[earliest:earliest+runtime_hours] = 1
        original_schedule[name] = schedule

    # Optimize schedule
    optimized_schedule, original_cost, optimized_cost = optimize_schedule(
        config.FLEXIBLE_APPLIANCES,
        config.HOURLY_PRICES,
        config.ALLOW_SIMULTANEOUS_APPLIANCES,
        config.MAX_SIMULTANEOUS_APPLIANCES
    )

    # Generate schedule DataFrame
    schedule_df = generate_schedule_dataframe(
        optimized_schedule,
        config.FLEXIBLE_APPLIANCES,
        config.HOURLY_PRICES
    )

    print("\nOptimized Schedule:")
    print(schedule_df)

    # Save schedule
    schedule_df.to_csv(os.path.join(output_dir, "optimized_schedule.csv"), index=False)

    # Create summary table
    summary_table = create_summary_table(
        config.FLEXIBLE_APPLIANCES,
        original_schedule,
        optimized_schedule,
        config.HOURLY_PRICES
    )

    print("\nSchedule Summary:")
    print(summary_table)

    summary_table.to_csv(os.path.join(output_dir, "schedule_summary.csv"), index=False)

    # ========================================================================
    # STEP 6: VISUALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: VISUALIZATION")
    print("="*80 + "\n")

    # Plot schedule heatmap
    plot_schedule_heatmap(
        optimized_schedule,
        config.FLEXIBLE_APPLIANCES,
        config.HOURLY_PRICES,
        title="Optimized Appliance Schedule",
        save_path=os.path.join(output_dir, "schedule_heatmap.png")
    )

    # Plot cost comparison
    plot_cost_comparison(
        original_cost, optimized_cost,
        config.FLEXIBLE_APPLIANCES,
        original_schedule, optimized_schedule,
        config.HOURLY_PRICES,
        save_path=os.path.join(output_dir, "cost_comparison.png")
    )

    # ========================================================================
    # STEP 7: RESULTS SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: RESULTS SUMMARY")
    print("="*80 + "\n")

    savings_metrics = {
        'original_cost': original_cost,
        'optimized_cost': optimized_cost,
        'absolute_savings': original_cost - optimized_cost,
        'percent_savings': (original_cost - optimized_cost) / original_cost * 100
    }

    results_summary = create_results_summary(
        metrics_xgb,
        savings_metrics,
        save_path=os.path.join(output_dir, "results_summary.csv")
    )

    print("\nFinal Results Summary:")
    print(results_summary)

    print("\n" + "="*80)
    print(" "*25 + "PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {os.path.abspath(output_dir)}")
    print("\nKey Findings:")
    print(f"  • Prediction RMSE: {metrics_xgb['RMSE']:.4f}")
    print(f"  • Prediction MAE: {metrics_xgb['MAE']:.4f}")
    print(f"  • Prediction MAPE: {metrics_xgb['MAPE']:.2f}%")
    print(f"  • Cost Savings: £{savings_metrics['absolute_savings']:.4f} ({savings_metrics['percent_savings']:.2f}%)")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy Consumption Optimizer")
    parser.add_argument("--data_path", type=str, default=config.RAW_DATA_DIR,
                       help="Path to REFIT dataset directory")
    parser.add_argument("--home_id", type=int, default=1,
                       help="Home ID to analyze (1-21)")
    parser.add_argument("--use_lstm", action="store_true",
                       help="Train LSTM model in addition to XGBoost")
    parser.add_argument("--output_dir", type=str, default=config.RESULTS_DIR,
                       help="Directory to save results")

    args = parser.parse_args()

    main(args.data_path, args.home_id, args.use_lstm, args.output_dir)
