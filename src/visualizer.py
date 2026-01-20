"""
Visualization utilities for energy consumption and optimization results.
Creates plots for predictions, schedules, and cost comparisons.

This module provides comprehensive visualization functions for analyzing
energy consumption patterns, model predictions, optimization schedules,
and cost savings using matplotlib and seaborn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set visualization style for consistent, professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)  # Default figure size
plt.rcParams['figure.dpi'] = 100          # Display resolution


def plot_predictions(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    timestamps: Optional[pd.DatetimeIndex] = None,
                    title: str = "Energy Consumption: Predicted vs Actual",
                    save_path: Optional[str] = None) -> None:
    """
    Plot predicted vs actual energy consumption.
    
    Creates a two-panel visualization: time series comparison and scatter plot
    for assessing prediction accuracy.
    
    Args:
        y_true: True values (actual energy consumption)
        y_pred: Predicted values (model predictions)
        timestamps: Optional timestamps for x-axis (uses indices if None)
        title: Plot title for the time series panel
        save_path: Optional path to save figure (saves as high-res PNG if provided)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Use indices if timestamps not provided
    x_axis = timestamps if timestamps is not None else np.arange(len(y_true))
    
    # Plot 1: Time series comparison
    ax1.plot(x_axis, y_true, label='Actual', alpha=0.7, linewidth=1.5)
    ax1.plot(x_axis, y_pred, label='Predicted', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Energy Consumption')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Add diagonal line (perfect prediction)
    # Points on this line indicate perfect predictions where y_true == y_pred
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    
    ax2.set_xlabel('Actual Energy Consumption')
    ax2.set_ylabel('Predicted Energy Consumption')
    ax2.set_title('Prediction Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_schedule_heatmap(schedule: Dict,
                         appliances_config: Dict,
                         hourly_prices: List[float],
                         title: str = "Optimized Appliance Schedule",
                         save_path: Optional[str] = None) -> None:
    """
    Plot appliance schedule as a heatmap.
    
    Args:
        schedule: Schedule dictionary with appliance ON/OFF states
        appliances_config: Appliance configurations
        hourly_prices: Hourly electricity prices
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Prepare data for heatmap
    appliance_names = list(schedule.keys())
    hours = list(range(24))
    
    # Create matrix for heatmap
    schedule_matrix = np.array([schedule[name] for name in appliance_names])
    
    # Plot 1: Schedule heatmap
    sns.heatmap(schedule_matrix, 
                xticklabels=[f"{h:02d}:00" for h in hours],
                yticklabels=appliance_names,
                cmap='YlOrRd',
                cbar_kws={'label': 'ON/OFF'},
                ax=ax1,
                linewidths=0.5,
                linecolor='gray')
    
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Appliance')
    ax1.set_title(title)
    
    # Plot 2: Electricity prices
    ax2.bar(hours, hourly_prices, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Price (£/kWh)')
    ax2.set_title('Hourly Electricity Prices')
    ax2.set_xticks(hours)
    ax2.set_xticklabels([f"{h:02d}" for h in hours])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_cost_comparison(original_cost: float,
                        optimized_cost: float,
                        appliances_config: Dict,
                        original_schedule: Dict,
                        optimized_schedule: Dict,
                        hourly_prices: List[float],
                        save_path: Optional[str] = None) -> None:
    """
    Plot cost comparison between original and optimized schedules.
    
    Args:
        original_cost: Original total cost
        optimized_cost: Optimized total cost
        appliances_config: Appliance configurations
        original_schedule: Original schedule
        optimized_schedule: Optimized schedule
        hourly_prices: Hourly electricity prices
        save_path: Optional path to save figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Total cost comparison (bar chart)
    costs = [original_cost, optimized_cost]
    labels = ['Original', 'Optimized']
    colors = ['#ff6b6b', '#51cf66']
    
    bars = ax1.bar(labels, costs, color=colors, alpha=0.7)
    ax1.set_ylabel('Total Cost (£)')
    ax1.set_title('Total Electricity Cost Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'£{height:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add savings annotation
    savings = original_cost - optimized_cost
    savings_pct = (savings / original_cost * 100) if original_cost > 0 else 0
    ax1.text(0.5, max(costs) * 0.5, 
            f'Savings: £{savings:.4f}\n({savings_pct:.2f}%)',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=14, fontweight='bold')
    
    # Plot 2: Cost by appliance
    appliance_names = list(appliances_config.keys())
    original_costs_by_appliance = []
    optimized_costs_by_appliance = []
    
    for name, config in appliances_config.items():
        power_kw = config['power_rating'] / 1000.0
        
        # Calculate costs
        orig_hours = np.where(original_schedule[name] == 1)[0]
        opt_hours = np.where(optimized_schedule[name] == 1)[0]
        
        orig_cost = sum(hourly_prices[h] * power_kw for h in orig_hours)
        opt_cost = sum(hourly_prices[h] * power_kw for h in opt_hours)
        
        original_costs_by_appliance.append(orig_cost)
        optimized_costs_by_appliance.append(opt_cost)
    
    x = np.arange(len(appliance_names))
    width = 0.35
    
    ax2.bar(x - width/2, original_costs_by_appliance, width, label='Original', color='#ff6b6b', alpha=0.7)
    ax2.bar(x + width/2, optimized_costs_by_appliance, width, label='Optimized', color='#51cf66', alpha=0.7)
    
    ax2.set_xlabel('Appliance')
    ax2.set_ylabel('Cost (£)')
    ax2.set_title('Cost by Appliance')
    ax2.set_xticks(x)
    ax2.set_xticklabels(appliance_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Hourly power consumption
    hours = list(range(24))
    original_hourly_power = np.zeros(24)
    optimized_hourly_power = np.zeros(24)
    
    for name, config in appliances_config.items():
        power_kw = config['power_rating'] / 1000.0
        original_hourly_power += original_schedule[name] * power_kw
        optimized_hourly_power += optimized_schedule[name] * power_kw
    
    ax3.plot(hours, original_hourly_power, marker='o', label='Original', linewidth=2, color='#ff6b6b')
    ax3.plot(hours, optimized_hourly_power, marker='s', label='Optimized', linewidth=2, color='#51cf66')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Total Power (kW)')
    ax3.set_title('Hourly Power Consumption')
    ax3.set_xticks(hours)
    ax3.set_xticklabels([f"{h:02d}" for h in hours], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Hourly cost
    original_hourly_cost = original_hourly_power * np.array(hourly_prices)
    optimized_hourly_cost = optimized_hourly_power * np.array(hourly_prices)
    
    ax4.bar(np.array(hours) - 0.2, original_hourly_cost, width=0.4, label='Original', color='#ff6b6b', alpha=0.7)
    ax4.bar(np.array(hours) + 0.2, optimized_hourly_cost, width=0.4, label='Optimized', color='#51cf66', alpha=0.7)
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Cost (£)')
    ax4.set_title('Hourly Electricity Cost')
    ax4.set_xticks(hours)
    ax4.set_xticklabels([f"{h:02d}" for h in hours], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_feature_importance(model, feature_names: List[str],
                           top_n: int = 20,
                           save_path: Optional[str] = None) -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        save_path: Optional path to save figure
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importances[indices], color='steelblue', alpha=0.7)
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def create_results_summary(metrics: Dict,
                          savings_metrics: Dict,
                          save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a summary table of all results.
    
    Args:
        metrics: Prediction metrics dictionary
        savings_metrics: Optimization savings metrics
        save_path: Optional path to save CSV
    
    Returns:
        Summary DataFrame
    """
    summary_data = {
        'Metric': [
            'Prediction RMSE',
            'Prediction MAE',
            'Prediction MAPE (%)',
            'Original Cost (£)',
            'Optimized Cost (£)',
            'Absolute Savings (£)',
            'Percent Savings (%)'
        ],
        'Value': [
            f"{metrics.get('RMSE', 0):.4f}",
            f"{metrics.get('MAE', 0):.4f}",
            f"{metrics.get('MAPE', 0):.2f}",
            f"{savings_metrics.get('original_cost', 0):.4f}",
            f"{savings_metrics.get('optimized_cost', 0):.4f}",
            f"{savings_metrics.get('absolute_savings', 0):.4f}",
            f"{savings_metrics.get('percent_savings', 0):.2f}"
        ]
    }
    
    df = pd.DataFrame(summary_data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved summary to {save_path}")
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Visualization module loaded successfully")
    print("Use the functions to create plots for your energy optimization results")
