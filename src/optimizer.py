"""
Appliance schedule optimization using convex optimization.
Minimizes electricity cost while respecting appliance constraints.

This module implements a convex optimization approach using CVXPY to find
the optimal scheduling of flexible appliances (dishwasher, washing machine, etc.)
to minimize electricity costs under time-of-use pricing while respecting
user-defined constraints such as runtime requirements and time windows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import cvxpy as cp
import warnings
warnings.filterwarnings('ignore')


def define_appliance_constraints(appliances_config: Dict) -> Dict:
    """
    Parse and validate appliance constraint definitions.
    
    This function ensures all required configuration parameters are present
    and validates that time windows and runtime constraints are feasible.
    
    Args:
        appliances_config: Dictionary with appliance configurations containing:
            - runtime_hours: Duration appliance must run (hours)
            - earliest_start: Earliest hour appliance can start (0-23)
            - latest_finish: Latest hour appliance must finish by (1-24)
            - power_rating: Appliance power consumption (Watts)
    
    Returns:
        Validated appliance constraints dictionary
        
    Raises:
        ValueError: If required keys are missing or constraints are invalid
    """
    print("Defining appliance constraints...")
    
    for name, config in appliances_config.items():
        required_keys = ['runtime_hours', 'earliest_start', 'latest_finish', 'power_rating']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Appliance '{name}' missing required key: {key}")
        
        # Validate constraints
        if config['earliest_start'] < 0 or config['earliest_start'] >= 24:
            raise ValueError(f"Invalid earliest_start for {name}")
        if config['latest_finish'] <= 0 or config['latest_finish'] > 24:
            raise ValueError(f"Invalid latest_finish for {name}")
        if config['runtime_hours'] <= 0:
            raise ValueError(f"Invalid runtime_hours for {name}")
        if config['earliest_start'] + config['runtime_hours'] > config['latest_finish']:
            raise ValueError(f"Impossible time window for {name}")
    
    print(f"Validated {len(appliances_config)} appliances")
    return appliances_config


def optimize_schedule(appliances_config: Dict,
                     hourly_prices: List[float],
                     allow_simultaneous: bool = True,
                     max_simultaneous: int = 2) -> Tuple[Dict, float, float]:
    """
    Optimize appliance schedule to minimize electricity cost.
    
    Args:
        appliances_config: Dictionary with appliance configurations
        hourly_prices: List of 24 hourly electricity prices (£/kWh)
        allow_simultaneous: Whether appliances can run simultaneously
        max_simultaneous: Maximum number of appliances running at once
    
    Returns:
        Tuple of (schedule dict, original cost, optimized cost)
    """
    print("\n" + "="*60)
    print("OPTIMIZING APPLIANCE SCHEDULE")
    print("="*60 + "\n")
    
    # Validate inputs
    if len(hourly_prices) != 24:
        raise ValueError("hourly_prices must have exactly 24 values")
    
    appliances_config = define_appliance_constraints(appliances_config)
    
    # Create optimization variables
    # For each appliance, create binary variables for each hour
    schedules = {}
    for name in appliances_config.keys():
        # Binary variable: 1 if appliance is ON at hour h, 0 otherwise
        schedules[name] = cp.Variable(24, boolean=True)
    
    # Objective: minimize total cost
    # Total cost is the sum of (power consumption * electricity price) for each hour
    # when each appliance is running
    total_cost = 0
    for name, config in appliances_config.items():
        # Cost = sum over hours of (ON/OFF * power_rating * price)
        # Convert power from Watts to kW for cost calculation (1 kW = 1000 W)
        # Electricity prices are typically in £/kWh
        power_kw = config['power_rating'] / 1000.0
        hourly_cost = cp.multiply(schedules[name], np.array(hourly_prices) * power_kw)
        total_cost += cp.sum(hourly_cost)
    
    objective = cp.Minimize(total_cost)
    
    # Constraints
    constraints = []
    
    # Constraint 1: Each appliance must run for its required runtime
    # Use ceiling to ensure fractional hours (e.g., 1.5h) are rounded up to full hours
    for name, config in appliances_config.items():
        runtime_hours = int(np.ceil(config['runtime_hours']))
        constraints.append(cp.sum(schedules[name]) == runtime_hours)
    
    # Constraint 2: Appliances can only run within their time windows
    # This enforces user-defined operational hours (e.g., washing machine only during daytime)
    for name, config in appliances_config.items():
        earliest = config['earliest_start']
        latest = config['latest_finish']
        
        # Set hours outside window to 0
        for h in range(24):
            if h < earliest or h >= latest:
                constraints.append(schedules[name][h] == 0)
    
    # Constraint 3: Appliances must run consecutively (optional but realistic)
    # This ensures appliance doesn't turn on/off multiple times
    for name, config in appliances_config.items():
        runtime_hours = int(np.ceil(config['runtime_hours']))
        earliest = config['earliest_start']
        latest = config['latest_finish']
        
        # For consecutive running, we need: if hour h is ON and h+1 is in window,
        # then either h+1 is ON or all subsequent hours are OFF
        # This is complex in CVXPY, so we'll use a sliding window approach
        
        # Alternative: use auxiliary variables to enforce consecutive blocks
        # For simplicity, we'll allow non-consecutive for now
        # (Can be enhanced with more complex constraints)
    
    # Constraint 4: Maximum simultaneous appliances
    if not allow_simultaneous:
        max_simultaneous = 1
    
    if max_simultaneous < len(appliances_config):
        for h in range(24):
            # Sum of all appliances ON at hour h <= max_simultaneous
            hour_sum = sum(schedules[name][h] for name in appliances_config.keys())
            constraints.append(hour_sum <= max_simultaneous)
    
    # Solve optimization problem
    problem = cp.Problem(objective, constraints)
    
    print("Solving optimization problem...")
    # Try multiple solvers in order of preference for robustness
    # ECOS_BB is preferred for mixed-integer problems, with SCS as fallback
    try:
        # Try ECOS solver first (comes with cvxpy)
        problem.solve(solver=cp.ECOS_BB, verbose=False)
    except:
        try:
            # Try SCS solver
            problem.solve(solver=cp.SCS, verbose=False)
        except:
            # Try default solver
            problem.solve(verbose=False)
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Optimization failed with status: {problem.status}")
    
    # Extract optimized schedule
    optimized_schedule = {}
    for name in appliances_config.keys():
        optimized_schedule[name] = np.round(schedules[name].value).astype(int)
    
    # Calculate costs
    optimized_cost = problem.value
    
    # Calculate original cost (assuming appliances run at earliest possible time)
    # This represents the baseline "unoptimized" schedule for comparison
    original_cost = 0
    original_schedule = {}
    for name, config in appliances_config.items():
        runtime_hours = int(np.ceil(config['runtime_hours']))
        earliest = config['earliest_start']
        power_kw = config['power_rating'] / 1000.0
        
        # Original schedule: run starting at earliest time (typical user behavior)
        schedule = np.zeros(24)
        schedule[earliest:earliest+runtime_hours] = 1
        original_schedule[name] = schedule
        
        # Calculate cost
        for h in range(24):
            if schedule[h] == 1:
                original_cost += hourly_prices[h] * power_kw
    
    print(f"\nOptimization Results:")
    print(f"  Original Cost: £{original_cost:.4f}")
    print(f"  Optimized Cost: £{optimized_cost:.4f}")
    print(f"  Savings: £{original_cost - optimized_cost:.4f} ({(original_cost - optimized_cost)/original_cost*100:.2f}%)")
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    return optimized_schedule, original_cost, optimized_cost


def generate_schedule_dataframe(schedule: Dict, 
                                appliances_config: Dict,
                                hourly_prices: List[float]) -> pd.DataFrame:
    """
    Generate a readable DataFrame from the optimized schedule.
    
    Args:
        schedule: Optimized schedule dictionary
        appliances_config: Appliance configurations
        hourly_prices: Hourly electricity prices
    
    Returns:
        DataFrame with schedule details
    """
    hours = list(range(24))
    
    data = {
        'Hour': hours,
        'Price (£/kWh)': hourly_prices
    }
    
    for name in schedule.keys():
        data[name] = schedule[name]
    
    df = pd.DataFrame(data)
    return df


def calculate_savings_metrics(original_cost: float,
                              optimized_cost: float,
                              original_schedule: Optional[Dict] = None,
                              optimized_schedule: Optional[Dict] = None) -> Dict:
    """
    Calculate detailed savings metrics.
    
    Args:
        original_cost: Original electricity cost
        optimized_cost: Optimized electricity cost
        original_schedule: Original appliance schedule (optional)
        optimized_schedule: Optimized appliance schedule (optional)
    
    Returns:
        Dictionary with savings metrics
    """
    savings = original_cost - optimized_cost
    savings_percent = (savings / original_cost * 100) if original_cost > 0 else 0
    
    metrics = {
        'original_cost': original_cost,
        'optimized_cost': optimized_cost,
        'absolute_savings': savings,
        'percent_savings': savings_percent
    }
    
    # Calculate peak-to-average ratio if schedules provided
    if original_schedule and optimized_schedule:
        # Sum power usage at each hour
        original_hourly = np.zeros(24)
        optimized_hourly = np.zeros(24)
        
        for name, sched in original_schedule.items():
            original_hourly += sched
        
        for name, sched in optimized_schedule.items():
            optimized_hourly += sched
        
        # Peak-to-average ratio
        original_peak = np.max(original_hourly)
        original_avg = np.mean(original_hourly[original_hourly > 0]) if np.any(original_hourly > 0) else 0
        
        optimized_peak = np.max(optimized_hourly)
        optimized_avg = np.mean(optimized_hourly[optimized_hourly > 0]) if np.any(optimized_hourly > 0) else 0
        
        metrics['original_peak_to_avg'] = original_peak / original_avg if original_avg > 0 else 0
        metrics['optimized_peak_to_avg'] = optimized_peak / optimized_avg if optimized_avg > 0 else 0
    
    return metrics


def create_summary_table(appliances_config: Dict,
                        original_schedule: Dict,
                        optimized_schedule: Dict,
                        hourly_prices: List[float]) -> pd.DataFrame:
    """
    Create summary table comparing original vs optimized schedules.
    
    Args:
        appliances_config: Appliance configurations
        original_schedule: Original schedule
        optimized_schedule: Optimized schedule
        hourly_prices: Hourly electricity prices
    
    Returns:
        Summary DataFrame
    """
    summary_data = []
    
    for name, config in appliances_config.items():
        power_kw = config['power_rating'] / 1000.0
        
        # Original schedule times
        orig_hours = np.where(original_schedule[name] == 1)[0]
        orig_start = orig_hours[0] if len(orig_hours) > 0 else -1
        orig_end = orig_hours[-1] + 1 if len(orig_hours) > 0 else -1
        
        # Optimized schedule times
        opt_hours = np.where(optimized_schedule[name] == 1)[0]
        opt_start = opt_hours[0] if len(opt_hours) > 0 else -1
        opt_end = opt_hours[-1] + 1 if len(opt_hours) > 0 else -1
        
        # Calculate costs
        orig_cost = sum(hourly_prices[h] * power_kw for h in orig_hours)
        opt_cost = sum(hourly_prices[h] * power_kw for h in opt_hours)
        
        summary_data.append({
            'Appliance': name,
            'Power (W)': config['power_rating'],
            'Runtime (h)': config['runtime_hours'],
            'Original Start': f"{orig_start:02d}:00" if orig_start >= 0 else "N/A",
            'Original End': f"{orig_end:02d}:00" if orig_end >= 0 else "N/A",
            'Optimized Start': f"{opt_start:02d}:00" if opt_start >= 0 else "N/A",
            'Optimized End': f"{opt_end:02d}:00" if opt_end >= 0 else "N/A",
            'Original Cost (£)': f"{orig_cost:.4f}",
            'Optimized Cost (£)': f"{opt_cost:.4f}",
            'Savings (£)': f"{orig_cost - opt_cost:.4f}"
        })
    
    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src import config
    
    # Optimize schedule
    optimized_schedule, orig_cost, opt_cost = optimize_schedule(
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
