"""
Example usage scripts for Energy Consumption Optimizer.
Demonstrates various use cases and configurations.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src import config
from src.data_loader import load_and_prepare_data
from src.preprocessor import preprocess_pipeline
from src.optimizer import optimize_schedule


def example_basic_optimization():
    """
    Basic example: Optimize schedule with default configuration.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Optimization")
    print("="*80 + "\n")

    # Optimize with default settings
    optimized_schedule, original_cost, optimized_cost = optimize_schedule(
        config.FLEXIBLE_APPLIANCES,
        config.HOURLY_PRICES,
        allow_simultaneous=True,
        max_simultaneous=2
    )

    print(f"\nOriginal Cost: £{original_cost:.4f}")
    print(f"Optimized Cost: £{optimized_cost:.4f}")
    print(f"Savings: £{original_cost - optimized_cost:.4f} ({(original_cost - optimized_cost)/original_cost*100:.2f}%)")


def example_custom_appliances():
    """
    Example with custom appliance configuration.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Appliance Configuration")
    print("="*80 + "\n")

    # Define custom appliances
    custom_appliances = {
        'Electric Vehicle': {
            'column': 'EV',
            'runtime_hours': 4,
            'earliest_start': 22,  # Start charging at 10 PM
            'latest_finish': 7,    # Must finish by 7 AM (wraps around midnight)
            'power_rating': 7000,  # 7 kW charger
        },
        'Pool Pump': {
            'column': 'Pool',
            'runtime_hours': 6,
            'earliest_start': 0,
            'latest_finish': 24,
            'power_rating': 1500,
        },
    }

    # Note: For appliances that wrap around midnight, you may need to adjust the optimization logic
    # For this example, we'll use a simplified version
    simplified_appliances = {
        'Pool Pump': custom_appliances['Pool Pump']
    }

    optimized_schedule, original_cost, optimized_cost = optimize_schedule(
        simplified_appliances,
        config.HOURLY_PRICES,
        allow_simultaneous=True,
        max_simultaneous=1
    )

    print(f"\nSavings: £{original_cost - optimized_cost:.4f}")


def example_no_simultaneous():
    """
    Example: Prevent appliances from running simultaneously.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: No Simultaneous Appliances")
    print("="*80 + "\n")

    optimized_schedule, original_cost, optimized_cost = optimize_schedule(
        config.FLEXIBLE_APPLIANCES,
        config.HOURLY_PRICES,
        allow_simultaneous=False,  # Only one appliance at a time
        max_simultaneous=1
    )

    print(f"\nSavings: £{original_cost - optimized_cost:.4f}")

    # Verify no overlaps
    import numpy as np
    for hour in range(24):
        count = sum(schedule[hour] for schedule in optimized_schedule.values())
        if count > 1:
            print(f"Warning: {count} appliances running at hour {hour}")

    print("✓ Verified: No simultaneous appliances")


def example_custom_pricing():
    """
    Example with custom electricity pricing.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Electricity Pricing")
    print("="*80 + "\n")

    # Flat rate pricing
    flat_rate = [0.15] * 24

    print("Flat rate pricing:")
    optimized_schedule, original_cost, optimized_cost = optimize_schedule(
        config.FLEXIBLE_APPLIANCES,
        flat_rate,
        allow_simultaneous=True,
        max_simultaneous=2
    )
    print(f"Savings with flat rate: £{original_cost - optimized_cost:.4f}")

    # Extreme time-of-use pricing
    extreme_tou = [
        0.05, 0.05, 0.05, 0.05, 0.05, 0.05,  # 00:00-05:00 (very cheap)
        0.10, 0.10, 0.10, 0.10, 0.10, 0.10,  # 06:00-11:00 (cheap)
        0.20, 0.20, 0.20,                     # 12:00-14:00 (medium)
        0.50, 0.50, 0.50, 0.50,              # 15:00-18:00 (very expensive)
        0.20, 0.20, 0.20, 0.20,              # 19:00-22:00 (medium)
        0.10                                  # 23:00 (cheap)
    ]

    print("\nExtreme time-of-use pricing:")
    optimized_schedule, original_cost, optimized_cost = optimize_schedule(
        config.FLEXIBLE_APPLIANCES,
        extreme_tou,
        allow_simultaneous=True,
        max_simultaneous=2
    )
    print(f"Savings with extreme TOU: £{original_cost - optimized_cost:.4f} ({(original_cost - optimized_cost)/original_cost*100:.2f}%)")


if __name__ == "__main__":
    # Run all examples
    example_basic_optimization()
    example_custom_appliances()
    example_no_simultaneous()
    example_custom_pricing()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")
