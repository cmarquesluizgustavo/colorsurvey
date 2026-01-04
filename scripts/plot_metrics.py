#!/usr/bin/env python3
"""
Plot training metrics from experiment runs.
Usage: python scripts/plot_metrics.py <path_to_run_directory>
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML.utils import plot_training_metrics


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('run_dir', type=str, help='Path to run directory')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    
    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}")
        return 1
    
    print(f"Plotting metrics from {run_dir}...")
    plot_path = plot_training_metrics(run_dir)
    
    if plot_path:
        print(f"✅ Plot saved to {plot_path}")
        return 0
    else:
        print("❌ Failed to generate plot")
        return 1


if __name__ == "__main__":
    sys.exit(main())
    
    # Determine trainer type
    if 'step_type' in df.columns and df['step_type'].str.contains('Step').any():
        print("Detected: Metric Learning")
        plot_metric_learning(df, run_dir)
    else:
        print("Detected: XGBoost")
        plot_xgboost(df, run_dir)


if __name__ == "__main__":
    main()
