#!/usr/bin/env python3
"""
Collect completed experiment results.

This script:
1. Identifies completed experiments from logs
2. Extracts metrics and creates experiment_results.csv
3. Copies metrics.csv and tensorboard files for completed experiments
4. Creates a tarball ready for download

Usage:
    python3 collect_results.py [experiment_name]
    
Example:
    python3 collect_results.py 3rd_experiments
    python3 collect_results.py 4th_experiments

Run on cluster: python3 collect_results.py 4th_experiments
Then download: scp my_cluster:colorsurvey/4th_experiments_results.tar.gz .
"""

import os
import sys
import re
import csv
import glob
import shutil
import tarfile
from pathlib import Path
from datetime import datetime


def parse_experiment_name(config_file):
    """Extract experiment name from config filename."""
    return config_file.replace('.json', '')


def parse_job_log(out_file):
    """Extract metrics from job output log."""
    metrics = {
        'experiment': None,
        'test_accuracy': None,
        'youdens_j': None,
        'avg_recall': None,
        'min_recall': None,
        'max_recall': None,
        'train_samples': None,
        'test_samples': None,
        'status': 'failed'
    }
    
    try:
        with open(out_file, 'r') as f:
            content = f.read()
            
            # Extract experiment name
            exp_match = re.search(r'Starting experiment: (.+\.json)', content)
            if exp_match:
                metrics['experiment'] = parse_experiment_name(exp_match.group(1))
            
            # Extract sample counts
            samples_match = re.search(r'Train samples: ([\d,]+), Test samples: ([\d,]+)', content)
            if samples_match:
                metrics['train_samples'] = int(samples_match.group(1).replace(',', ''))
                metrics['test_samples'] = int(samples_match.group(2).replace(',', ''))
            
            # Extract final metrics
            acc_match = re.search(r'Final Accuracy: ([\d.]+)', content)
            if acc_match:
                metrics['test_accuracy'] = float(acc_match.group(1))
            
            youdens_match = re.search(r"Final Youden's J: ([\d.]+)", content)
            if youdens_match:
                metrics['youdens_j'] = float(youdens_match.group(1))
            
            recall_match = re.search(r'Per-class recall: mean=([\d.]+), min=([\d.]+), max=([\d.]+)', content)
            if recall_match:
                metrics['avg_recall'] = float(recall_match.group(1))
                metrics['min_recall'] = float(recall_match.group(2))
                metrics['max_recall'] = float(recall_match.group(3))
            
            # Check for completion
            if 'Training complete!' in content:
                metrics['status'] = 'completed'
    
    except FileNotFoundError:
        pass
    
    return metrics


def parse_experiment_details(exp_name):
    """Extract model details from experiment name."""
    # Format: ml_mlp_129colors_balanced_fixed_ctriplet_m0.5_dim16
    parts = exp_name.split('_')
    
    model = 'mlp'
    
    # Find colors
    colors = None
    for part in parts:
        if 'colors' in part:
            colors = part.replace('colors', '')
            break
    
    # Find sampling strategy
    if 'balanced_fixed' in exp_name:
        sampling = 'balanced_fixed'
    elif 'unbalanced' in exp_name:
        sampling = 'unbalanced'
    else:
        sampling = 'unknown'
    
    return model, colors, sampling


def main():
    os.chdir(os.path.expanduser('~/colorsurvey'))
    
    # Get experiment name from command line or use default
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else '3rd_experiments'
    
    print("Collecting completed experiment results...")
    print(f"Experiment: {experiment_name}")
    print("=" * 60)
    
    # Find all completed experiment runs directly from runs/ directory
    completed_experiments = []
    
    if not os.path.exists('runs'):
        print("No runs directory found!")
        return
    
    # Iterate through all experiment directories in runs/
    for exp_dir in os.listdir('runs'):
        exp_path = os.path.join('runs', exp_dir)
        if not os.path.isdir(exp_path):
            continue
        
        # Find the run subdirectory
        run_dirs = sorted([d for d in os.listdir(exp_path) if d.startswith('run_')])
        if not run_dirs:
            continue
        
        # Use the latest run
        run_dir = os.path.join(exp_path, run_dirs[-1])
        metrics_file = os.path.join(run_dir, 'metrics.csv')
        
        if not os.path.exists(metrics_file):
            continue
        
        # Read metrics from CSV
        try:
            with open(metrics_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    continue
                
                # Get final row metrics
                last_row = rows[-1]
                
                # Parse experiment details
                model, colors, sampling = parse_experiment_details(exp_dir)
                
                completed_experiments.append({
                    'experiment': exp_dir,
                    'model': model,
                    'colors': colors,
                    'sampling': sampling,
                    'final_cycle': int(last_row['cycle']),
                    'test_accuracy': float(last_row['accuracy']),
                    'youdens_j': float(last_row['youdens_j']),
                    'train_accuracy': float(last_row.get('train_accuracy', 0)),
                    'train_youdens_j': float(last_row.get('train_youdens_j', 0)),
                    'timestamp': last_row['timestamp'],
                    'run_dir': run_dir,
                    'status': 'completed'
                })
                
        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")
            continue
    
    print(f"Found {len(completed_experiments)} completed experiments")
    
    if not completed_experiments:
        print("No completed experiments found!")
        return
    
    # Sort by Youden's J (descending)
    completed_experiments.sort(key=lambda x: x['youdens_j'], reverse=True)
    
    # Create results directory with experiment name
    results_dir = f'{experiment_name}_results'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/metrics', exist_ok=True)
    os.makedirs(f'{results_dir}/tensorboards', exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    
    # Create experiment_results.csv
    csv_path = f'{results_dir}/experiment_results.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['Rank', 'Experiment', 'Model', 'Colors', 'Sampling', 
                     'Test_Accuracy', 'Youden_J', 'Final_Cycle',
                     'Train_Accuracy', 'Train_Youden_J', 'Timestamp', 'Status']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for rank, exp in enumerate(completed_experiments, 1):
            writer.writerow({
                'Rank': rank,
                'Experiment': exp['experiment'],
                'Model': exp['model'],
                'Colors': exp['colors'],
                'Sampling': exp['sampling'],
                'Test_Accuracy': f"{exp['test_accuracy']:.4f}",
                'Youden_J': f"{exp['youdens_j']:.4f}",
                'Final_Cycle': exp['final_cycle'],
                'Train_Accuracy': f"{exp['train_accuracy']:.4f}",
                'Train_Youden_J': f"{exp['train_youdens_j']:.4f}",
                'Timestamp': exp['timestamp'],
                'Status': exp['status']
            })
    
    print(f"Created {csv_path}")
    
    # Copy metrics and tensorboard files for completed experiments
    for exp in completed_experiments:
        exp_name = exp['experiment']
        run_dir = exp['run_dir']
        
        # Copy metrics.csv
        metrics_src = os.path.join(run_dir, 'metrics.csv')
        if os.path.exists(metrics_src):
            metrics_dst = f'{results_dir}/metrics/{exp_name}_metrics.csv'
            shutil.copy2(metrics_src, metrics_dst)
            print(f"  Copied metrics: {exp_name}")
        
        # Copy tensorboard events
        tb_pattern = f'{run_dir}/events.out.tfevents.*'
        tb_files = glob.glob(tb_pattern)
        
        if tb_files:
            tb_dst_dir = f'{results_dir}/tensorboards/{exp_name}'
            os.makedirs(tb_dst_dir, exist_ok=True)
            for tb_file in tb_files:
                shutil.copy2(tb_file, tb_dst_dir)
        
        # Copy model files from models/ subdirectory
        models_src_dir = os.path.join(run_dir, 'models')
        if os.path.exists(models_src_dir):
            model_dst_dir = f'{results_dir}/models/{exp_name}'
            os.makedirs(model_dst_dir, exist_ok=True)
            for model_file in os.listdir(models_src_dir):
                if model_file.endswith('.pth'):
                    model_src = os.path.join(models_src_dir, model_file)
                    shutil.copy2(model_src, os.path.join(model_dst_dir, model_file))
    
    print(f"\nCollected {len(completed_experiments)} experiment results")
    
    # Create tarball with experiment name
    tarball_name = f'{experiment_name}_results.tar.gz'
    print(f"\nCreating {tarball_name}...")
    
    with tarfile.open(tarball_name, 'w:gz') as tar:
        tar.add(results_dir, arcname=results_dir)
    
    print(f"Created {tarball_name}")
    print(f"\nTo download: scp my_cluster:~/colorsurvey/{tarball_name} .")
    
    # Cleanup
    shutil.rmtree(results_dir)
    print("\nDone!")


if __name__ == '__main__':
    main()
