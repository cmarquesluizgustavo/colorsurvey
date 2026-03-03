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


def detect_experiment_type(last_row):
    """Detect whether this is a CLIP or classic (XGBoost/MLP) experiment."""
    if last_row.get('r_at_1', '') != '':
        return 'clip'
    return 'classic'


def parse_experiment_details(exp_name):
    """Extract model details from experiment name.
    
    Handles both formats:
      - Classic: ml_mlp_129colors_balanced_fixed_ctriplet_m0.5_dim16
      - CLIP:    clip_oklch_129c_dim128_t0.03_lr0.0003_wd0.0
    """
    # CLIP format
    if exp_name.startswith('clip_'):
        parts = exp_name.split('_')
        model = 'clip'
        # color space (e.g. oklch)
        color_space = parts[1] if len(parts) > 1 else 'unknown'
        # number of colors (e.g. 129c -> 129)
        colors = None
        embed_dim = None
        for part in parts:
            if part.endswith('c') and part[:-1].isdigit():
                colors = part[:-1]
            if part.startswith('dim'):
                embed_dim = part.replace('dim', '')
        sampling = 'none'
        return model, colors, sampling, {'color_space': color_space, 'embed_dim': embed_dim}

    # Classic format
    parts = exp_name.split('_')
    model = 'mlp'
    colors = None
    for part in parts:
        if 'colors' in part:
            colors = part.replace('colors', '')
            break
    if 'balanced_fixed' in exp_name:
        sampling = 'balanced_fixed'
    elif 'unbalanced' in exp_name:
        sampling = 'unbalanced'
    else:
        sampling = 'unknown'
    return model, colors, sampling, {}


def main():
    os.chdir(os.path.expanduser('~/colorsurvey'))
    
    # Get experiment name from command line or use default
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else '3rd_experiments'
    
    print("Collecting completed experiment results...")
    print(f"Experiment: {experiment_name}")
    print("=" * 60)
    
    # Results live in runs/ (HTCondor output dir)
    runs_base = 'runs'
    if not os.path.exists(runs_base):
        # fallback for older setups
        runs_base = 'experiments/runs'
    if not os.path.exists(runs_base):
        print(f"No runs directory found!")
        return
    
    completed_experiments = []
    
    for exp_dir in os.listdir(runs_base):
        exp_path = os.path.join(runs_base, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        
        run_dirs = sorted([d for d in os.listdir(exp_path) if d.startswith('run_')])
        if not run_dirs:
            continue
        
        run_dir = os.path.join(exp_path, run_dirs[-1])
        metrics_file = os.path.join(run_dir, 'metrics.csv')
        
        if not os.path.exists(metrics_file):
            continue
        
        try:
            with open(metrics_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    continue
                
                last_row = rows[-1]
                exp_type = detect_experiment_type(last_row)
                model, colors, sampling, extra = parse_experiment_details(exp_dir)
                
                entry = {
                    'experiment': exp_dir,
                    'type': exp_type,
                    'model': model,
                    'colors': colors,
                    'sampling': sampling,
                    'final_cycle': last_row.get('cycle', ''),
                    'timestamp': last_row.get('timestamp', ''),
                    'run_dir': run_dir,
                    'status': 'completed',
                }
                
                if exp_type == 'clip':
                    entry.update({
                        'total_loss':   float(last_row.get('total_loss', 0) or 0),
                        'r_at_1':       float(last_row.get('r_at_1', 0) or 0),
                        'r_at_5':       float(last_row.get('r_at_5', 0) or 0),
                        'r_at_10':      float(last_row.get('r_at_10', 0) or 0),
                        'median_rank':  float(last_row.get('median_rank', 0) or 0),
                        'delta_e':      float(last_row.get('delta_e', 0) or 0),
                        'temperature':  float(last_row.get('temperature', 0) or 0),
                        'color_space':  extra.get('color_space', ''),
                        'embed_dim':    extra.get('embed_dim', ''),
                        # classic fields empty
                        'test_accuracy': '', 'youdens_j': '',
                        'train_accuracy': '', 'train_youdens_j': '',
                    })
                else:
                    entry.update({
                        'test_accuracy':   float(last_row.get('accuracy', 0) or 0),
                        'youdens_j':       float(last_row.get('youdens_j', 0) or 0),
                        'train_accuracy':  float(last_row.get('train_accuracy', 0) or 0),
                        'train_youdens_j': float(last_row.get('train_youdens_j', 0) or 0),
                        # clip fields empty
                        'total_loss': '', 'r_at_1': '', 'r_at_5': '', 'r_at_10': '',
                        'median_rank': '', 'delta_e': '', 'temperature': '',
                        'color_space': '', 'embed_dim': '',
                    })
                
                completed_experiments.append(entry)
                
        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")
            continue
    
    print(f"Found {len(completed_experiments)} completed experiments")
    
    if not completed_experiments:
        print("No completed experiments found!")
        return
    
    # Sort: CLIP by r_at_1 desc, classic by youdens_j desc
    clip_exps     = [e for e in completed_experiments if e['type'] == 'clip']
    classic_exps  = [e for e in completed_experiments if e['type'] != 'clip']
    clip_exps.sort(key=lambda x: x.get('r_at_1', 0), reverse=True)
    classic_exps.sort(key=lambda x: x.get('youdens_j', 0), reverse=True)
    completed_experiments = clip_exps + classic_exps
    
    # Output dirs
    results_dir = f'{experiment_name}_results'
    os.makedirs(f'{results_dir}/metrics', exist_ok=True)
    os.makedirs(f'{results_dir}/tensorboards', exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    
    # Unified CSV with all columns
    csv_path = f'{results_dir}/experiment_results.csv'
    fieldnames = [
        'Rank', 'Experiment', 'Type', 'Model', 'Colors', 'Sampling',
        'Color_Space', 'Embed_Dim',
        # CLIP metrics
        'Total_Loss', 'R_at_1', 'R_at_5', 'R_at_10', 'Median_Rank', 'Delta_E', 'Temperature',
        # Classic metrics
        'Test_Accuracy', 'Youden_J', 'Train_Accuracy', 'Train_Youden_J',
        # Shared
        'Final_Cycle', 'Timestamp', 'Status',
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, exp in enumerate(completed_experiments, 1):
            def fmt(v):
                return f"{v:.4f}" if isinstance(v, float) else v
            writer.writerow({
                'Rank': rank,
                'Experiment': exp['experiment'],
                'Type': exp['type'],
                'Model': exp['model'],
                'Colors': exp['colors'],
                'Sampling': exp['sampling'],
                'Color_Space': exp.get('color_space', ''),
                'Embed_Dim': exp.get('embed_dim', ''),
                'Total_Loss': fmt(exp.get('total_loss', '')),
                'R_at_1': fmt(exp.get('r_at_1', '')),
                'R_at_5': fmt(exp.get('r_at_5', '')),
                'R_at_10': fmt(exp.get('r_at_10', '')),
                'Median_Rank': fmt(exp.get('median_rank', '')),
                'Delta_E': fmt(exp.get('delta_e', '')),
                'Temperature': fmt(exp.get('temperature', '')),
                'Test_Accuracy': fmt(exp.get('test_accuracy', '')),
                'Youden_J': fmt(exp.get('youdens_j', '')),
                'Train_Accuracy': fmt(exp.get('train_accuracy', '')),
                'Train_Youden_J': fmt(exp.get('train_youdens_j', '')),
                'Final_Cycle': exp['final_cycle'],
                'Timestamp': exp['timestamp'],
                'Status': exp['status'],
            })
    
    print(f"Created {csv_path}")
    
    # Copy metrics, tensorboards, models
    for exp in completed_experiments:
        exp_name = exp['experiment']
        run_dir = exp['run_dir']
        
        metrics_src = os.path.join(run_dir, 'metrics.csv')
        if os.path.exists(metrics_src):
            shutil.copy2(metrics_src, f'{results_dir}/metrics/{exp_name}_metrics.csv')
            print(f"  Copied metrics: {exp_name}")
        
        for tb_file in glob.glob(f'{run_dir}/events.out.tfevents.*'):
            tb_dst = f'{results_dir}/tensorboards/{exp_name}'
            os.makedirs(tb_dst, exist_ok=True)
            shutil.copy2(tb_file, tb_dst)
        
        models_src = os.path.join(run_dir, 'models')
        if os.path.exists(models_src):
            model_dst = f'{results_dir}/models/{exp_name}'
            os.makedirs(model_dst, exist_ok=True)
            for mf in os.listdir(models_src):
                if mf.endswith('.pth') or mf.endswith('.json'):
                    shutil.copy2(os.path.join(models_src, mf), os.path.join(model_dst, mf))
    
    print(f"\nCollected {len(completed_experiments)} experiment results")
    
    tarball_name = f'{experiment_name}_results.tar.gz'
    print(f"\nCreating {tarball_name}...")
    with tarfile.open(tarball_name, 'w:gz') as tar:
        tar.add(results_dir, arcname=results_dir)
    
    print(f"Created {tarball_name}")
    print(f"\nTo download: scp zeus:~/colorsurvey/{tarball_name} .")
    
    shutil.rmtree(results_dir)
    print("\nDone!")


if __name__ == '__main__':
    main()
