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
    python3 collect_results.py [experiment_name] --type clip
    python3 collect_results.py [experiment_name] --type clip --sort Colors,Loss_Type,Embed_Dim
    
Example:
    python3 collect_results.py 3rd
    python3 collect_results.py 8th --local --type clip --sort Colors,Loss_Type,Embed_Dim

--local reads and writes experiments/<round>/. On the cluster, --type clip collects only
the runs that belong to the round (see round_runs).

Run on the cluster: python3 collect_results.py 4th
Then fetch the tarball it writes with cluster/fetch.sh.
"""

import math
import os
import sys
import re
import csv
import glob
import json
import shutil
import tarfile


# ── Column profiles ──────────────────────────────────────────────────

CLIP_FIELDNAMES = [
    'Experiment', 'Colors', 'Subset', 'Loss_Type', 'Embed_Dim',
    'Total_Loss', 'Loss_C2T', 'Loss_T2C', 'Loss_T2C_Strict', 'Loss_T2C_Above_Uniform',
    'Loss_T2C_Strict_Above_Uniform', 'T2C_Positives',
    'R_at_1', 'R_at_5', 'R_at_10', 'Median_Rank', 'Avg_Rank',
    'Class_Oriented_R_at_1', 'Class_Oriented_R_at_5', 'Class_Oriented_R_at_10',
    'MRR', 'Mean_Log_Odds', 'CLIP_Youden_J', 'Youden_J_at_5',
    'Temperature', 'Final_Cycle', 'Timestamp', 'Status', 'Metrics_Source', 'Metrics_Epoch',
    'R_at_1_A', 'R_at_1_B', 'Last_Beats_Best',
]

# Columns that sort ascending (grouping); everything else sorts descending (metrics)
ASCENDING_COLUMNS = {'Experiment', 'Colors', 'Subset', 'Loss_Type', 'Embed_Dim',
                     'Final_Cycle', 'Timestamp', 'Status', 'Metrics_Source',
                     'Metrics_Epoch', 'Last_Beats_Best'}

LOSS_TYPE_MAP = {'mask': 'masked', 'orig': 'original', 'supc': 'supcon',
                 'proto': 'prototype'}


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
            
            # Check if completed successfully
            if 'Training complete!' in content or 'Results saved to' in content:
                metrics['status'] = 'completed'
    
    except Exception as e:
        print(f"Error parsing {out_file}: {e}")
        
    return metrics


def detect_experiment_type(last_row):
    """Detect whether this is a CLIP or classic (XGBoost/MLP) experiment."""
    if last_row.get('r_at_1', '') != '':
        return 'clip'
    return 'classic'


def parse_experiment_details(exp_name):
    """Extract model parameters from experiment name.
    
    Handles formats:
      - Classic: ml_mlp_129colors_balanced_fixed_ctriplet_m0.5_dim16
      - CLIP v5: clip_oklch_129c_dim128_t0.03_lr0.0003_wd0.0
      - CLIP v6: clip_15c_dim4_ch4_th32
      - CLIP ablated: clip_96c_no3_proto_dim64_ch16_th256
    """
    # CLIP format
    if exp_name.startswith('clip_'):
        parts = exp_name.split('_')
        model = 'clip'
        colors = None
        embed_dim = None
        color_space = None
        subset = 'all'
        for part in parts:
            if part.endswith('c') and part[:-1].isdigit():
                colors = part[:-1]
            elif part.startswith('no') and part[2:].isdigit():
                subset = part
            elif part.startswith('skip') and part[4:].isdigit():
                subset = part
            elif part.startswith('dim') and part[3:].isdigit():
                embed_dim = part[3:]
            elif part in ('rgb', 'oklch'):
                color_space = part
        # 6th experiments dropped color_space from the name (always rgb)
        if color_space is None:
            color_space = 'rgb'
        sampling = 'none'
        return model, colors, sampling, {'color_space': color_space, 'embed_dim': embed_dim, 'subset': subset}

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


def _parse_clip_loss_type(exp_name):
    """Extract loss type from CLIP experiment name.

    Searches for known loss keys (proto, mask, orig, supc) or full loss names.
    """
    parts = exp_name.split('_')
    for part in parts:
        if part in LOSS_TYPE_MAP:
            return LOSS_TYPE_MAP[part]
        if part in LOSS_TYPE_MAP.values():
            return part
    return ''


def _optional_float(last_row, *keys):
    """Float value from the first key that is present and non-empty, else ''.

    Keeps genuinely-missing metrics blank instead of reporting them as 0.0:
    before the 18th round the loss components were logged for the prototype loss
    only, and the newer metrics are absent from runs predating them.

    Several keys allow reading a column that has been renamed: runs from before
    the 2026-08-17 `loss_t2c_above_floor` -> `loss_t2c_above_uniform` rename
    (the 12th round among them) still carry the old name.
    """
    for key in keys:
        raw = last_row.get(key, '')
        if raw not in ('', None):
            return float(raw)
    return ''


def _build_clip_entry(exp_dir, last_row, models_dir, final_metrics=None):
    """Build a dict with only CLIP-profile keys from a metrics row.

    Status flags rows whose numbers are not results, keeping the values so the failure
    stays diagnosable:
      'diverged'   - the final loss is not finite; retrieval metrics computed from NaN
                     scores come back impossible (R@1 0.0 next to R@5 1.0, MRR inf).
      'incomplete' - models_dir exists but holds no final checkpoint: training stopped
                     before it ended, so the last row is a partial epoch count.
    """
    _, colors, _, extra = parse_experiment_details(exp_dir)
    # Retrieval metrics and Temperature come from final_metrics.json (best checkpoint,
    # scored on the clean test rows) when there is one; losses, Final_Cycle and Timestamp always come
    # from the last row. Metrics_Source says where the metrics came from:
    #   'final'      - written by the trainer at the end of training
    #   'rescored'   - written afterwards by scripts/rescore_checkpoints.py from the
    #                  saved best checkpoint, for runs that predate the trainer writing it
    #   'last_epoch' - neither exists: the last training epoch's row, scored on only
    #                  max_eval_samples test samples and up to `patience` epochs past
    #                  the checkpoint
    # The clean rows are the test rows after the first max_eval_samples, which chose the
    # epoch (research_plan.md note 10), split into halves A and B. R_at_1_A / R_at_1_B:
    # pick winners on A, report them on B. Last_Beats_Best: the last epoch beat the
    # restored best on A by more than noise (runs that saved the last epoch).
    parts = {}
    if final_metrics is not None:
        metric_row = {**last_row, **final_metrics}
        source = final_metrics.get('source', 'final')
        metrics_epoch = final_metrics.get('epoch', '')
        parts = final_metrics['parts']
    else:
        metric_row = last_row
        source = 'last_epoch'
        metrics_epoch = last_row.get('cycle', '')
    total_loss = float(last_row.get('total_loss', 0) or 0)
    if not math.isfinite(total_loss):
        status = 'diverged'
    elif os.path.isdir(models_dir) and not os.path.exists(os.path.join(models_dir, FINAL_CHECKPOINT)):
        status = 'incomplete'
    else:
        status = 'completed'
    return {
        'Experiment':     exp_dir,
        'Colors':         int(colors) if colors else 0,
        'Subset':         extra.get('subset', 'all'),
        'Loss_Type':      _parse_clip_loss_type(exp_dir),
        'Embed_Dim':      int(extra.get('embed_dim', 0) or 0),
        'Total_Loss':     total_loss,
        'Loss_C2T':               _optional_float(last_row, 'loss_c2t'),
        'Loss_T2C':               _optional_float(last_row, 'loss_t2c'),
        # Exact-label t->c loss, comparable across t2c_eps (18th round on); before
        # t2c_eps existed, the prototype loss logged that same quantity as loss_t2c.
        'Loss_T2C_Strict':        _optional_float(last_row, 'loss_t2c_strict', 'loss_t2c'),
        'Loss_T2C_Above_Uniform': _optional_float(last_row, 'loss_t2c_above_uniform',
                                                   'loss_t2c_above_floor'),
        'Loss_T2C_Strict_Above_Uniform': _optional_float(last_row, 'loss_t2c_strict_above_uniform',
                                                         'loss_t2c_above_uniform',
                                                         'loss_t2c_above_floor'),
        'T2C_Positives':          _optional_float(last_row, 't2c_positives'),
        'R_at_1':         float(metric_row.get('r_at_1', 0) or 0),
        'R_at_5':         float(metric_row.get('r_at_5', 0) or 0),
        'R_at_10':        float(metric_row.get('r_at_10', 0) or 0),
        'Median_Rank':    float(metric_row.get('median_rank', 0) or 0),
        'Avg_Rank':               _optional_float(metric_row, 'avg_rank'),
        'Class_Oriented_R_at_1':  _optional_float(metric_row, 'class_oriented_r_at_1'),
        'Class_Oriented_R_at_5':  _optional_float(metric_row, 'class_oriented_r_at_5'),
        'Class_Oriented_R_at_10': _optional_float(metric_row, 'class_oriented_r_at_10'),
        'MRR':            float(metric_row.get('mrr', 0) or 0),
        'Mean_Log_Odds':  float(metric_row.get('mean_log_odds', 0) or 0),
        'CLIP_Youden_J':  float(metric_row.get('youdens_j', 0) or 0),
        'Youden_J_at_5':          _optional_float(metric_row, 'youdens_j_at_5'),
        'Temperature':    float(metric_row.get('temperature', 0) or 0),
        'Final_Cycle':    last_row.get('cycle', ''),
        'Timestamp':      last_row.get('timestamp', ''),
        'Status':         status,
        'Metrics_Source': source,
        'Metrics_Epoch':  metrics_epoch,
        'R_at_1_A':       parts.get('a', {}).get('r_at_1', ''),
        'R_at_1_B':       parts.get('b', {}).get('r_at_1', ''),
        'Last_Beats_Best': final_metrics.get('last_beats_best', '') if final_metrics else '',
    }


def _multi_sort_key(entry, sort_cols):
    """Return a tuple for multi-column sorting.

    Grouping columns (in ASCENDING_COLUMNS) sort ascending.
    Metric columns sort descending (negated).
    """
    keys = []
    for col in sort_cols:
        val = entry.get(col, '')
        if col in ASCENDING_COLUMNS:
            # ascending: str/int as-is
            keys.append((val if not isinstance(val, str) else val.lower()))
        else:
            # descending: negate numeric, or invert string
            keys.append(-val if isinstance(val, (int, float)) else val)
    return tuple(keys)


# Written only when training ends normally (early stop or last epoch); a run that
# crashed has best_color_clip_model.pth at most.
FINAL_CHECKPOINT = 'color_clip_model.pth'

# Training flags that change what a run writes, not what it trains.
_OUTPUT_ONLY_KEYS = ('save_scores', 'save_ranking')


def _comparable(config):
    training = {k: v for k, v in config.get('training', {}).items() if k not in _OUTPUT_ONLY_KEYS}
    return {**config, 'training': training}


def round_runs(root, round_name):
    """Map each experiment of a round to the run directory that produced it.

    A run belongs to the round when its name is one of the round's configs and its saved
    config.json equals that config, output-only flags aside. Names are reused across
    rounds with different configs, so the name alone cannot tell rounds apart. When several
    runs match, the newest wins. Experiments with no matching run are absent.

    Raises FileNotFoundError if the round has no configs/ directory.
    """
    config_dir = os.path.join(root, 'experiments', round_name, 'configs')
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"no configs at experiments/{round_name}/configs")
    wanted = {}
    for path in glob.glob(os.path.join(config_dir, '*.json')):
        with open(path) as f:
            wanted[os.path.basename(path)[:-5]] = _comparable(json.load(f))

    bases = [os.path.join(root, 'runs'), os.path.join(root, 'experiments', 'runs')]
    bases += sorted(glob.glob(os.path.join(root, 'experiments', 'runs_*')))
    found = {}
    for base in bases:
        for name, config in wanted.items():
            for run_dir in glob.glob(os.path.join(base, name, 'run_*')):
                saved = os.path.join(run_dir, 'config.json')
                if not os.path.exists(saved):
                    continue
                with open(saved) as f:
                    if _comparable(json.load(f)) != config:
                        continue
                if name not in found or os.path.basename(run_dir) > os.path.basename(found[name]):
                    found[name] = run_dir
    return found


def _find_experiment_dirs():
    """Find all experiment result directories across HTCondor output layouts.
    
    Scans:
      - runs/<exp>/run_*/              (flat / manually merged)
      - experiments/runs/<exp>/run_*/   (older layout)
      - experiments/runs_N/<exp>/run_*/ (HTCondor per-job remapped)
    
    Returns dict {exp_name: run_dir_path} (latest run per experiment).
    """
    exp_dirs: dict[str, str] = {}

    def _scan_base(base):
        if not os.path.isdir(base):
            return
        for name in os.listdir(base):
            path = os.path.join(base, name)
            if not os.path.isdir(path):
                continue
            run_dirs = sorted(d for d in os.listdir(path) if d.startswith('run_'))
            if run_dirs:
                exp_dirs[name] = os.path.join(path, run_dirs[-1])

    # Layout 1: flat runs/ directory
    _scan_base('runs')
    # Layout 2: experiments/runs/
    _scan_base('experiments/runs')
    # Layout 3: HTCondor per-job remapped (experiments/runs_0/, runs_1/, ...)
    for remapped in sorted(glob.glob('experiments/runs_*')):
        _scan_base(remapped)

    return exp_dirs


def _find_local_metrics(experiment_name):
    """Build {exp_name: metrics_path} from <experiment>/metrics/ directory.

    Used with --local to rebuild experiment_results.csv without needing
    the original runs/ directories.
    """
    metrics_dir = os.path.join('experiments', experiment_name, 'metrics')
    result = {}
    if not os.path.isdir(metrics_dir):
        return result
    for f in os.listdir(metrics_dir):
        if f.endswith('_metrics.csv'):
            exp_name = f.replace('_metrics.csv', '')
            result[exp_name] = os.path.join(metrics_dir, f)
    return result


def main():
    local_mode = '--local' in sys.argv
    exp_type_flag = None
    sort_cols = None
    positional = []
    skip_next = False
    for i, a in enumerate(sys.argv[1:], 1):
        if skip_next:
            skip_next = False
            continue
        if a == '--local':
            continue
        if a == '--type':
            exp_type_flag = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
            skip_next = True
            continue
        if a == '--sort':
            sort_cols = [c.strip() for c in sys.argv[i + 1].split(',')]
            skip_next = True
            continue
        positional.append(a)
    experiment_name = positional[0] if positional else '3rd'

    if not local_mode:
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("Collecting completed experiment results...")
    print(f"Experiment: {experiment_name}")
    if local_mode:
        print("Mode: local (rebuilding CSV from metrics/ directory)")
    print("=" * 60)

    if local_mode:
        local_metrics = _find_local_metrics(experiment_name)
        if not local_metrics:
            print("No metrics files found!")
            return
        print(f"Found {len(local_metrics)} metrics files")
        # Build a fake exp_results dict: {exp_name: metrics_path}
        # We'll handle this below by reading directly from the path
        exp_results = local_metrics
    elif exp_type_flag == 'clip':
        try:
            exp_results = round_runs('.', experiment_name)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        if not exp_results:
            print("No experiment results found!")
            return
        print(f"Matched {len(exp_results)} runs to the round's configs")
    else:
        exp_results = _find_experiment_dirs()
        if not exp_results:
            print("No experiment results found!")
            return
        print(f"Discovered {len(exp_results)} experiment directories")
    
    # ── CLIP-only path ────────────────────────────────────────────────
    if exp_type_flag == 'clip':
        completed = []
        for exp_dir, run_dir in exp_results.items():
            metrics_file = run_dir if local_mode else os.path.join(run_dir, 'metrics.csv')
            if not os.path.exists(metrics_file):
                continue
            try:
                with open(metrics_file, 'r') as f:
                    rows = list(csv.DictReader(f))
                if not rows or detect_experiment_type(rows[-1]) != 'clip':
                    continue
                models_dir = (os.path.join('experiments', experiment_name, 'models', exp_dir)
                              if local_mode else os.path.join(run_dir, 'models'))
                final_path = (os.path.join('experiments', experiment_name, 'metrics',
                                           f'{exp_dir}_final_metrics.json')
                              if local_mode else os.path.join(run_dir, 'final_metrics.json'))
                final_metrics = None
                if os.path.exists(final_path):
                    with open(final_path) as f:
                        final_metrics = json.load(f)
                completed.append(_build_clip_entry(exp_dir, rows[-1], models_dir, final_metrics))
            except Exception as e:
                print(f"Error processing {exp_dir}: {e}")

        print(f"Found {len(completed)} completed CLIP experiments")
        if not completed:
            print("No completed CLIP experiments found!")
            return

        # Sort
        if sort_cols:
            completed.sort(key=lambda e: _multi_sort_key(e, sort_cols))
        else:
            completed.sort(key=lambda e: -e.get('R_at_1', 0))

        # Write CSV
        if local_mode:
            csv_path = os.path.join('experiments', experiment_name, 'experiment_results.csv')
        else:
            results_dir = f'{experiment_name}_results'
            os.makedirs(f'{results_dir}/metrics', exist_ok=True)
            os.makedirs(f'{results_dir}/tensorboards', exist_ok=True)
            os.makedirs(f'{results_dir}/models', exist_ok=True)
            csv_path = f'{results_dir}/experiment_results.csv'

        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else v
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CLIP_FIELDNAMES)
            writer.writeheader()
            for entry in completed:
                writer.writerow({k: fmt(entry[k]) for k in CLIP_FIELDNAMES})

        print(f"Created {csv_path} ({len(completed)} experiments)")

        if local_mode:
            print("Done!")
            return

        # Copy artifacts (same as default path)
        for entry in completed:
            exp_name = entry['Experiment']
            run_dir = exp_results[exp_name]
            metrics_src = os.path.join(run_dir, 'metrics.csv') if not local_mode else run_dir
            if os.path.exists(metrics_src):
                shutil.copy2(metrics_src, f'{results_dir}/metrics/{exp_name}_metrics.csv')
            final_src = os.path.join(run_dir, 'final_metrics.json')
            if os.path.exists(final_src):
                shutil.copy2(final_src, f'{results_dir}/metrics/{exp_name}_final_metrics.json')
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

        tarball_name = f'{experiment_name}_results.tar.gz'
        print(f"\nCreating {tarball_name}...")
        with tarfile.open(tarball_name, 'w:gz') as tar:
            tar.add(results_dir, arcname=results_dir)
        print(f"Created {tarball_name}")
        print(f"\nWrote {tarball_name}")
        shutil.rmtree(results_dir)
        print("\nDone!")
        return

    # ── Default path (union format, unchanged) ────────────────────────
    completed_experiments = []
    
    for exp_dir, run_dir in exp_results.items():
        metrics_file = run_dir if local_mode else os.path.join(run_dir, 'metrics.csv')
        
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
                        'mrr':          float(last_row.get('mrr', 0) or 0),
                        'mean_log_odds': float(last_row.get('mean_log_odds', 0) or 0),
                        'clip_youdens_j': float(last_row.get('youdens_j', 0) or 0),
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
                        'median_rank': '', 'temperature': '',
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
    if local_mode:
        csv_path = os.path.join('experiments', experiment_name, 'experiment_results.csv')
    else:
        results_dir = f'{experiment_name}_results'
        os.makedirs(f'{results_dir}/metrics', exist_ok=True)
        os.makedirs(f'{results_dir}/tensorboards', exist_ok=True)
        os.makedirs(f'{results_dir}/models', exist_ok=True)
        csv_path = f'{results_dir}/experiment_results.csv'
    fieldnames = [
        'Rank', 'Experiment', 'Type', 'Model', 'Colors', 'Sampling',
        'Color_Space', 'Embed_Dim',
        # CLIP metrics
        'Total_Loss', 'R_at_1', 'R_at_5', 'R_at_10', 'Median_Rank',
        'MRR', 'Mean_Log_Odds', 'CLIP_Youden_J',
        'Temperature',
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
                'MRR': fmt(exp.get('mrr', '')),
                'Mean_Log_Odds': fmt(exp.get('mean_log_odds', '')),
                'CLIP_Youden_J': fmt(exp.get('clip_youdens_j', '')),
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
    
    if local_mode:
        print(f"\nRebuilt CSV with {len(completed_experiments)} experiments")
        print("Done!")
        return

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
    print(f"\nWrote {tarball_name}")
    
    shutil.rmtree(results_dir)
    print("\nDone!")


if __name__ == '__main__':
    main()
