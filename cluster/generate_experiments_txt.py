"""
Generate sorted experiments.txt with per-job memory requests.

Sorting: smallest configs first (top_n_colors, embed_dim, hidden complexity).
Memory:  129c → 1000 MB,  1184c/4214c → 1500 MB.
"""
import glob
import os
import re

MEMORY_MAP = {129: 1000, 1184: 1500, 4214: 1500}


def parse_sort_key(path):
    """Extract numeric sort key from config filename."""
    name = os.path.basename(path)
    colors = int(re.search(r'_(\d+)c_', name).group(1))
    dim = int(re.search(r'_dim(\d+)', name).group(1))
    ch = list(map(int, re.search(r'_ch([\dx]+)', name).group(1).split('x')))
    th = list(map(int, re.search(r'_th([\dx]+)', name).group(1).split('x')))
    return (colors, dim, len(ch), ch, len(th), th)


def main():
    config_dir = os.path.join(os.path.dirname(__file__), '..', '6th_experiments', 'configs')

    # Load completed experiment names to exclude (from metrics/ folder)
    metrics_dir = os.path.join(os.path.dirname(__file__), '..', '6th_experiments', 'metrics')
    completed = set()
    if os.path.isdir(metrics_dir):
        for f in os.listdir(metrics_dir):
            if f.endswith('_metrics.csv'):
                completed.add(f.replace('_metrics.csv', ''))

    all_configs = sorted(glob.glob(os.path.join(config_dir, 'clip_*.json')),
                         key=parse_sort_key)

    # Filter out completed
    configs = [p for p in all_configs
               if os.path.basename(p).replace('.json', '') not in completed]

    out_path = os.path.join(os.path.dirname(__file__), 'experiments.txt')
    with open(out_path, 'w') as f:
        for path in configs:
            rel = '6th_experiments/configs/' + os.path.basename(path)
            colors = int(re.search(r'_(\d+)c_', os.path.basename(path)).group(1))
            mem = MEMORY_MAP[colors]
            f.write(f'{rel}, {mem}\n')

    print(f"Total configs: {len(all_configs)}, Completed: {len(completed)}, "
          f"Pending: {len(configs)}")
    print(f"Wrote {len(configs)} entries to {out_path}")
    # Show first and last few
    with open(out_path) as f:
        lines = f.readlines()
    for line in lines[:3]:
        print(f"  {line.rstrip()}")
    print(f"  ... ({len(lines) - 6} more)")
    for line in lines[-3:]:
        print(f"  {line.rstrip()}")


if __name__ == '__main__':
    main()
