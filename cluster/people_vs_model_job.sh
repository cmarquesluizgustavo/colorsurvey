#!/bin/bash
# HTCondor job for cluster/people_vs_model.sub: one model's people-vs-model scores.
#   $1  round, e.g. 14th
#   $2  run name (output <run>.npz)
set -e

tar -xzf in.tar.gz

# Same packages as geometry_job.sh.
mkdir -p .local_packages
export PYTHONPATH="$PWD/.local_packages:$PYTHONPATH"
python3 -c "import torch" 2>/dev/null || \
    python3 -m pip install --no-cache-dir --target=.local_packages torch --index-url https://download.pytorch.org/whl/cpu
for pair in numpy:numpy pandas:pandas sklearn:scikit-learn imblearn:imbalanced-learn scipy:scipy matplotlib:matplotlib tensorboard:tensorboard; do
    python3 -c "import ${pair%%:*}" 2>/dev/null || \
        python3 -m pip install --no-cache-dir --target=.local_packages "${pair##*:}"
done

echo "People vs model for $2 on $(hostname)"
python3 annexes/people_vs_model/build.py score "$1" "$2"
