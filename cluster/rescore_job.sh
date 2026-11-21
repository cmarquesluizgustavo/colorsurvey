#!/bin/bash
# HTCondor worker wrapper for scripts/rescore_checkpoints.py.
#   $1  job id (expects rescore_<job>.tar.gz + code.tar.gz + mainsurvey.csv)
#   $2  optional flag token: "crowd" or "nocrowd"
set -e

JOB_ID="$1"
MODE="${2:-crowd}"

tar -xzf code.tar.gz
tar -xzf "rescore_${JOB_ID}.tar.gz"

mkdir -p .local_packages .mplconfig
export PYTHONPATH="$PWD/.local_packages:$PYTHONPATH"
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 -c "import torch" 2>/dev/null || \
    python3 -m pip install --no-cache-dir --target=.local_packages torch --index-url https://download.pytorch.org/whl/cpu
for pair in numpy:numpy pandas:pandas sklearn:scikit-learn imblearn:imbalanced-learn scipy:scipy matplotlib:matplotlib tensorboard:tensorboard; do
    python3 -c "import ${pair%%:*}" 2>/dev/null || \
        python3 -m pip install --no-cache-dir --target=.local_packages "${pair##*:}"
done

EXTRA_FLAG=""
if [ "$MODE" = "nocrowd" ]; then
    EXTRA_FLAG="--no-crowd"
fi

echo "Rescoring job ${JOB_ID} (${MODE}) on $(hostname)"
python3 scripts/rescore_checkpoints.py --batch "manifest_${JOB_ID}.txt" --force ${EXTRA_FLAG}

python3 -c '
import os, tarfile
job_id = "'"${JOB_ID}"'"
with tarfile.open(f"rescore_out_{job_id}.tar.gz", "w:gz") as tar:
    with open(f"manifest_{job_id}.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rnd, exp = [p.strip() for p in line.split(",", 1)]
            for suf in ("_final_metrics.json", "_per_name.csv"):
                path = os.path.join("experiments", rnd, "metrics", f"{exp}{suf}")
                if os.path.exists(path):
                    tar.add(path)
'
