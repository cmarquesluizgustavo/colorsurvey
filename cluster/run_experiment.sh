#!/bin/bash
# Wrapper script to run experiments on cluster nodes
# Uses system Python + user-local packages to avoid venv complications

set -e

# HTCondor flattens directory structure - recreate it
echo "Recreating ML/ directory structure on $(hostname)..."
mkdir -p ML/models ML/trainers ML/losses

# Move Python files back to ML/ directory
mv data_prep.py ML/ 2>/dev/null || true
mv main.py ML/ 2>/dev/null || true
mv metrics.py ML/ 2>/dev/null || true
mv utils.py ML/ 2>/dev/null || true
mv embeddings.py ML/ 2>/dev/null || true
mv models/*.py ML/models/ 2>/dev/null || true
mv trainers/*.py ML/trainers/ 2>/dev/null || true
mv losses/*.py ML/losses/ 2>/dev/null || true
mv __pycache__ ML/ 2>/dev/null || true

# Rename CSV to match config expectations
mv mainsurvey_data_with_space.csv mainsurvey_data.csv 2>/dev/null || true

# Move config to experiments/ directory
mkdir -p experiments
mv *.json experiments/ 2>/dev/null || true

# Create experiments/runs directory for output
mkdir -p experiments/runs

# Install missing packages to local directory
echo "Checking for required packages on $(hostname)..."
mkdir -p .local_packages
export PYTHONPATH="$PWD/.local_packages:$PYTHONPATH"

MISSING=""
TORCH_MISSING=0
python3 -c "import torch" 2>/dev/null || TORCH_MISSING=1
python3 -c "import numpy" 2>/dev/null || MISSING="$MISSING numpy"
python3 -c "import pandas" 2>/dev/null || MISSING="$MISSING pandas"
python3 -c "import sklearn" 2>/dev/null || MISSING="$MISSING scikit-learn"
python3 -c "import imblearn" 2>/dev/null || MISSING="$MISSING imbalanced-learn"
python3 -c "import tensorboard" 2>/dev/null || MISSING="$MISSING tensorboard"

# Install torch CPU-only (much smaller than default CUDA build)
if [ "$TORCH_MISSING" -eq 1 ]; then
    echo "Installing torch (CPU-only)..."
    python3 -m pip install --no-cache-dir --target=.local_packages torch --index-url https://download.pytorch.org/whl/cpu 2>&1 || {
        echo "ERROR: Failed to install torch"
        exit 1
    }
fi

if [ -n "$MISSING" ]; then
    echo "Installing missing packages:$MISSING"
    python3 -m pip install --no-cache-dir --target=.local_packages $MISSING 2>&1 || {
        echo "ERROR: Failed to install packages:$MISSING"
        exit 1
    }
fi

if [ "$TORCH_MISSING" -eq 0 ] && [ -z "$MISSING" ]; then
    echo "All packages available"
fi
echo "Setup complete"

# Extract just the filename from the config path argument
CONFIG_FILE=$(basename "$@")

# Run the experiment
echo "Starting experiment: $CONFIG_FILE"
python3 ML/main.py --config "experiments/$CONFIG_FILE"
