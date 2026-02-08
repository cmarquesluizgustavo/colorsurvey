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
mv models/*.py ML/models/ 2>/dev/null || true
mv trainers/*.py ML/trainers/ 2>/dev/null || true
mv losses/*.py ML/losses/ 2>/dev/null || true
mv __pycache__ ML/ 2>/dev/null || true

# Move config to experiments/ directory
mkdir -p experiments
mv *.json experiments/ 2>/dev/null || true

# Create experiments/runs directory for output
mkdir -p experiments/runs

# Install missing packages to local directory (avoid /nonexistent home issue)
echo "Checking for required packages on $(hostname)..."
python3 -c "import torch; import numpy; import pandas; import sklearn; print('Core packages available')" 2>&1

# Only install packages that might be missing
echo "Installing minimal additional packages..."
mkdir -p .local_packages
export PYTHONPATH="$PWD/.local_packages:$PYTHONPATH"
python3 -m pip install --no-cache-dir --target=.local_packages imbalanced-learn tensorboard 2>&1 || echo "Note: Some packages may have failed to install, attempting to continue..."
echo "Setup complete"

# Extract just the filename from the config path argument
CONFIG_FILE=$(basename "$@")

# Run the experiment
echo "Starting experiment: $CONFIG_FILE"
python3 ML/main.py --config "experiments/$CONFIG_FILE"
