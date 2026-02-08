#!/bin/bash

# Local script to fetch results from cluster
# Usage: ./fetch_results.sh [experiment_folder]
#   e.g., ./fetch_results.sh 4th_experiments

set -e

CLUSTER_USER="my_cluster"
CLUSTER_PATH="~/colorsurvey"
LOCAL_RESULTS_DIR="${1:-3rd_experiments}"

echo "=================================="
echo "Fetching Results from Cluster"
echo "=================================="
echo "Target directory: $LOCAL_RESULTS_DIR"
echo ""

# Step 1: Run collection script on cluster
echo ""
echo "Step 1: Running collection script on cluster..."
ssh $CLUSTER_USER "cd $CLUSTER_PATH && python3 cluster/collect_results.py $LOCAL_RESULTS_DIR"

# Step 2: Download tarball
echo ""
echo "Step 2: Downloading results tarball..."
cd $LOCAL_RESULTS_DIR
scp $CLUSTER_USER:$CLUSTER_PATH/${LOCAL_RESULTS_DIR}_results.tar.gz .

# Step 3: Extract and organize
echo ""
echo "Step 3: Extracting and organizing files..."
echo "  - Removing old files..."
rm -rf metrics tensorboards models experiment_results.csv

echo "  - Extracting tarball..."
tar -xzf ${LOCAL_RESULTS_DIR}_results.tar.gz

echo "  - Moving files to current directory..."
mv ${LOCAL_RESULTS_DIR}_results/* .
rmdir ${LOCAL_RESULTS_DIR}_results

# Step 4: Verify results
echo ""
echo "=================================="
echo "Results Summary"
echo "=================================="
echo "Metrics files:      $(ls metrics/ | wc -l | xargs)"
echo "Tensorboard dirs:   $(ls tensorboards/ | wc -l | xargs)"
echo "Model dirs:         $(ls models/ | wc -l | xargs)"
echo ""
echo "Total size:"
echo "  Metrics:      $(du -sh metrics/ | awk '{print $1}')"
echo "  Tensorboards: $(du -sh tensorboards/ | awk '{print $1}')"
echo "  Models:       $(du -sh models/ | awk '{print $1}')"
echo ""
echo "Results CSV: experiment_results.csv"
echo ""

# Step 5: Generate plots
echo "=================================="
echo "Generating Visualization Plots"
echo "=================================="
python ../cluster/visualize_results.py

echo ""
echo "=================================="
echo "✅ Done!"
echo "=================================="
echo ""
echo "View plots in: 3rd_experiments/plots/"
echo "Launch TensorBoard: ./cluster/launch_tensorboard.sh"
