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
echo "Metrics files:      $(ls metrics/ 2>/dev/null | wc -l | xargs)"
echo "Tensorboard dirs:   $(ls tensorboards/ 2>/dev/null | wc -l | xargs)"
echo "Model dirs:         $(ls models/ 2>/dev/null | wc -l | xargs)"
echo ""
echo "Total size:"
[ -d metrics/ ]      && echo "  Metrics:      $(du -sh metrics/ | awk '{print $1}')"
[ -d tensorboards/ ] && echo "  Tensorboards: $(du -sh tensorboards/ | awk '{print $1}')"
[ -d models/ ]       && echo "  Models:       $(du -sh models/ | awk '{print $1}')"
echo ""
echo "Results CSV: experiment_results.csv"
echo ""
echo "Top 5 results:"
head -6 experiment_results.csv | column -t -s,

# Cleanup downloaded tarball
rm -f ${LOCAL_RESULTS_DIR}_results.tar.gz

# Step 5: Generate plots (optional)
VISUALIZE_SCRIPT="../cluster/visualize_results.py"
if [ -f "$VISUALIZE_SCRIPT" ]; then
    echo "=================================="
    echo "Generating Visualization Plots"
    echo "=================================="
    python "$VISUALIZE_SCRIPT"
    echo "View plots in: ${LOCAL_RESULTS_DIR}/plots/"
else
    echo "(Skipping plots — visualize_results.py not found)"
fi

echo ""
echo "=================================="
echo "✅ Done! Results in: ${LOCAL_RESULTS_DIR}/"
echo "=================================="
