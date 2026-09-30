#!/bin/bash
# Harvest a round's results from the cluster.
#
#   bash cluster/fetch.sh 14th            replace the local copy of experiments/14th/
#   bash cluster/fetch.sh 14th --merge    add only what is missing
#
# Replace mode drops the local metrics/, tensorboards/, models/ and CSV and extracts
# the tarball fresh. Merge mode keeps existing files, copies in the new ones, then
# rebuilds the CSV from every local metrics file — use it while a round is still
# producing results.
#
# Runs locally. Reads connection details from .cluster.env.
set -e
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ROUND="${1:-}"
if [ -z "$ROUND" ]; then
    echo "usage: bash cluster/fetch.sh <round> [--merge]" >&2
    exit 1
fi
MERGE=0
[ "${2:-}" = "--merge" ] && MERGE=1

DEST="$REPO_ROOT/experiments/$ROUND"
TARBALL="${ROUND}_results.tar.gz"
mkdir -p "$DEST"

echo "=== Collecting on the cluster ==="
ssh "$CLUSTER_HOST" "cd $CLUSTER_PATH && python3 cluster/collect_results.py $ROUND --type clip"

echo
echo "=== Downloading $TARBALL ==="
scp "$CLUSTER_HOST:$CLUSTER_PATH/$TARBALL" "$DEST/"

cd "$DEST"
if [ "$MERGE" = "1" ]; then
    echo
    echo "=== Merging (existing files kept) ==="
    rm -rf _incoming && mkdir _incoming
    tar -xzf "$TARBALL" -C _incoming
    for dir in metrics tensorboards models; do
        src="_incoming/${ROUND}_results/$dir"
        [ -d "$src" ] || continue
        mkdir -p "$dir"
        before=$(ls "$dir" | wc -l | tr -d ' ')
        cp -Rn "$src"/* "$dir"/ 2>/dev/null || true
        after=$(ls "$dir" | wc -l | tr -d ' ')
        printf "  %-14s %s -> %s\n" "$dir" "$before" "$after"
    done
    rm -rf _incoming

    echo
    echo "=== Rebuilding CSV from all local metrics ==="
    cd "$REPO_ROOT"
    python3 cluster/collect_results.py "$ROUND" --local --type clip \
        --sort Colors,Loss_Type,Embed_Dim
    cd "$DEST"
else
    echo
    echo "=== Extracting (replacing local copy) ==="
    rm -rf metrics tensorboards models experiment_results.csv
    tar -xzf "$TARBALL"
    mv "${ROUND}_results"/* .
    rmdir "${ROUND}_results"
fi
rm -f "$TARBALL"

echo
echo "=== $ROUND ==="
for dir in metrics tensorboards models; do
    [ -d "$dir" ] && printf "  %-14s %s\n" "$dir" "$(ls "$dir" | wc -l | tr -d ' ')"
done
[ -f experiment_results.csv ] && \
    echo "  CSV entries    $(tail -n +2 experiment_results.csv | wc -l | tr -d ' ')"
