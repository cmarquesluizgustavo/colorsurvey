#!/bin/bash
# Cluster and job status, in one ssh session.
#
#   bash cluster/status.sh              queue, nodes, log and run counts
#   bash cluster/status.sh --errors     also print the contents of non-empty .err files
#   bash cluster/status.sh --logs       also tail the most recently written .out
#
# Runs locally. Reads connection details from .cluster.env.
set -e
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SHOW_ERRORS=0
SHOW_LOGS=0
for arg in "$@"; do
    case "$arg" in
        --errors) SHOW_ERRORS=1 ;;
        --logs)   SHOW_LOGS=1 ;;
        *) echo "unknown option: $arg" >&2; exit 1 ;;
    esac
done

ssh "$CLUSTER_HOST" bash -s -- \
    "$CLUSTER_PATH" "$CLUSTER_USER" "$SHOW_ERRORS" "$SHOW_LOGS" <<'REMOTE'
set -u
PROJECT="$1"; USER_NAME="$2"; SHOW_ERRORS="$3"; SHOW_LOGS="$4"
cd "$PROJECT" || { echo "ERROR: no $PROJECT on $(hostname); check CLUSTER_PATH" >&2; exit 1; }

echo "=== QUEUE ==="
condor_q -submitter "$USER_NAME" -af ClusterId Args 2>/dev/null \
  | awk '{id=$1; n[id]++; if(!(id in first)) first[id]=$2}
         END {for (i in n) printf "  cluster %s : %3d jobs : %s\n", i, n[i], first[i]}' \
  | sort
condor_q -submitter "$USER_NAME" -totals 2>/dev/null | grep "Total for query" || echo "  (queue empty)"

if [ "$(condor_q -submitter "$USER_NAME" -held -af ClusterId 2>/dev/null | wc -l)" -gt 0 ]; then
    echo; echo "=== HELD ==="
    condor_q -submitter "$USER_NAME" -held 2>/dev/null | head -12
fi

echo
echo "=== NODES ==="
# The FreeGb column here is the per-node free memory; no need to probe the workers.
condor_status -compact 2>/dev/null || echo "  (condor_status unavailable)"

# Logs live at logs/experiment_N.* for rounds submitted before round-namespacing and at
# logs/<round>/experiment_N.* after it, so every lookup here is recursive.
echo
echo "=== LOGS ==="
printf "  %-6s %s\n" ".log" "$(find logs -name 'experiment_*.log' 2>/dev/null | wc -l | tr -d ' ')"
printf "  %-6s %s\n" ".out" "$(find logs -name 'experiment_*.out' 2>/dev/null | wc -l | tr -d ' ')"
ERR_COUNT=$(find logs -name 'experiment_*.err' -size +0 2>/dev/null | wc -l | tr -d ' ')
printf "  %-6s %s non-empty\n" ".err" "$ERR_COUNT"
DONE=$(find logs -name 'experiment_*.log' 2>/dev/null | xargs -r grep -l '005.*Job terminated' 2>/dev/null | wc -l | tr -d ' ')
echo "  jobs terminated: $DONE"

echo
echo "=== RUNS ==="
echo "  completed run dirs: $(ls -d runs/*/run_* experiments/runs/*/run_* experiments/runs_*/*/run_* 2>/dev/null | wc -l | tr -d ' ')"

if [ "$ERR_COUNT" -gt 0 ] && [ "$SHOW_ERRORS" = "1" ]; then
    echo
    echo "=== ERRORS ==="
    for f in $(find logs -name 'experiment_*.err' -size +0 2>/dev/null | head -5); do
        echo "--- $f"
        tail -15 "$f"
    done
fi

if [ "$SHOW_LOGS" = "1" ]; then
    echo
    echo "=== NEWEST .out ==="
    NEWEST=$(find logs -name 'experiment_*.out' -size +0 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
    if [ -n "$NEWEST" ]; then
        echo "--- $NEWEST"
        tail -30 "$NEWEST"
    else
        echo "  (no output yet)"
    fi
fi
REMOTE
