#!/bin/bash
# Submits a round's jobs. Runs on the cluster, from the project root.

set -e

# Round tag: namespaces logs/ so a new submission cannot clobber the logs of a round
# that is still running ($(Process) restarts at 0 every time). Results are not
# namespaced -- they merge into runs/ under their own experiment names.
# Derived from experiments.txt (12th_experiments/configs/x.json -> "12th"); override with $1.
if [ ! -s cluster/experiments.txt ]; then
    echo "ERROR: cluster/experiments.txt is missing or empty." >&2
    echo "Run: python3 cluster/generate_experiments_txt.py <round>_experiments" >&2
    exit 1
fi
# The tag comes from the first line, so every line must belong to the same round --
# otherwise one round's jobs would be filed under another's logs/ and runs_*.
MIXED=$(awk -F/ '{print $1}' cluster/experiments.txt | sort -u)
if [ "$(echo "$MIXED" | wc -l)" -gt 1 ]; then
    echo "ERROR: cluster/experiments.txt mixes rounds:" >&2
    echo "$MIXED" | sed 's/^/         /' >&2
    echo "       Generate and submit one round at a time." >&2
    exit 1
fi

ROUND="${1:-$(awk -F/ 'NR==1{print $1}' cluster/experiments.txt | sed 's/_experiments$//')}"
if [ -z "$ROUND" ]; then
    echo "ERROR: could not derive a round tag; pass one explicitly: $0 14th" >&2
    exit 1
fi

echo "================================================"
echo "Color Survey - HTCondor Job Submission"
echo "================================================"

# Refuse to submit a round that is already queued. The round tag keeps DIFFERENT rounds
# out of each other's logs, but two submissions of the SAME round share logs/<round>/ and
# are indistinguishable to it. (Happened once: clusters 606616/606617 both ran the 14th.)
# Override with FORCE=1 for a deliberate re-submit.
if command -v condor_q >/dev/null 2>&1; then
    QUEUED=$(condor_q -submitter "$USER" -af Args 2>/dev/null \
             | grep -c "^${ROUND}_experiments/" || true)
    if [ "${QUEUED:-0}" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
        echo "ERROR: round '$ROUND' already has $QUEUED job(s) in the queue." >&2
        echo "       Submitting again would make both write to logs/$ROUND/." >&2
        echo "       Inspect:  condor_q -submitter $USER -af ClusterId Args | grep ${ROUND}_experiments" >&2
        echo "       Re-submit anyway:  FORCE=1 $0 $ROUND" >&2
        exit 1
    fi
fi

# HTCondor will not create this itself — jobs go on hold if it is missing.
mkdir -p "logs/$ROUND"
echo "Round tag: $ROUND  (logs/$ROUND/)"
echo "Jobs to submit: $(wc -l < cluster/experiments.txt)"

# Make run_experiment.sh executable
chmod +x cluster/run_experiment.sh

# Submit jobs to HTCondor
echo "Submitting experiments to HTCondor..."
condor_submit -a "round=$ROUND" cluster/job.sub

echo
echo "✅ Jobs submitted successfully!"
echo
echo "================================================"
echo "Monitoring Commands:"
echo "================================================"
echo "  condor_q                    # View all jobs in queue"
echo "  condor_q -submitter $USER   # View only your jobs"
echo "  condor_status               # Check cluster resources"
echo "  condor_q -held              # View jobs with errors"
echo "  condor_q -analyze <job_id>  # Debug specific job"
echo "  condor_rm <job_id>          # Cancel specific job"
echo "  condor_rm $USER             # Cancel all your jobs"
echo
echo "================================================"
echo "Results will be in: runs/<experiment_name>/run_<timestamp>/"
echo "================================================"
echo
