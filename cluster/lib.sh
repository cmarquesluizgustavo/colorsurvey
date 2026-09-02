# Loads .cluster.env for the scripts that talk to the cluster from this machine.
# Source it; do not execute it. Exits the caller if the file or a required value
# is missing.
#
# Exports: CLUSTER_HOST, CLUSTER_USER, CLUSTER_PATH, CLUSTER_NODES (optional),
# and REPO_ROOT.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER_ENV="$REPO_ROOT/.cluster.env"

if [ ! -f "$CLUSTER_ENV" ]; then
    echo "ERROR: $CLUSTER_ENV not found." >&2
    echo "       cp .cluster.env.example .cluster.env  and fill it in." >&2
    exit 1
fi

set -a
. "$CLUSTER_ENV"
set +a

for _var in CLUSTER_HOST CLUSTER_USER CLUSTER_PATH; do
    if [ -z "${!_var}" ]; then
        echo "ERROR: $_var is empty in $CLUSTER_ENV" >&2
        exit 1
    fi
done
unset _var
