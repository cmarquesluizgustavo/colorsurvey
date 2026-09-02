# Running experiments on the cluster

Jobs are submitted one round at a time. A round is a directory like `14th_experiments/`
holding a `generate_configs.py` and the `configs/*.json` it writes.

## Setup

`status.sh` and `fetch.sh` run on your machine and need connection details:

```bash
cp .cluster.env.example .cluster.env   # then fill it in
```

`.cluster.env` is gitignored. The scripts that run on the cluster itself take their
identity from the environment there and need no configuration.

## Submit a round

Both steps run **on the cluster**, from the project root:

```bash
python3 cluster/generate_experiments_txt.py 14th_experiments
bash cluster/submit.sh
```

`generate_experiments_txt.py` writes `cluster/experiments.txt` — one line per pending
config with its memory request — skipping any experiment that already has a metrics file
in the round. `submit.sh` derives the round tag from that list, creates `logs/<tag>/`,
and queues one job per line.

It refuses to submit when the list mixes rounds, or when that round already has jobs
queued; `FORCE=1 bash cluster/submit.sh 14th` overrides the second check.

## Watch it

```bash
bash cluster/status.sh            # queue, nodes, log and run counts
bash cluster/status.sh --errors   # plus the contents of non-empty .err files
bash cluster/status.sh --logs     # plus a tail of the most recent .out
```

Note that `.err` is non-empty for healthy runs too — matplotlib and torch both warn there.

## Fetch results

```bash
bash cluster/fetch.sh 14th_experiments            # replace the local copy
bash cluster/fetch.sh 14th_experiments --merge    # add only what is missing
```

Use `--merge` while a round is still producing results: it keeps what you already have,
copies in the rest, and rebuilds `experiment_results.csv` from every local metrics file.

A run that ends with a non-finite loss is recorded as `Status: diverged` — its retrieval
metrics are computed from NaN scores and are meaningless. Filter on that column.

## Where things land

Each job writes one run directory, returned to a shared `runs/` on the cluster:

```
runs/<experiment_name>/run_<timestamp>/
├── config.json
├── metrics.csv          # one row per epoch
├── events.out.tfevents.*
├── training_curves.png
└── models/{best_,}color_clip_model.pth
```

Run directories are unique per job, so they merge rather than collide. Logs are the
exception — HTCondor names them by job number, which restarts at 0 on every submission,
so they are namespaced as `logs/<round>/experiment_<job>.{log,out,err}`. Rounds submitted
before that existed are at `logs/experiment_<job>.*`; the scripts read both.

`fetch.sh` brings a round back as `metrics/`, `models/`, `tensorboards/` and
`experiment_results.csv` under the round directory.

## Memory

`generate_experiments_txt.py` requests 1000 MB for 14- and 96-colour runs and 2000 MB
above that. The larger request exceeds the smallest slot tier, so those jobs only match
the bigger workers — expect them to queue longer.
