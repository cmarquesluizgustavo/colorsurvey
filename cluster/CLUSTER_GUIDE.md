# HTCondor environment

Reference for the cluster environment. The project workflow is in `README.md`.

## Useful commands

```bash
condor_q -submitter $USER     # your jobs in the queue
condor_q -held                # jobs stopped by an error
condor_q -analyze <job_id>    # why a job matches no slot
condor_status                 # slots and free memory per node
condor_history -limit 10      # finished jobs
condor_rm <cluster_id>        # cancel one submission
condor_rm $USER               # cancel all your jobs
```

## Held jobs

Two common causes:

**Not enough memory.** Memory is requested per job, in the second column of
`cluster/experiments.txt`. To change it, adjust `MEMORY_MAP` in
`cluster/generate_experiments_txt.py` and regenerate the list. Nodes have different
per-slot memory tiers, so a larger request narrows the eligible slots and the job waits
longer in the queue.

**Output transfer failure.** HTCondor does not create intermediate directories on the
submit node: if the destination of a `transfer_output_remaps` does not exist, the
transfer fails, the job is held and its output is lost. The job log shows
`SHADOW ... failed to write to file ... (errno 2)`.

## Submitting a subset

Edit `cluster/experiments.txt` and comment out (with `#`) or delete the lines you do not
want to run. All lines must belong to the same round.

## GPU

One node has a GPU, but its driver is old and the installed PyTorch falls back to CPU
(`CUDA initialization: The NVIDIA driver on your system is too old`). Experiments run on
CPU; the warning in `.err` is expected.
