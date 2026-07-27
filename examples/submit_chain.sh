#!/bin/bash
# Submit a chain of train_caravan.sbatch jobs, each depending on the previous one (--dependency=afterany), so
# training continues automatically across the cluster's 72h max walltime until all epochs are done.
#
# caravan_training.py resumes from the last saved checkpoint on every (re)start, so:
#   - Each job in the chain is safe to run even if an earlier one finished all epochs already (it just skips
#     straight to the testing phase and exits quickly).
#   - If you need more or fewer jobs than the default, pass the count as the first argument.
#
# Usage: ./submit_chain.sh [num_jobs]

set -euo pipefail
cd "$(dirname "$0")"

NUM_JOBS="${1:-3}"  # 3 x 72h = 216h of budget for an estimated ~90h run, with margin for queueing/variance

prev_jobid=""
for i in $(seq 1 "$NUM_JOBS"); do
    if [[ -z "$prev_jobid" ]]; then
        jobid=$(sbatch --parsable train_caravan.sbatch)
    else
        jobid=$(sbatch --parsable --dependency=afterany:"$prev_jobid" train_caravan.sbatch)
    fi
    echo "Submitted job $i/$NUM_JOBS: $jobid$( [[ -n "$prev_jobid" ]] && echo " (after $prev_jobid)" )"
    prev_jobid="$jobid"
done

echo
echo "Chain submitted. Monitor with: squeue -u $USER"
