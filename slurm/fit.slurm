#!/bin/bash
# Stage 1: fit every archive day's surface once and bank the SVI parameters
# and the measured half-spread curve. This is ~98% of the study's compute and
# does not depend on any swept parameter, so it runs once per archive, not
# once per grid cell.
#
#   sbatch slurm/fit.slurm
#
# Sizing: 158 archive days at ~1.8 s each is ~5 CPU-minutes total. That is
# small enough that the array exists for elasticity rather than necessity --
# 8 tasks x 8 cores finishes in well under a minute, and one task with
# --cpus-per-task=16 and no array would also be fine. See slurm/README.md.

#SBATCH --job-name=volfit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --array=0-7
#SBATCH --output=slurm/logs/fit_%A_%a.out
#SBATCH --error=slurm/logs/fit_%A_%a.out

set -euo pipefail

module purge
# Confirm the exact name with `module avail anaconda3` -- versions move.
module load anaconda3/2024.6
conda activate btc-vol

cd "$SLURM_SUBMIT_DIR"

# Each task writes fits_<i>of<n>.parquet and spreads_<i>of<n>.parquet. No two
# tasks touch the same file, so there is no locking and no combine step; the
# sweep stage globs whatever is there.
srun python scripts/hedge_frequency.py fit \
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" \
    --workers "${SLURM_CPUS_PER_TASK}"
