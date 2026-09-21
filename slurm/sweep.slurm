#!/bin/bash
# Stage 2: sweep the (asset x hedge interval x slippage) grid over the fits
# stage 1 banked. Cheap per cell -- the surfaces are already calibrated, so
# each cell is a delta-hedge walk over cached parameters.
#
#   sbatch --dependency=afterok:<fit job id> slurm/sweep.slurm
#
# The grid axes come from the environment, so widening the sweep never means
# editing the file the job runs from. Every task in the array must agree on
# the axes, which is why they are exported here rather than passed per task.

#SBATCH --job-name=volsweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=slurm/logs/sweep_%A_%a.out
#SBATCH --error=slurm/logs/sweep_%A_%a.out

set -euo pipefail

module purge
module load anaconda3/2024.6
conda activate btc-vol

cd "$SLURM_SUBMIT_DIR"

# Defaults reproduce the 48-cell grid in HANDOFF.md section 11.6. Override any
# axis at submit time, e.g.
#   HEDGE_HOURS="1 2 3 4 6 8 12 24" SLIPPAGE_BPS="0 1 2 3 5 7 10 15" \
#       sbatch --export=ALL,HEDGE_HOURS,SLIPPAGE_BPS slurm/sweep.slurm
# and raise --array to match: cells = assets x intervals x slippage levels.
export HEDGE_HOURS="${HEDGE_HOURS:-1 2 4 6 12 24}"
export SLIPPAGE_BPS="${SLIPPAGE_BPS:-0 2 5 10}"
export Z_GRID="${Z_GRID:--2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0}"
export TENOR_DAYS="${TENOR_DAYS:-7 30}"

echo "grid: hedge=[$HEDGE_HOURS] slip=[$SLIPPAGE_BPS] tenor=[$TENOR_DAYS]"

srun python scripts/hedge_frequency.py sweep \
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" \
    --workers "${SLURM_CPUS_PER_TASK}"
