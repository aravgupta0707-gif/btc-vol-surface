#!/bin/bash
# Chain both stages: fit, then sweep once the fits are all on disk.
#
#   bash slurm/submit.sh
#   HEDGE_HOURS="1 2 3 4 6 8 12 24" bash slurm/submit.sh    # widened grid
#
# The dependency is afterok, so a failed fit task stops the sweep rather than
# letting it run against a half-written cache and report a result computed
# from whichever days happened to land.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p slurm/logs data/research/hedge_fits data/research/hedge_cells

# Stale shards from an earlier run would be globbed in alongside the new ones
# and silently double-count days, so clear them first.
rm -f data/research/hedge_fits/*.parquet data/research/hedge_cells/*.parquet

fit_id=$(sbatch --parsable slurm/fit.slurm)
echo "fit    job $fit_id"

sweep_id=$(sbatch --parsable --dependency=afterok:"$fit_id" \
    --export=ALL,HEDGE_HOURS,SLIPPAGE_BPS,Z_GRID,TENOR_DAYS slurm/sweep.slurm)
echo "sweep  job $sweep_id  (waits on $fit_id)"

echo
echo "watch:   squeue -u \$USER"
echo "report:  python scripts/hedge_frequency.py report"
