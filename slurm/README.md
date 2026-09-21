# Running the hedging-frequency sweep on a cluster

Two stages, both shardable, written for Slurm and tested locally by running
the same shards serially. Stage 1 calibrates every archive day once; stage 2
sweeps the grid over that cache. The split matters: fitting is ~98% of the
compute and depends on none of the swept parameters, so re-fitting per cell
would burn ~1.9 CPU-hours recomputing 48 identical copies of the same 158
surfaces.

## Local is the default. Check before you submit.

The default 48-cell grid takes **~3 minutes on a 4-core laptop** — 2m31s to
fit, 46s to sweep. A cluster earns its keep only on a widened grid. The axis
that actually scales is the sweep: intervals x slippage levels x strikes x
tenors. The fit stage is bounded at 158 tasks by the archive itself and will
not get slower no matter how wide the sweep goes.

Measured throughput at `--workers 4`, which makes the decision arithmetic
rather than a guess:

| stage | rate | the whole archive / grid |
|---|---|---|
| fit | ~1.05 archive-day/s | 158 days → 2m31s |
| sweep | ~1.04 cell/s | 48 cells → 46s |

So projected local minutes ≈ `(archive_days + cells) / 60`, where
`cells = assets x intervals x slippage levels`. **Run it locally under ~30
minutes of projected time; submit above that**, or if a sweep would want more
RAM than the laptop has.

Worked examples: the default grid is (158 + 48)/60 ≈ 3 min — run it locally.
The widened `HEDGE_HOURS="1 2 3 4 6 8 12 24"` x `SLIPPAGE_BPS="0 1 2 3 5 7 10
15"` grid is 128 cells, (158 + 128)/60 ≈ 5 min — still local. You need
something like 1,700 cells before submitting beats sitting through it, which
means adding strike or tenor axes, not just finer hedging.

Widen until it hurts, *then* submit.

## What has to be on the cluster

`git clone` gets you the code and nothing else — every data directory is
gitignored. Two sets of files must be copied across:

```bash
rsync -avz data/historical/            adroit:~/btc-vol-surface/data/historical/
rsync -avz data/research/*_funding_hourly.parquet \
                                       adroit:~/btc-vol-surface/data/research/
```

That is ~105 MB of archive (158 files) plus ~3.8 MB of hourly index. Nothing
else is needed — `data/raw/` is the live collection and no stage reads it.

**Compute nodes have no outbound network.** `data/history.py` normally tops
its cache up from Deribit on every call, which on a firewalled node would
retry four times and then raise a connection error that reads like a data
problem. The sweep calls it with `offline=True`, which reads the parquet and
never opens a socket — hence the second rsync. Keep `collector.py` and
`fetch_archive.py` on a machine that can reach Deribit; move only the compute.

## Submitting

```bash
bash slurm/submit.sh
```

Submits stage 1, then stage 2 with `--dependency=afterok`, so a failed fit
task stops the sweep instead of letting it report a number computed from
whichever days happened to land. It also clears stale shards first — leftover
parquet from an earlier run would be globbed in alongside the new ones and
silently double-count days.

Widen the grid from the environment, without editing the file the job runs:

```bash
HEDGE_HOURS="1 2 3 4 6 8 12 24" SLIPPAGE_BPS="0 1 2 3 5 7 10 15" \
    bash slurm/submit.sh
```

Cells = assets x intervals x slippage levels, so that example is 2 x 8 x 8 =
128 cells; raise `--array` in `sweep.slurm` to match if you want one task per
few cells.

Then, on the login node or back on the laptop with the cells rsynced home:

```bash
python scripts/hedge_frequency.py report
```

## Check these before the first submit

Three things in `fit.slurm` and `sweep.slurm` are cluster-specific and I could
not verify them from here:

| Line | Confirm with |
|---|---|
| `module load anaconda3/2024.6` | `module avail anaconda3` |
| no `--partition` given (uses the default) | `sinfo -s`, `snodes` |
| no `--account` given | `sacctmgr show assoc user=$USER` if accounting is on |

Create the environment once on the cluster:

```bash
module load anaconda3/2024.6
conda create -n btc-vol python=3.11 --yes
conda activate btc-vol
pip install -r requirements.txt
```

`streamlit` and `plotly` in `requirements.txt` are for the dashboard and are
not needed by either stage; skip them if the install is awkward.

## Sharding

`--shard i/n` takes `items[i::n]` — strided, not contiguous. Archive days grow
from ~7k quotes in 2020 to ~22k in 2026, so contiguous chunks would hand one
unlucky task every expensive day while others idled. Each task writes its own
`fits_<i>of<n>.parquet` / `cells_<i>of<n>.parquet`; nothing is shared, nothing
locks, and the next stage globs whatever it finds.

Because the loader globs, a re-run at a *different* `n` would leave both
generations on disk and read them together — every day counted twice, means
barely moving, t inflating by ~sqrt(2). Both loaders refuse that rather than
averaging it: you get a `ValueError` naming the directory and the duplicate
count. `submit.sh` clears the directories first, and a local
`hedge_frequency.py run` clears them too, since it owns the whole study. A
bare `fit`/`sweep` does not — shards are each other's peers and must not
delete each other's work — so clear by hand if you run one of those.

## No GPU

Nothing here benefits from one. The hot loop is a 2-parameter Nelder-Mead over
an *n x 3* least-squares solve, n ~ 100-200 — microscopic arrays behind a
sequential optimiser, which is the shape GPUs are worst at. The parallelism is
across days and cells, at ~1 s granularity, which is what these array jobs
exploit. Do not request `--gres=gpu`.
