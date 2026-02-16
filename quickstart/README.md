# Quickstart Mini Experiment

This directory is a self-contained SMORES mini experiment using precomputed
`data/simulation` files. It shows the minimum setup to run a full simulation
and inspect outputs.

## What Is Included

- `ml1m.yaml`: runnable simulation config
- `data/simulation/`: consumers, providers, items, and popularity files
- `bin/run_smores.py`: run one config directly

## Prerequisites

Run from the repo root (`smores`) first:

```bash
uv sync
source .venv/bin/activate
```

## Quickstart Run

From the repo root:

```bash
python quickstart/bin/run_smores.py --config_file quickstart/ml1m.yaml
```

Outputs are written to `quickstart/output/ml1m-mini/`.

## Expected Output Files

- `consumer_utility.parquet`
- `provider_utility.parquet`
- `rchoice_utility.parquet`
- `cycle_metrics.parquet`
- `item_stats.parquet`
- `user_journeys.parquet`
- `debug_*.log`

## Notes

- This mini setup intentionally uses only `data/simulation`.
- The runner resolves relative paths based on `ml1m.yaml`, so output stays
  inside `quickstart/` regardless of launch directory.
