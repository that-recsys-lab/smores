# SMORES

SMORES (Simulator for Modular Recommendation EcoSystems) is a simulation
framework for studying recommender behavior, user switching, and ecosystem
dynamics under different modeling assumptions.

## Install

```bash
uv sync
source .venv/bin/activate
uv pip install -e .
```

## Quickstart

Run the included mini experiment:

```bash
python quickstart/bin/run_smores.py --config_file quickstart/ml1m.yaml
```

Output files are written to `quickstart/output/ml1m-mini/`.

For details, see `quickstart/README.md`.

## Run Tests

```bash
pytest -q
```

## Configuration

Configuration structure and supported `class_name` values are documented in
`docs/config.md`.

## Repository Layout

- `smores/`: core simulation package
- `quickstart/`: minimal runnable example
- `tests/`: unit and integration tests with fixtures
- `docs/`: user-facing documentation

## License

This project is licensed under `LICENSE`.
