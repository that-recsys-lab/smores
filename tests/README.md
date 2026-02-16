# Tests Layout

This directory is organized into two parts:

- `tests/fixtures/config/`: YAML configs used by tests
- `tests/fixtures/data/`: small CSV fixtures used by tests

Code tests remain grouped by module domain:

- `tests/item/`
- `tests/recommender/`
- `tests/stakeholders/`
- `tests/trigger/`
- `tests/utils/`

Shared fixture paths are defined in `tests/paths.py`.
