import argparse
from pathlib import Path

import yaml

from smores import Smores
from smores.utils import SmoresConfig


def _resolve_path(path_value: str, base_dir: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def load_config(config_file: str):
    config_path = Path(config_file).expanduser().resolve()

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        config = SmoresConfig.model_validate(config_data)

        # Keep quickstart runs self-contained regardless of current working directory.
        config_dir = config_path.parent
        config.data.directory = _resolve_path(config.data.directory, config_dir)
        config.output.directory = _resolve_path(config.output.directory, config_dir)
        return config
    except (yaml.YAMLError) as e:
        print("Error loading configuration:", e)
        return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Execute SMORES simulation from configuration file"
    )
    parser.add_argument(
        "--config_file", required=True, help="Path to config YAML file"
    )

    args = parser.parse_args()

    config = load_config(args.config_file)

    if config is None:
        exit(-1)

    smores = Smores(config)

    smores.run_experiment()

    exit(0)
