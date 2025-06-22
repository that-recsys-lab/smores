import argparse
import os
import yaml
from smores import Smores
from smores.utils import SmoresConfig

def read_args():
    parser = argparse.ArgumentParser(
        description="SMORES: Simulation MOdel for Recommender EcoSystems"
    )

    parser.add_argument("config_file", help="Path to the configuration file.")

    input_args = parser.parse_args()
    arg_check(vars(input_args))
    return vars(input_args)


def arg_check(input_args):
    config_file = input_args["config_file"]
    if not os.path.exists(config_file):
        print(
            f"Configuration file {config_file} not found. Working directory: {os.getcwd()} Exiting."
        )
        exit(-1)
    else:
        return

def load_config(config_file):
    try:
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
            config = SmoresConfig(**config_data)
            return config
    except (yaml.YAMLError) as e:
        print("Error loading configuration:", e)
        return None


if __name__ == "__main__":

    args = read_args()
    config = load_config(args["config_file"])

    if config == None:
        exit(-1)

    smores = Smores(config)

    smores.run_experiment()

    exit(0)