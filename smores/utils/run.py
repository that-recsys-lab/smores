import pandas as pd
import numpy as np
from collections import Counter
import random
import os
import json
from datetime import datetime
from smores.stakeholders.stakeholders import Item, Provider, Consumer, Recommender


def create_experiment_directory(base_dir, experiment_name):
    """
    Create a directory for the experiment and return its path.

    Args:
        base_dir (str): Base directory for all experiments.
        experiment_name (str): Name of the experiment.

    Returns:
        str: Path to the experiment's run directory.
    """
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = os.path.join(experiment_dir, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def make_json_serializable(data):
    """
    Recursively converts non-serializable objects (e.g., sets) into JSON serializable types.

    Args:
        data: The object to be processed.

    Returns:
        A JSON serializable version of the object.
    """

    keys_to_skip = {
        "consumers",
        "providers",
    }

    if isinstance(data, set):
        return list(data)
    elif isinstance(data, dict):
        return {
            key: make_json_serializable(value)
            for key, value in data.items()
            if key not in keys_to_skip
        }
    elif isinstance(data, list):
        return [make_json_serializable(item) for item in data]
    else:
        return data


def save_experiment_parameters(params, save_path):
    """
    Save experiment parameters to a JSON file.

    Args:
        params (dict): Dictionary of experiment parameters.
        save_path (str): Path to save the JSON file.
    """
    serializable_params = make_json_serializable(params)
    with open(save_path, "w") as f:
        json.dump(serializable_params, f, indent=4)


def calculate_category_frequencies(recommenders, exp_name):
    """
    Calculate category frequencies from historical recommendations.

    Args:
        recommenders (dict): Dictionary of recommenders.
        exp_name (str): Name of the experiment.
    Returns:
        DataFrame: DataFrame containing category frequencies.
    """
    rows_to_append = []
    for recommender_id, recommender in recommenders.items():
        category_counter = Counter()
        for doc_list in recommender.historical_recommendations:
            for item in doc_list:
                for category in item.genres:
                    category_counter[category] += 1
        for category, frequency in category_counter.items():
            rows_to_append.append(
                {
                    "experiment_name": exp_name,
                    "recommender_id": recommender_id,
                    "category": category,
                    "frequency": frequency,
                }
            )

    return pd.DataFrame(rows_to_append)


def run_experiment(
    experiment,
    model,
    consumer_choice_model,
    experiment_name,
    num_days,
    num_cycles,
    slate_size,
    recommenders,
    consumers,
    providers,
    base_dir="experiments/results",
    **kwargs
):
    """
    Run the experiment with the given parameters.

    Args:
        experiment (function): Function to run the experiment.
        experiment_name (str): Name of the experiment.
        num_days (int): Number of days in the experiment.
        num_cycles (int): Number of cycles in the experiment.
        slate_size (int): Slate size for recommenders.
        recommenders (list): List of recommenders and their parameters.
        base_dir (str): Base directory to save experiment results.

    Returns:
        None
    """

    run_dir = create_experiment_directory(base_dir, experiment_name)

    # Save experiment parameters
    experiment_params = {
        "experiment_name": experiment_name,
        "num_days": num_days,
        "num_cycles": num_cycles,
        "slate_size": slate_size,
        "recommenders": [rec["params"] for rec in recommenders],
    }

    save_experiment_parameters(experiment_params, os.path.join(run_dir, "params.json"))

    # Initialize recommenders
    recommender_objects = {}
    for i, rec in enumerate(recommenders):
        recommender_type = rec["type"]
        recommender_name = rec["name"]
        recommender_params = rec["params"]
        recommender_id = f"{recommender_name}_{recommender_type.__name__}_{i}"  # Generate a unique ID for each recommender
        recommender_params["recommender_id"] = recommender_id

        # Use main consumers and providers if not provided
        recommender_params["consumers"] = recommender_params.get("consumers", consumers)
        recommender_params["providers"] = recommender_params.get("providers", providers)

        # Create the recommender object
        recommender_objects[recommender_id] = recommender_type(**recommender_params)

    print(f"Running experiment: {experiment_name}")

    # Run the experiment
    provider_df, consumer_df, recommender_df, consumer_recommender_df = experiment(
        consumers=consumers,
        providers=providers,
        recommenders=recommender_objects,
        num_days=num_days,
        slate_size=slate_size,
        num_cycles=num_cycles,
        model=model,
        **kwargs  # Forward extra keyword arguments (e.g., forget_interactions, transfer_interactions)
    )

    # Calculate category frequencies
    category_freq_df = calculate_category_frequencies(
        recommender_objects, experiment_name,
    )

    # Save results
    provider_df.to_parquet(os.path.join(run_dir, "provider_data.parquet"), index=False)
    consumer_df.to_parquet(os.path.join(run_dir, "consumer_data.parquet"), index=False)
    recommender_df.to_parquet(os.path.join(run_dir, "recommender_data.parquet"), index=False)
    consumer_recommender_df.to_parquet(
        os.path.join(run_dir, "consumer_recommender_data.parquet"), index=False
    )
    category_freq_df.to_parquet(
        os.path.join(run_dir, "category_frequencies.parquet"), index=False
    )

    print(f"Results saved in: {run_dir}")
    print("Experiment Complete!")