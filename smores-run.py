import random
import numpy as np
import pandas as pd
from smores.stakeholders.stakeholders import Document, Provider, Consumer, Recommender
from smores.samplers.samplers import provider_sampler, consumer_sampler
from smores.preprocessing.buhayh_et_al_24 import process_recommendation_data
from smores.recommender.SVD import SurpriseSVD
from smores.recommender.genre_calibrated_popularity import GenreCalibratedPopularity
from smores.simulation.monilithic_ecosystem import MonolithicEcosystem
from smores.utils.run import run_experiment


def setup_providers(items_df, niche_items, num_providers=10, num_docs=500):
    """
    Setup and return providers for the experiment.

    Args:
        items_df (DataFrame): All items.
        niche_items (DataFrame): Niche items.
        num_providers (int): Total number of providers.
        num_docs (int): Number of items per provider.

    Returns:
        list: All providers.
        list: Niche providers.
        set: IDs of niche providers.
    """
    # Create niche providers
    niche_providers_list = provider_sampler(
        niche_items,
        int(num_providers * 0.1),  # 10% niche providers
        len(niche_items),
        starting_id=1,
    )
    niche_providers_set = {provider.provider_id for provider in niche_providers_list}

    # Create mainstream providers
    mainstream_providers_list = provider_sampler(
        items_df,
        int(num_providers * 0.9),  # 90% mainstream providers
        num_docs,
        starting_id=len(niche_providers_list) + 1,
    )

    providers_list = niche_providers_list + mainstream_providers_list
    random.shuffle(providers_list)

    return providers_list, niche_providers_set


# Movie Lens
ratings_df = pd.read_csv("data/raw/ml-latest-small/ratings.csv")
movies_df = pd.read_csv("data/raw/ml-latest-small/movies.csv")
movies_with_ratings_df = ratings_df.merge(movies_df, on="movieId", how="left")
items_df = (
    movies_with_ratings_df[["movieId", "rating", "genres"]]
    .groupby(["movieId", "genres"])
    .mean()
    .reset_index()
)

# Buhay et al. 24 data preprocessing
try:
    consumers_list, niche_consumers_set, mainstream_consumers_set, niche_items = (
        process_recommendation_data(
            items_df, movies_with_ratings_df, niche_genre="Western"
        )
    )
    print("Data preprocessing completed successfully.")
    print(f"Number of consumers: {len(consumers_list)}")
    print(f"Number of niche consumers: {len(niche_consumers_set)}")
    print(f"Number of mainstream consumers: {len(mainstream_consumers_set)}")
    print(f"Niche items: {niche_items.shape[0]}")

    # Setup providers
    providers_list, niche_providers_set = setup_providers(
        items_df, niche_items, num_providers=10, num_docs=500
    )
    print(f"Number of providers: {len(providers_list)}")
    print(f"Number of niche providers: {len(niche_providers_set)}")

except Exception as e:
    print(f"Error during data preprocessing or provider setup: {e}")


# Run experiment
run_experiment(
    experiment=MonolithicEcosystem,
    random_seed=42,
    experiment_name="Experiment_1",
    num_days=5,
    num_cycles=5,
    slate_size=5,
    niche_consumers_set=niche_consumers_set,
    niche_providers_set=niche_providers_set,
    consumers=consumers_list,
    providers=providers_list,
    recommenders=[
        {
            "type": SurpriseSVD,
            "params": {
                "fee_per_click": 0.1,
                "fee_per_show": 0.01,
                "base_fee": 0.1,
                "prohibited_categories": set(),
                "weighted_category": {},
                "specialized_categories": set(),
                # "consumers_list": recommender_1_consumers, # optional
                # "providers_list": recommender_1_providers # optional
            },
        },
    ],
    base_dir="experiments/results",
)
