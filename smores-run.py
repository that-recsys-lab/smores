import random
import numpy as np
import pandas as pd
from smores.stakeholders.stakeholders import Document, Provider, Consumer, Recommender
from smores.preprocessing.buhayh_et_al_24 import buhayh_et_al_24
# from smores.preprocessing.generate_predictions import generate_predictions
from smores.preprocessing.genre_preferences import genre_preferences
from smores.recommender.SVD import SurpriseSVD
from smores.recommender.genre_calibrated_popularity import GenreCalibratedPopularity
from smores.simulation.monilithic_ecosystem import MonolithicEcosystem
from smores.utils.run import run_experiment
from smores.preprocessing.buhayh_et_al_24 import setup_providers
from smores.stakeholders.choice import category_similarity_logit

# Movie Lens
ratings_df = pd.read_csv("data/raw/ml-latest-small/ratings.csv")
movies_df = pd.read_csv("data/raw/ml-latest-small/movies.csv")
consumer_item_rating_genre_df = ratings_df.merge(movies_df, on="movieId", how="left")
items_df = (
    consumer_item_rating_genre_df[["movieId", "rating", "genres"]]
    .groupby(["movieId", "genres"])
    .mean()
    .reset_index()
)

# Buhay et al. 24 data preprocessing
try:
    print("Generating consumer preferences based on genre preferences...")

    # consumers_list, niche_consumers_set, mainstream_consumers_set, niche_items = (
    #     buhayh_et_al_24(consumer_item_rating_genre_df, niche_genre="Western")
    # )
    
    consumers_list, niche_consumers_set, mainstream_consumers_set, niche_items = (
        genre_preferences(consumer_item_rating_genre_df, niche_genre="Western", threshold=0.01)
    )

    print(f"Type of consumers_list: {type(consumers_list)}")
    print(f"Type of niche_consumers_set: {type(niche_consumers_set)}")
    print(f"Type of mainstream_consumers_set: {type(mainstream_consumers_set)}")
    print(f"Type of niche_items: {type(niche_items)}")

    print("-----------------------------------------------------------------")  

    print("Data preprocessing completed successfully.")
    print(f"Number of consumers: {len(consumers_list)}")
    print(f"Number of niche consumers: {len(niche_consumers_set)}")
    print(f"Number of mainstream consumers: {len(mainstream_consumers_set)}")
    print(f"Niche items percentage: {niche_items.shape[0] / items_df.shape[0] * 100:.2f}%")

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
    consumer_choice_model=category_similarity_logit,
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
            "name": "mainstream_recommender",
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
