 import os
import numpy as np
import pandas as pd
import random
import pickle
from smores.stakeholders.stakeholders import Item, Provider, Consumer, Recommender
from smores.preprocessing.genre_preferences import genre_preferences
from smores.recommender.SVD import SurpriseSVD
from smores.recommender.genre_calibrated_popularity import GenreCalibratedPopularity
from smores.simulation.monolithic import monolithic
from smores.simulation.threshold_swithcing import threshold_switching
from smores.simulation.ucb_switching import ucb_switching
from smores.utils.run import run_experiment
from smores.utils.train_model import train_model
from smores.stakeholders.choice import category_similarity_logit
from smores.preprocessing.create_item_objects import create_item_objects_from_csv
from smores.preprocessing.historical_distribution import historical_distribution

seeds = [
    #38429,  # Seed 1
     # 71653,  # Seed 2
   # 12947,  # Seed 3
     # 58201,  # Seed 4
    94736,  # Seed 5
]

# Set the base experiment name
base_experiment_name = "ml-1m-ucb-switching"

# Iterate through the selected seeds
for seed_value in seeds:
    print(f"\n\n======= RUNNING EXPERIMENT WITH SEED {seed_value} =======\n\n")
    
    # Set random seeds for reproducibility
    np.random.seed(seed_value)
    random.seed(seed_value)
    
    # Define the dataset directory
    dataset_directory = "data/raw/ml-1m"
    experiment_name = f"{base_experiment_name}-seed-{seed_value}"
    
    # Load datasets
    print("Loading datasets...")
    ratings_path = os.path.join(dataset_directory, "ratings.csv")
    items_path = os.path.join(dataset_directory, "provider_items.csv")
    
    ratings_df = pd.read_csv(ratings_path)
    items_df = pd.read_csv(items_path)
    
    # Renaming columns
    items_df.rename(columns={
        "movieId": "itemId", 
        "genres": "genres", 
        "providerId": "providerId"
        }, inplace=True)
    ratings_df.rename(columns={
        "userId": "consumerId", 
        "movieId": "itemId", 
        "rating": "rating"
        }, inplace=True)
    
    # Merge datasets
    consumer_item_rating_genre_df = ratings_df.merge(items_df, on="itemId", how="left")
    items_df = (
        consumer_item_rating_genre_df[["itemId", "rating", "genres", "providerId"]]
        .groupby(["itemId", "genres", "providerId"])
        .mean()
        .reset_index()
    )
    
    # Check if the model already exists
    print("Checking for existing model...")
    model_filename = f"experiments/results/{experiment_name}/model.pkl"
    
    if os.path.exists(model_filename):
        try:
            print(f"Loading model from {model_filename}...")
            with open(model_filename, "rb") as model_file:
                model = pickle.load(model_file)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a new model...")
            model = train_model(ratings_df, experiment_name)
    else:
        print("Model not found. Training a new model...")
        model = train_model(ratings_df, experiment_name)
    
    try:
        print("Generating consumer historical distribution...")
        # Generate historical distribution for consumers
        hist_distribution = historical_distribution(ratings_df, items_df, dataset_directory)
    
        print("Generating consumer preferences based on genre preferences...")
        # Pass the dataset directory to genre_preferences
        consumers_list, consumers_set = genre_preferences(
            consumer_item_rating_genre_df, hist_distribution, dataset_directory
        )
    
        # Get unique genres from the dataset
        unique_genres = items_df["genres"].str.split("|").explode().unique()
    
        # Group by 'providerId' and create provider objects
        providers_list = []
        for provider_id, items in items_df.groupby("providerId"):
            provider_items = create_item_objects_from_csv(items, provider_id, unique_genres)
            provider = Provider(
                provider_id=provider_id,
                items=provider_items,
            )
            providers_list.append(provider)
    
        print("Data preprocessing completed successfully.")
        print(f"Number of consumers: {len(consumers_list)}")
        print(f"Number of providers: {len(providers_list)}")
    
    except Exception as e:
        print(f"Error during data preprocessing or provider setup: {e}")
        raise
    
    # Get a list of the most popular items => items with the most ratings
    def get_top_items(ratings_df, items_df, n=30, genre=None):
        # Calculate item popularity by the number of ratings
        item_popularity = (
            ratings_df.groupby("itemId").size().reset_index(name="num_ratings")
        )
    
        if genre:
            # Extract the first genre for each item
            items_df["first_genre"] = items_df["genres"].str.split("|").str[0]
            items_with_genre = items_df[items_df["first_genre"].str.lower() == genre.lower()]
            item_popularity = item_popularity[
                item_popularity["itemId"].isin(items_with_genre["itemId"])
            ]
    
        item_popularity = item_popularity.sort_values(by="num_ratings", ascending=False)
        most_popular_item_ids = item_popularity["itemId"].tolist()[:n]
        return most_popular_item_ids
    
    all_genres_most_popular_item_ids = get_top_items(ratings_df, items_df, n=30)
    niche_genres_most_popular_item_ids = get_top_items(ratings_df, items_df, n=30, genre="Horror")
    
    # Clear up memory
    del ratings_df, items_df, consumer_item_rating_genre_df
    
    scenarios = {
        "coldstart": {"FORGET": True, "TRANSFER": False},
        "user_ownership": {"FORGET":True, "TRANSFER": True},
        "universal_profile": {"FORGET": False, "TRANSFER": True},
        "algorithm_specific": {"FORGET": False, "TRANSFER": False},
    }
    
    # Run experiment
    run_experiment(
        experiment=ucb_switching,
        model=model,
        consumer_choice_model=category_similarity_logit,
        experiment_name=experiment_name,
        num_days=10,
        num_cycles=10,
        slate_size=5,
        consumers=consumers_list,
        providers=providers_list,
        recommenders=[
            {
                "type": SurpriseSVD,
                "name": "mainstream_recommender",
                "params": {
                    "fee_per_click": 0.1,
                    "fee_per_show": 0.01,
                    "base_fee": 0.0,
                    "most_popular_item_ids": all_genres_most_popular_item_ids,
                    "prohibited_genres": set(),
                    "weighted_category": {},
                    "specialized_genres": set(),
                    "consumers": consumers_list,  # optional
                },
            },
            {
                "type": SurpriseSVD,
                "name": "niche_recommender",
                "params": {
                    "fee_per_click": 0.1,
                    "fee_per_show": 0.01,
                    "base_fee": 0.0,
                    "most_popular_item_ids": niche_genres_most_popular_item_ids,
                    "prohibited_genres": set(),
                    "weighted_category": {},
                    "specialized_genres": set(["Horror"]),
                    "consumers": [], # optional
                },
            },
        ],
        forget_interactions=scenarios["algorithm_specific"]["FORGET"],
        transfer_interactions=scenarios["algorithm_specific"]["TRANSFER"],
        base_dir="experiments/results",
    )
    
    print(f"\n\n======= COMPLETED EXPERIMENT WITH SEED {seed_value} =======\n\n")