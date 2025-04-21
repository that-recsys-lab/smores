import os
import pickle
import numpy as np
import pandas as pd
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

def run_experiment_for_scenario(scenario, dataset_directory, experiment_name):
    print(f"\n{'-'*10} Starting scenario: {scenario['name']} {'-'*10}\n")

    # Load datasets
    print("Loading datasets...")
    ratings_path = os.path.join(dataset_directory, "ratings.csv")
    items_path = os.path.join(dataset_directory, "provider_items.csv")
    ratings_df = pd.read_csv(ratings_path)
    items_df = pd.read_csv(items_path)

    # Renaming columns for consistency
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

    # Merge datasets to get consumer, item, rating and genre info
    consumer_item_rating_genre_df = ratings_df.merge(items_df, on="itemId", how="left")
    items_df = (
        consumer_item_rating_genre_df[["itemId", "rating", "genres", "providerId"]]
        .groupby(["itemId", "genres", "providerId"])
        .mean()
        .reset_index()
    )

    # Check for an existing model; retrain if needed
    model_filename = os.path.join("experiments", "results", experiment_name, "model.pkl")
    if os.path.exists(model_filename):
        try:
            print("Loading existing model...")
            with open(model_filename, "rb") as model_file:
                model = pickle.load(model_file)
            print("Model loaded successfully.")
        except Exception as e:
            print("Error loading model:", e)
            print("Training a new model...")
            model = train_model(ratings_df, experiment_name)
    else:
        print("Model not found. Training a new model...")
        model = train_model(ratings_df, experiment_name)

    # Preprocessing: Build consumer historical distribution and preferences
    try:
        print("Generating consumer historical distribution...")
        historical_dist = historical_distribution(ratings_df, items_df, dataset_directory)
        print("Generating consumer preferences based on genre preferences...")
        consumers_list, consumers_set = genre_preferences(
            consumer_item_rating_genre_df, historical_dist, dataset_directory
        )

        # Get unique genres
        unique_genres = items_df["genres"].str.split("|").explode().unique()

        # Create provider objects grouped by providerId
        providers_list = []
        for provider_id, group in items_df.groupby("providerId"):
            provider_items = create_item_objects_from_csv(group, provider_id, unique_genres)
            # Each provider gets all its items 
            provider = Provider(provider_id=provider_id, items=provider_items)
            providers_list.append(provider)

        print("Data preprocessing completed successfully.")
        print(f"Number of consumers: {len(consumers_list)}")
        print(f"Number of providers: {len(providers_list)}")

    except Exception as e:
        print("Error during data preprocessing or provider setup:", e)
        raise

    # Define helper function to get popular items
    def get_top_items(ratings_df, items_df, n=30, genre=None):
        item_popularity = ratings_df.groupby("itemId").size().reset_index(name="num_ratings")
        if genre:
            items_df["first_genre"] = items_df["genres"].str.split("|").str[0]
            items_with_genre = items_df[items_df["first_genre"].str.lower() == genre.lower()]
            item_popularity = item_popularity[
                item_popularity["itemId"].isin(items_with_genre["itemId"])
            ]
        item_popularity = item_popularity.sort_values(by="num_ratings", ascending=False)
        return item_popularity["itemId"].tolist()[:n]

    # Get lists of popular items for the recommenders
    all_genres_most_popular_item_ids = get_top_items(ratings_df, items_df, n=30)
    niche_genres_most_popular_item_ids = get_top_items(ratings_df, items_df, n=30, genre="Horror")

    # Clear up memory (if needed)
    del ratings_df, items_df, consumer_item_rating_genre_df

    # Define experiment types (using threshold_switching as an example)
    experiments = [
        ("threshold_switching", threshold_switching),
        #("monolithic", monolithic),
        ("ucb_switching", ucb_switching),
    ]

    # Run experiment(s) for this scenario
    for exp_name, experiment_func in experiments:
        scenario_experiment_name = f"{experiment_name}_{exp_name}_{scenario['name']}"
        print(f"\nRunning {exp_name} for scenario: {scenario['name']} with settings: {scenario}")
        run_experiment(
            experiment=experiment_func,
            model=model,
            consumer_choice_model=category_similarity_logit,
            experiment_name=scenario_experiment_name,
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
                        "consumers": consumers_list,
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
                        "specialized_genres": {"Horror"},
                        "consumers": [],
                    },
                },
            ],
            base_dir="experiments/results",
            forget_interactions=scenario["forget_interactions"],
            transfer_interactions=scenario["transfer_interactions"]
        )

    print(f"\n{'-'*10} Scenario '{scenario['name']}' completed. {'-'*10}\n")

if __name__ == "__main__":
    # Set the dataset directory and experiment name as before
    dataset_directory = "data/raw/ml-1m"
    experiment_name = "ml-1m-ucb-switching"

    scenarios = [
        {"name": "cold_start", "forget_interactions": True, "transfer_interactions": False},
        {"name": "user_ownership", "forget_interactions": True, "transfer_interactions": True},
        {"name": "universal_profile", "forget_interactions": False, "transfer_interactions": True},
        {"name": "algorithm_specific_profile", "forget_interactions": False, "transfer_interactions": False},
    ]

    for scenario in scenarios:
        run_experiment_for_scenario(scenario, dataset_directory, experiment_name)

    print("All scenarios completed. Exiting now.")