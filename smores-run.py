import os
import numpy as np
import pandas as pd
from smores.stakeholders.stakeholders import Item, Provider, Consumer, Recommender
from smores.preprocessing.genre_preferences import genre_preferences
from smores.recommender.SVD import SurpriseSVD
from smores.recommender.genre_calibrated_popularity import GenreCalibratedPopularity
from smores.simulation.monilithic_ecosystem import monilithic_ecosystem
from smores.simulation.threshold_swithcing import threshold_switching
from smores.utils.run import run_experiment
from smores.utils.train_model import train_model
from smores.stakeholders.choice import category_similarity_logit
from smores.preprocessing.create_item_objects import create_item_objects_from_csv
from smores.preprocessing.historical_distribution import historical_distribution
import pickle

# Define the dataset directory
# dataset_directory = "data/raw/ml-tmdb"
# experiment_name = "divergance_full"

# dataset_directory = "data/raw/ml-latest-small"
# experiment_name = "divergance_small"

dataset_directory = "data/raw/ml-1m"
experiment_name = "kl_divergance_1m-ml"

# Load datasets
print("Loading datasets...")
ratings_path = os.path.join(dataset_directory, "ratings.csv")
movies_path = os.path.join(dataset_directory, "provider_movies.csv")

ratings_df = pd.read_csv(ratings_path)
movies_df = pd.read_csv(movies_path)

# Merge datasets
consumer_item_rating_genre_df = ratings_df.merge(movies_df, on="movieId", how="left")
items_df = (
    consumer_item_rating_genre_df[["movieId", "rating", "genres", "providerId"]]
    .groupby(["movieId", "genres", "providerId"])
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
    print("Genrating consumer historical distribution...")
    # Generate historical distribution for consumers
    historical_distribution = historical_distribution(ratings_df, movies_df, dataset_directory)

    print("Generating consumer preferences based on genre preferences...")
    # Pass the dataset directory to genre_preferences
    consumers_list, consumers_set = genre_preferences(
        consumer_item_rating_genre_df, historical_distribution, dataset_directory
    )

    # Group by 'providerId' and create provider objects
    providers_list = []
    for provider_id, items in items_df.groupby("providerId"):
        provider_items = create_item_objects_from_csv(items, provider_id)
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


# Get a list of the most popular movies => movies with the most ratings
def get_top_movies(ratings_df, movies_df, n=30, genre=None):
    movie_popularity = (
        ratings_df.groupby("movieId").size().reset_index(name="num_ratings")
    )
    if genre:
        movies_with_genre = movies_df[
            movies_df["genres"].str.contains(genre, case=False, na=False)
        ]
        movie_popularity = movie_popularity[
            movie_popularity["movieId"].isin(movies_with_genre["movieId"])
        ]
    movie_popularity = movie_popularity.sort_values(by="num_ratings", ascending=False)
    most_popular_movie_ids = movie_popularity["movieId"].tolist()[:n]
    return most_popular_movie_ids

all_genres_most_popular_movie_ids = get_top_movies(ratings_df, movies_df, n=30)
animation_most_popular_movie_ids = get_top_movies(ratings_df, movies_df, n=30, genre="Animation")

# Clear up memory
del ratings_df, movies_df, consumer_item_rating_genre_df, items_df

# Run experiment
run_experiment(
    experiment=threshold_switching,
    model=model,
    consumer_choice_model=category_similarity_logit,
    experiment_name=experiment_name,
    num_days=10,
    num_cycles=5,
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
                "most_popular_movie_ids": all_genres_most_popular_movie_ids,
                "prohibited_categories": set(),
                "weighted_category": {},
                "specialized_categories": set(),
                "consumers": consumers_list,  # optional
                # "providers": recommender_1_providers # optional
            },
        },
        {
            "type": SurpriseSVD,
            "name": "niche_recommender",
            "params": {
                "fee_per_click": 0.1,
                "fee_per_show": 0.01,
                "base_fee": 0.0,
                "most_popular_movie_ids": animation_most_popular_movie_ids,
                "prohibited_categories": set(),
                "weighted_category": {},
                "specialized_categories": set(),
                "consumers": [], # optional
                # "providers": recommender_1_providers # optional
            },
        },
    ],
    base_dir="experiments/results",
)
