import os
import pandas as pd
from smores.stakeholders.stakeholders import Item, Provider, Consumer
from smores.preprocessing.genre_preferences import genre_preferences
from smores.recommender.SVD import SurpriseSVD
from smores.recommender.genre_calibrated_popularity import GenreCalibratedPopularity
from smores.simulation.threshold_swithcing import threshold_switching
from smores.utils.run import run_experiment
from smores.utils.train_model import train_model
from smores.stakeholders.choice import category_similarity_logit
from smores.preprocessing.create_item_objects import create_item_objects_from_csv
from smores.preprocessing.historical_distribution import historical_distribution
import pickle
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

# Define dataset and experiment name
dataset_directory = "data/raw/ml-1m"
experiment_name = "ml-1m-ucb-switching-full"

# Load datasets
print("Loading datasets...")
ratings_path = os.path.join(dataset_directory, "ratings.csv")
items_path = os.path.join(dataset_directory, "provider_items.csv")

ratings_df = pd.read_csv(ratings_path)
items_df = pd.read_csv(items_path)

# Rename columns if needed
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

# Load or train model
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

# Generate historical distribution and consumer preferences
print("Generating consumer historical distribution...")
historical_distribution = historical_distribution(ratings_df, items_df, dataset_directory)

print("Generating consumer preferences based on genre preferences...")
consumers_list, consumers_set = genre_preferences(consumer_item_rating_genre_df, historical_distribution, dataset_directory)

# Get unique genres from the dataset
unique_genres = items_df["genres"].str.split("|").explode().unique()

# Create Provider objects from grouped items
providers_list = []
for provider_id, items in items_df.groupby("providerId"):
    provider_items = create_item_objects_from_csv(items, provider_id, unique_genres)
    provider = Provider(provider_id=provider_id, items=provider_items)
    providers_list.append(provider)

print("Data preprocessing completed successfully.")
print(f"Number of consumers: {len(consumers_list)}")
print(f"Number of providers: {len(providers_list)}")

# Define helper function to get popular items
def get_top_items(ratings_df, items_df, n=30, genre=None):
    item_popularity = ratings_df.groupby("itemId").size().reset_index(name="num_ratings")
    if genre:
        items_df["first_genre"] = items_df["genres"].str.split("|").str[0]
        items_with_genre = items_df[items_df["first_genre"].str.lower() == genre.lower()]
        item_popularity = item_popularity[item_popularity["itemId"].isin(items_with_genre["itemId"])]
    item_popularity = item_popularity.sort_values(by="num_ratings", ascending=False)
    most_popular_item_ids = item_popularity["itemId"].tolist()[:n]
    return most_popular_item_ids

all_genres_most_popular_item_ids = get_top_items(ratings_df, items_df, n=30)
niche_genres_most_popular_item_ids = get_top_items(ratings_df, items_df, n=30, genre="Horror")

# Clean up large variables from memory
del ratings_df, items_df, consumer_item_rating_genre_df

# Run the full simulation experiment (5 days, 5 cycles, slate size 3)
run_experiment(
    experiment=lambda **kwargs: threshold_switching(**kwargs, forget_flag=False),  # or True
    model=model,
    consumer_choice_model=category_similarity_logit,
    experiment_name=experiment_name,
    num_days=5,      # Adjust as needed for a quick test
    num_cycles=1,    # A single cycle is often enough to verify behavior
    slate_size=3,
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
)

# Enhanced logging after experiment completion
print("Full simulation experiment complete.")
print("Summary of connected consumers:")
for rec_name, rec in {"Mainstream": consumers_list[0].connected_recommenders}.items():
    print(f"{rec_name}: {len(rec)}")
