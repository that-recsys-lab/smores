import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm  # Import tqdm for the progress bar


def historical_distribution(ratings_df, items_df, dataset_directory):
    """
    Generate historical distribution for consumers based on genre preferences.

    Args:
        ratings_df (DataFrame): DataFrame containing user-item ratings.
        items_df (DataFrame): DataFrame containing item genres.
        dataset_directory (str): Path to the dataset directory.

    Returns:
        dict: Consumer historical genre distribution as a dictionary.
    """
    distribution_file = os.path.join(
        dataset_directory, "consumer_historical_distribution.csv"
    )

    if os.path.exists(distribution_file):
        print("Loading consumer historical distribution from file...")
        historical_distribution_df = pd.read_csv(distribution_file, index_col=0)
    else:
        print("Generating consumer historical distribution... This might take a while")

        # Merge ratings with items on itemId
        items_df["genres"] = items_df["genres"].apply(lambda x: x.split("|"))
        merged = pd.merge(ratings_df, items_df, on="itemId")

        # Initialize a dictionary to hold user genre counts
        user_genre_counts = defaultdict(lambda: defaultdict(int))

        # Count genres for each user with a progress bar
        for _, row in tqdm(
            merged.iterrows(), total=len(merged), desc="Processing rows"
        ):
            user_id = row["consumerId"]
            genres = row["genres"]
            for genre in genres:
                if genre == "(no genres listed)":
                    continue
                user_genre_counts[user_id][genre] += 1

        # Convert counts to proportions
        user_genre_distribution = {}
        for user_id, genre_counts in user_genre_counts.items():
            total_count = sum(genre_counts.values())
            user_genre_distribution[user_id] = {
                genre: count / total_count for genre, count in genre_counts.items()
            }

        # Convert to DataFrame and save to file
        historical_distribution_df = pd.DataFrame.from_dict(
            user_genre_distribution, orient="index"
        ).fillna(0)
        historical_distribution_df.to_csv(distribution_file)
        print(f"Historical distribution saved to {distribution_file}")

    # Convert DataFrame back to dictionary
    historical_distribution_dict = historical_distribution_df.to_dict(orient="index")

    return historical_distribution_dict
