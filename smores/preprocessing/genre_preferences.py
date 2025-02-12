import os
import pandas as pd
import random
import numpy as np
from collections import defaultdict
from smores.stakeholders.stakeholders import Consumer
from tqdm import tqdm


def genre_preferences(consumer_item_rating_genre_df, historical_distribution, dataset_directory):
    """
    Generate consumer preferences based on genre counts.

    Args:
        consumer_item_rating_genre_df (DataFrame): DataFrame containing user-item ratings and genres.
        dataset_directory (str): Path to the dataset directory.

    Returns:
        tuple: List of Consumer instances and a set of consumer IDs.
    """
    preferences_file = os.path.join(
        dataset_directory, "normalized_category_preferences.csv"
    )

    if os.path.exists(preferences_file):
        print("Loading normalized category preferences from file...")
        normalized_category_preferences_df = pd.read_csv(preferences_file, index_col=0)
    else:
        print("Generating normalized category preferences... This might take a while")
        normalized_category_preferences_df = generate_genre_preferences(
            consumer_item_rating_genre_df
        )
        normalized_category_preferences_df.to_csv(preferences_file)
        print(f"Normalized preferences saved to {preferences_file}")

    # Create consumers list
    consumers_list = consumer_sampler(normalized_category_preferences_df, historical_distribution)
    random.shuffle(consumers_list)

    consumers_set = set(consumer.consumer_id for consumer in consumers_list)
    return consumers_list, consumers_set


def normalize_counts(counts):
    """
    Normalize genre counts to sum to 1.

    Args:
        counts (dict): Dictionary of genre counts.

    Returns:
        dict: Normalized genre preferences.
    """
    total = sum(counts.values())
    if total == 0:
        return {genre: 0 for genre in counts}  # Handle case where total is zero
    return {genre: count / total for genre, count in counts.items()}


def process_user_genre_preferences(user_data):
    """
    Calculate genre preferences for a single user based on item counts and normalize them.

    Args:
        user_data (DataFrame): DataFrame of a single user's ratings and genres.

    Returns:
        DataFrame: User's genre preferences normalized.
    """
    user_data = user_data.dropna(subset=["genres"])
    
    # Step 1: Count items per genre
    genre_counts = defaultdict(int)
    for _, row in user_data.iterrows():
        genres = row["genres"].split("|")
        for genre in genres:
            genre_counts[genre] += 1

    # Step 2: Normalize counts
    normalized_counts = normalize_counts(genre_counts)

    # Convert to DataFrame
    genre_df = pd.DataFrame(
        [{"genre": genre, "normalized_preference": pref} for genre, pref in normalized_counts.items()]
    )

    return genre_df


def generate_genre_preferences(df):
    """
    Process user-by-user genre preferences and normalize them based on counts.

    Args:
        df (DataFrame): Input DataFrame with consumerId, itemId, rating, genres.

    Returns:
        DataFrame: Normalized genre preferences for all users.
    """
    user_ids = df["consumerId"].unique()

    # Initialize an empty list to store results
    all_genre_preferences = []

    # Process each user independently with a progress bar
    for user_id in tqdm(user_ids, desc="Processing Users", unit="user"):
        user_data = df[df["consumerId"] == user_id]  # Filter data for one user
        user_preferences = process_user_genre_preferences(user_data)

        # Add the user ID
        user_preferences["consumerId"] = user_id
        all_genre_preferences.append(user_preferences)

    # Combine all results into a single DataFrame
    result_df = pd.concat(all_genre_preferences, ignore_index=True)

    # Pivot the DataFrame to create a user-genre matrix
    normalized_preferences = result_df.pivot(
        index="consumerId", columns="genre", values="normalized_preference"
    ).fillna(0)

    return normalized_preferences


def consumer_sampler(category_preferences_df, historical_distribution):
    """
    Create Consumer instances based on user-specific preferences.

    Args:
        category_preferences_df (DataFrame): DataFrame of category preferences.

    Returns:
        list: List of Consumer instances with user-specific preferences.
    """
    # Pre-compute consumer attributes to reduce DataFrame access overhead
    consumers = []
    for consumer_id, row in tqdm(
        category_preferences_df.iterrows(), desc="Sampling consumers", unit="consumer"
    ):
        category_preferences = row.to_dict()
        sensitivity = random.uniform(0.5, 0.6)

        consumers.append(
            Consumer(
                consumer_id=consumer_id,
                sensitivity=sensitivity,
                category_preferences=category_preferences,
                historical_distribution=historical_distribution.get(consumer_id, {})
            )
        )
    return consumers
