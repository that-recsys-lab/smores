import pandas as pd
import numpy as np
import random

from smores.stakeholders.stakeholders import Consumer, Provider, Document
from smores.samplers.samplers import provider_sampler, consumer_sampler

pd.options.mode.chained_assignment = None  # default='warn'


# Function to calculate Bayesian average per user
def bayesian_average_per_user(rating, user_avg, reg_factor):
    bayesian_avg = (rating + reg_factor * user_avg) / (1 + reg_factor)
    return bayesian_avg


def normalize(vector):
    total = sum(vector.values())
    if total == 0:
        return {key: 0 for key in vector}  # Handle case where total is zero
    return {key: value / total for key, value in vector.items()}


def generate_genre_preferences(movies_with_ratings_df):
    # User Genre preferences using bayesian average

    # Step 1: Explode genres
    df_exploded = movies_with_ratings_df.copy()
    df_exploded["genres"] = df_exploded["genres"].str.split("|")
    df_exploded = df_exploded.explode("genres")

    # Step 2: Calculate Bayesian average for each genre per user
    grouped = df_exploded.groupby(["userId", "genres"])["rating"]
    genre_avg = grouped.mean().reset_index()

    # Calculate overall average rating per user
    user_avg_ratings = df_exploded.groupby("userId")["rating"].mean()

    # Regularization factor per user (you can adjust this based on your dataset characteristics)
    reg_factor = 3

    # Apply Bayesian average calculation per user
    genre_avg["bayesian_avg_rating"] = genre_avg.apply(
        lambda row: bayesian_average_per_user(
            row["rating"], user_avg_ratings[row["userId"]], reg_factor
        ),
        axis=1,
    )

    genre_avg["rating"] = genre_avg["bayesian_avg_rating"].apply(lambda x: (x - 1) / 4)

    # Step 3: Pivot the table to have genres as columns
    user_genre_preference = genre_avg.pivot(
        index="userId", columns="genres", values="rating"
    ).fillna(0)
    user_genre_preference = user_genre_preference.drop(
        "(no genres listed)", axis=1, errors="ignore"
    )  # Handle cases where column may not exist

    # Normalize genre preferences for each user
    normalized_user_genre_preference = user_genre_preference.apply(
        lambda row: normalize(row.to_dict()), axis=1
    )
    normalized_user_genre_preference = pd.DataFrame(
        normalized_user_genre_preference.tolist(), index=user_genre_preference.index
    )

    return normalized_user_genre_preference


def prepare_user_preferences(
    movies_with_ratings_df, sample_size=600, niche_genre="Western", niche_percentage=0.1
):
    """
    Generate, sample, and update user genre preferences for niche and mainstream interest groups.

    Args:
        movies_with_ratings_df (DataFrame): DataFrame with movie ratings.
        sample_size (int): Number of users to sample.
        niche_genre (str): Niche genre to prioritize.
        niche_percentage (float): Percentage of users interested in niche genres.

    Returns:
        tuple: DataFrames for niche and mainstream interest users, and the set of other genres.
    """
    # Generate and sample user preferences
    user_genre_preference = generate_genre_preferences(movies_with_ratings_df)
    sampled_users = (
        user_genre_preference.sample(sample_size).sample(frac=1).reset_index(drop=True)
    )

    # Split users into niche and mainstream interest groups
    ten_percent = int(len(sampled_users) * niche_percentage)
    users_niche_movies = sampled_users[:ten_percent]
    users_mainstream_interest = sampled_users[ten_percent:]

    # Update genre preferences
    other_genres = list(users_niche_movies.columns)
    other_genres.remove(niche_genre)

    users_niche_movies[niche_genre] = np.where(
        users_niche_movies[niche_genre] > 0, users_niche_movies[niche_genre] * 4, 0.2
    )
    users_niche_movies[other_genres] /= 4
    users_mainstream_interest[niche_genre] /= 4

    return users_niche_movies, users_mainstream_interest, other_genres


def create_consumers_and_items(
    users_niche_movies, users_mainstream_interest, items_df, niche_genre
):
    """
    Create consumer lists and retrieve niche items based on genre preferences.

    Args:
        users_niche_movies (DataFrame): Users interested in niche genres.
        users_mainstream_interest (DataFrame): Users with mainstream interest.
        items_df (DataFrame): DataFrame containing document details.
        niche_genre (str): Niche genre to prioritize.

    Returns:
        tuple: Consumer lists, consumer ID sets, and niche items DataFrame.
    """
    # Create consumer lists
    niche_consumers_list = consumer_sampler(
        category_preferences_df=users_niche_movies,
        favorite_categories=set([niche_genre]),
    )
    mainstream_consumers_list = consumer_sampler(
        category_preferences_df=users_mainstream_interest
    )
    consumers_list = niche_consumers_list + mainstream_consumers_list
    random.shuffle(consumers_list)

    # Create consumer ID sets
    niche_consumers_set = set(
        [consumer.consumer_id for consumer in niche_consumers_list]
    )
    mainstream_consumers_set = set(
        [consumer.consumer_id for consumer in mainstream_consumers_list]
    )

    # Retrieve niche items
    niche_items = items_df[items_df["genres"].str.contains(niche_genre)]

    return (
        consumers_list,
        niche_consumers_set,
        mainstream_consumers_set,
        niche_items,
    )


def process_recommendation_data(
    items_df,
    movies_with_ratings_df,
    niche_genre="Western",
    sample_size=600,
    niche_percentage=0.1,
):
    """
    Execute the workflow to prepare recommendation data, including scaling ratings, generating
    user preferences, and creating consumer lists and niche items.

    Args:
        items_df (DataFrame): DataFrame with document details.
        movies_with_ratings_df (DataFrame): DataFrame with movie ratings.
        niche_genre (str): Niche genre to prioritize.
        sample_size (int): Number of users to sample.
        niche_percentage (float): Percentage of users interested in niche genres.

    Returns:
        tuple: Updated consumers list, consumer ID sets, and niche items.
    """
    # Scale ratings to [0, 1]
    items_df["rating"] = items_df["rating"].apply(lambda x: x / 5)

    # Generate and update user preferences
    users_niche_movies, users_mainstream_interest, _ = prepare_user_preferences(
        movies_with_ratings_df, sample_size, niche_genre, niche_percentage
    )

    # Create consumers and retrieve niche items
    return create_consumers_and_items(
        users_niche_movies, users_mainstream_interest, items_df, niche_genre
    )
