import pandas as pd
import numpy as np
import random
from smores.stakeholders.stakeholders import Consumer


def genre_preferences(
    consumer_item_rating_genre_df, niche_genre="Western", threshold=0.01
):
    """
    Generate consumer lists and retrieve niche items based on genre preferences.

    Args:
        consumer_item_rating_genre_df (DataFrame): DataFrame containing user-item ratings and genres.
        niche_genre (str): Niche genre to prioritize.
        threshold (float): Threshold for considering a user as interested in the niche genre.

    Returns:
        tuple: Consumer lists, niche consumers set, mainstream consumers set, and niche items DataFrame.
    """
    # Step 1: Generate normalized genre preferences
    normalized_category_preferences_df = generate_genre_preferences(
        consumer_item_rating_genre_df
    )

    # Step 2: Split users into niche and mainstream interest groups based on the threshold
    users_niche_movies = normalized_category_preferences_df[
        normalized_category_preferences_df[niche_genre] > threshold
    ]
    users_mainstream_interest = normalized_category_preferences_df.drop(
        users_niche_movies.index
    )

    # Step 3: Create consumer lists
    niche_consumers_list = consumer_sampler(
        category_preferences_df=users_niche_movies,
        favorite_categories=set([niche_genre]),
    )
    mainstream_consumers_list = consumer_sampler(
        category_preferences_df=users_mainstream_interest
    )
    consumers_list = niche_consumers_list + mainstream_consumers_list
    random.shuffle(consumers_list)

    # Step 4: Create consumer ID sets
    niche_consumers_set = set(consumer.consumer_id for consumer in niche_consumers_list)
    mainstream_consumers_set = set(
        consumer.consumer_id for consumer in mainstream_consumers_list
    )

    # Step 5: Retrieve niche items
    niche_items = consumer_item_rating_genre_df[
        consumer_item_rating_genre_df["genres"].str.contains(niche_genre)
    ]

    return consumers_list, niche_consumers_set, mainstream_consumers_set, niche_items


def generate_genre_preferences(consumer_item_rating_genre_df):
    """
    Generate a normalized DataFrame of user genre preferences based on ratings and frequency.

    Args:
        consumer_item_rating_genre_df (DataFrame): DataFrame containing user-item ratings and genres.

    Returns:
        DataFrame: Normalized preferences per user and genre.
    """
    df_exploded = explode_genres(consumer_item_rating_genre_df)
    user_genre_preference = calculate_bayesian_average(df_exploded)
    normalized_category_preferences_df = normalize_preferences(
        user_genre_preference, df_exploded
    )

    return normalized_category_preferences_df


def explode_genres(consumer_item_rating_genre_df):
    """
    Explodes the genres column into multiple rows for each user-item interaction.

    Args:
        consumer_item_rating_genre_df (DataFrame): DataFrame containing user-item ratings and genres.

    Returns:
        DataFrame: Expanded DataFrame with one row per genre.
    """
    df_exploded = consumer_item_rating_genre_df.copy()
    df_exploded["genres"] = df_exploded["genres"].str.split("|")
    return df_exploded.explode("genres")


def calculate_bayesian_average(df_exploded, reg_factor=3):
    """
    Calculate the Bayesian average rating for each genre per user.

    Args:
        df_exploded (DataFrame): Exploded DataFrame with user-item-genre interactions.
        reg_factor (int): Regularization factor for Bayesian average calculation.

    Returns:
        DataFrame: Pivot table with Bayesian average ratings per user and genre.
    """
    grouped = df_exploded.groupby(["userId", "genres"])["rating"]
    genre_avg = grouped.mean().reset_index()
    user_avg_ratings = df_exploded.groupby("userId")["rating"].mean()

    # Incorporate frequency into Bayesian average
    genre_avg["frequency"] = grouped.size().values
    genre_avg["bayesian_avg_rating"] = genre_avg.apply(
        lambda row: (
            (
                row["rating"] * row["frequency"]
                + reg_factor * user_avg_ratings[row["userId"]]
            )
            / (row["frequency"] + reg_factor)
        ),
        axis=1,
    )

    return genre_avg.pivot(
        index="userId", columns="genres", values="bayesian_avg_rating"
    ).fillna(0)


def normalize_preferences(user_genre_preference, df_exploded):
    """
    Normalize genre preferences considering both ratings and frequency.

    Args:
        user_genre_preference (DataFrame): Bayesian average ratings per user and genre.
        df_exploded (DataFrame): Exploded DataFrame with user-item-genre interactions.

    Returns:
        DataFrame: Normalized preferences per user and genre.
    """
    genre_frequency = (
        df_exploded.groupby(["userId", "genres"]).size().unstack(fill_value=0)
    )
    combined_preferences = user_genre_preference * genre_frequency
    normalized = combined_preferences.apply(
        lambda row: {
            key: value / sum(row) if sum(row) != 0 else 0
            for key, value in row.to_dict().items()
        },
        axis=1,
    )
    return pd.DataFrame(normalized.tolist(), index=combined_preferences.index)


def consumer_sampler(
    category_preferences_df, prohibited_categories=set(), favorite_categories=set()
):
    """
    Create multiple Consumer instances based on user-specific preferences from the given DataFrame.

    Args:
        category_preferences_df (DataFrame): DataFrame of category preferences.
        prohibited_categories (set): Set of prohibited categories.
        favorite_categories (set): Set of favorite categories.

    Returns:
        list: List of Consumer instances with user-specific preferences.
    """
    sampled_consumers = []

    for consumer_id in category_preferences_df.index:
        category_preferences = category_preferences_df.loc[consumer_id].to_dict()
        sensitivity = np.random.uniform(low=0.5, high=0.6)

        consumer = Consumer(
            consumer_id=consumer_id,
            sensitivity=sensitivity,
            category_preferences=category_preferences,
            prohibited_categories=prohibited_categories,
            favorite_categories=favorite_categories,
        )
        sampled_consumers.append(consumer)

    return sampled_consumers
