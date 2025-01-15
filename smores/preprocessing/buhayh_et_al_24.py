import pandas as pd
import numpy as np
import random

from smores.stakeholders.stakeholders import Consumer, Provider, Item

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
        items_df (DataFrame): DataFrame containing item details.
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


def buhayh_et_al_24(
    movies_with_ratings_df,
    niche_genre="Western",
    sample_size=600,
    niche_percentage=0.1,
):
    """
    Execute the workflow to prepare recommendation data, including scaling ratings, generating
    user preferences, and creating consumer lists and niche items.

    Args:
        items_df (DataFrame): DataFrame with item details.
        movies_with_ratings_df (DataFrame): DataFrame with movie ratings.
        niche_genre (str): Niche genre to prioritize.
        sample_size (int): Number of users to sample.
        niche_percentage (float): Percentage of users interested in niche genres.

    Returns:
        tuple: Updated consumers list, consumer ID sets, and niche items.
    """
    items_df = (
        movies_with_ratings_df[["movieId", "rating", "genres"]]
        .groupby(["movieId", "genres"])
        .mean()
        .reset_index()
    )
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


def create_docuemnts_objects_from_csv(items_df, provider_id):
    items = []
    for idx, row in items_df.iterrows():
        item = Item(
            row["movieId"], row["rating"], set(row["genres"].split("|")), provider_id
        )
        items.append(item)
    return items


def provider_sampler(
    items_df,
    num_providers,
    num_docs,
    starting_id=1,
    doc_mean_std=(500, 100),
    category_sampling=False,
):
    """
    Create multiple Provider instances with sampled items and randomly generated attributes.

    Args:
        items_df (DataFrame): DataFrame containing item data.
        num_providers (int): Number of providers to generate.
        num_items_mean_std (tuple): Mean and standard deviation of the normal distribution for the number of items.

    Returns:
        list: List of Provider instances with sampled attributes.
    """
    # Initialize a list to store the sampled Provider instances
    sampled_providers = []

    # Generate multiple Provider instances
    for provider_id in range(starting_id, num_providers + starting_id):
        if num_docs <= 0:
            num_docs = int(np.random.normal(doc_mean_std[0], doc_mean_std[1]))

        # Sample items for the current provider
        sampled_items = items_df.sample(n=num_docs)
        # Create objects of movies
        sampled_items = create_docuemnts_objects_from_csv(sampled_items, provider_id)

        # Create a Provider instance with sampled items and generated attributes
        provider = Provider(
            provider_id=provider_id,
            items=sampled_items,
        )

        # Append the generated Provider instance to the list
        sampled_providers.append(provider)

    return sampled_providers


def consumer_sampler(
    category_preferences_df,
    num_consumers=None,
    n_trials=0,
    prohibited_categories=set(),
    favorite_categories=set(),
    excluded_list=[],
):
    """
    Create multiple Consumer instances based on user-specific preferences from the merged DataFrame.

    Args:
        category_preferences_df (dataframe): A dataframe of category preferences
        num_consumers (int): Number of Consumer instances to generate.
        n_trials (int): Maximum number of sampling trials.
        random_state (int): Random state for reproducibility.
        prohibited_categories (set): Set of prohibited categories.
        excluded_list (list): List of already excluded consumers.

    Returns:
        list: List of Consumer instances with user-specific preferences.
    """
    sampled_consumers = []

    if num_consumers is None:
        num_consumers = len(category_preferences_df)

    for consumer_id in category_preferences_df.index[:num_consumers]:
        category_preferences = category_preferences_df.loc[consumer_id].to_dict()
        sensitivity = np.random.uniform(low=0.5, high=0.6)
        starting_nqe = 0.5

        consumer = Consumer(
            consumer_id=consumer_id,
            sensitivity=sensitivity,
            category_preferences=category_preferences,
            prohibited_categories=prohibited_categories,
            favorite_categories=favorite_categories,
        )
        sampled_consumers.append(consumer)

    return sampled_consumers


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
