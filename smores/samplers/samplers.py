import pandas as pd
import numpy as np
from smores.stakeholders.stakeholders import Consumer, Provider, Document


def create_docuemnts_objects_from_csv(items_df, provider_id):
    items = []
    for idx, row in items_df.iterrows():
        document = Document(
            row["movieId"], row["rating"], set(row["genres"].split("|")), provider_id
        )
        items.append(document)
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
        items_df (DataFrame): DataFrame containing document data.
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
