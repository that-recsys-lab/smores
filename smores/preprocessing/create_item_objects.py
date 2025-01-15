import pandas as pd
from smores.stakeholders.stakeholders import Item, Provider, Consumer, Recommender


def create_item_objects_from_csv(items_df, provider_id):
    """
    Create item objects from a grouped DataFrame.
    """
    items = []
    for idx, row in items_df.iterrows():
        item = Item(
            row["movieId"], row["rating"], set(row["genres"].split("|")), provider_id
        )
        items.append(item)
    return items
