import pandas as pd
from smores.stakeholders.stakeholders import Item, Provider, Consumer, Recommender


def create_item_objects_from_csv(items_df, provider_id, dataset_genres):
    """
    Create item objects from a grouped DataFrame.
    """
    items = []
    for idx, row in items_df.iterrows():
        try:
            item = Item(
            row["itemId"], row["rating"], set(row["genres"].split("|")), provider_id, dataset_genres
            )
            items.append(item)
        except KeyError as e:
            print(f"Missing key in row {idx}: {e}")
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
    return items
