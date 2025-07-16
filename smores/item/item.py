from collections import defaultdict
from pathlib import Path
from csv import DictReader
from json import loads
from pydantic import BaseModel, PositiveInt, NonNegativeInt
from icecream import ic

class Item(BaseModel):
    item_id: PositiveInt
    provider_id: NonNegativeInt
    features: list[float]

class ItemCollection():
    def __init__(self):
        self.items: list[Item] = []

    def add_item(self, item: Item):
        self.items.append(item)

    def add_items(self, items: list):
        self.items = self.items + items

    def size(self):
        return len(self.items)

    def __iter__(self):
        return self.items.__iter__()
    
class ItemMap():
    def __init__(self):
        self.item_map: dict[int, Item] = defaultdict(None)
        self.provider_map: dict[int, list[int]] = defaultdict(list)
        self.genre_map: dict[int, list[int]] = defaultdict(list)

    def add_item(self, item: Item):
        self.item_map[item.item_id] = item
        self.provider_map[item.provider_id].append(item.item_id)
        for i, val in enumerate(item.features):
            if val > 0:
                self.genre_map[i].append(item.item_id)

    def get_item(self, item_id: int):
        return self.item_map[item_id]
    
    def exists_item(self, item_id):
        return item_id in self.item_map
    
    def get_provider_items(self, provider_id: int):
        return self.provider_map[provider_id]
    
    def all_items(self):
        return list(self.item_map.keys())
    
    def get_genre_items(self, genre):
        return self.genre_map[genre]
    
    def load_items(self, item_data_path: Path):
        with open(item_data_path, 'r') as item_file:
            reader = DictReader(item_file)
            for row in reader:
                feature_list_str = row['features']
                feature_list = loads(feature_list_str)
                row['features'] = feature_list
                item: Item = Item.model_validate(row)
                self.add_item(item)





'''
class Item:
    def __init__(self, item_id, quality, genres, provider_id, dataset_genres):
        self.item_id = item_id
        self.quality = quality
        self.genres = genres
        self.dataset_genres = dataset_genres
        self.provider_id = provider_id
        self.weight = 0.5
        self.genre_vector = self.normalize_genres()

    def normalize_genres(self):
        genres_vector = {}

        # Assign weights to the genres based on their position
        if self.genres:
            total_genres = len(self.genres)
            decreasing_weights = [
                (total_genres - i) for i in range(total_genres)
            ]
            total_weight = sum(decreasing_weights)
            normalized_weights = [weight / total_weight for weight in decreasing_weights]

            for cat, weight in zip(self.genres, normalized_weights):
                if cat in self.dataset_genres:
                    genres_vector[cat] = weight

        return genres_vector

    def __str__(self):
        return (
            f"Item ID: {self.item_id}, Quality: {self.quality}, "
            f"genres: {self.genres}, Provider ID: {self.provider_id}, "
            f"Normalized genres Vector: {self.normalized_genres_vector}"
        )
'''



