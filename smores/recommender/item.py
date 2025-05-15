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

class ItemList():
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

