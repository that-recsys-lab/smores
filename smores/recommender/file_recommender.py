import csv
from collections import defaultdict
from lenskit.data.items import ItemList

import smores
from smores.recommender import Recommender, RecommenderFactory
from smores.samplers.rejection_sampler import sample_uninteracted_items


class FileBasedRecommender(Recommender):
    """Recommender that load items from a file"""
    
    def __init__(self):
        super().__init__()
        self.items = defaultdict(float)
        self.file_path = None
        # Needs no training
        self.trained = True
    
    def setup(self, config):
        super().setup(config)

        self.file_path = smores.Smores.state.data_directory / \
                config.params['file_name']
        self.load_items()

        # Pre-compute probabilities once
        self.item_ids = list(self.items.keys())
        popularities = list(self.items.values())
        popularity_sum = sum(popularities)
        self.base_probabilities = [x/popularity_sum for x in popularities]
    
    def load_items(self):
        """Load item from a CSV file"""
#        try:
        with open(self.file_path, 'r') as f:
            reader = csv.DictReader(f)
            invalid_items = 0
            for row in reader:
                if 'item_id' in row:
                    item_id = int(row['item_id'])
                    popularity = float(row['popularity'])
                    if smores.Smores.state.items.exists_item(item_id):
                        self.items[item_id] = popularity
                    else:
                        invalid_items += 1
                        
        smores.Smores.state.logger.info(f"Loaded {len(self.items)} items")
        smores.Smores.state.logger.debug(f"Items not found: {invalid_items} items.")

# This error should be fatal
 #       except IOError as e:
 #           smores.Smores.state.logger.error(f"Error loading file {self.file_path}")

    def train(self):
        pass
    
    def isDatasetViable(self):
        return len(self.items) > 0
    
    def isProfileViable(self, user_id):
        return True
    
    def get_recommendations(self, user_id) -> ItemList:
        # Get items user has already interacted with
        prior_interactions = self.get_user(user_id)
        interacted_items = set()
        if prior_interactions is not None:
            interacted_items = set(prior_interactions.ids())

        slate_size = smores.Smores.state.slate_size

        # Use rejection sampling to get uninteracted items
        items = sample_uninteracted_items(
            self.item_ids,
            self.base_probabilities,
            interacted_items,
            slate_size
        )

        actual_slate_size = len(items)
        scores = [1.0] * actual_slate_size
        ranks = list(range(1, actual_slate_size + 1))

        return ItemList(None, item_ids=items, scores=scores, rank=ranks)
    

RecommenderFactory.register('file_based', FileBasedRecommender)