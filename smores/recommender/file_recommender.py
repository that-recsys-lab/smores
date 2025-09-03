import csv
from collections import defaultdict
from lenskit.data.items import ItemList

import smores
from smores.recommender import Recommender, RecommenderFactory


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
        self.usable_items = self.items.copy()
        
        # Initialize cache variables for _scale_popularity
        self._cached_probabilities = None
        self._cache_dirty = True
    
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
        prior_interactions = self.get_user(user_id)
        if prior_interactions is not None:
            for item in prior_interactions.ids():
                self.usable_items[item] = 0 # set probability to 0
            self._cache_dirty = True  # Invalidate cache when items change

        slate_size = smores.Smores.state.slate_size
        if len(self.usable_items) < slate_size:
            slate_size = len(self.usable_items)
        
        probabilities = self._scale_popularity()
        items = smores.Smores.state.rand.choice(list(self.usable_items.keys()), slate_size, p=probabilities)
        
        scores = [1.0] * slate_size
        ranks = list(range(1, slate_size + 1))

        # reset self.usable items to match self.items
        if prior_interactions is not None:
            for item in prior_interactions.ids():
                self.usable_items[item] = self.items[item]
            self._cache_dirty = True  # Invalidate cache when items are reset
        
        return ItemList(None, item_ids=items, scores=scores, rank=ranks)
    
    def _scale_popularity(self):
        if self._cached_probabilities is None or self._cache_dirty:
            popularities = list(self.usable_items.values())
            popularity_sum = sum(popularities)
            self._cached_probabilities = [x/popularity_sum for x in popularities]
            self._cache_dirty = False
        return self._cached_probabilities


RecommenderFactory.register('file_based', FileBasedRecommender)