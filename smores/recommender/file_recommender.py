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
        old_len = len(self.items)
        self.items = self.check_items()
        smores.Smores.state.logger.debug(f"Check items removed {old_len - len(self.items)} items.")
    
    def load_items(self):
        """Load item from a CSV file"""
#        try:
        with open(self.file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'item_id' in row:
                    self.items[int(row['item_id'])] = float(row['probability'])
                        
            smores.Smores.state.logger.info(f"Loaded {len(self.items)} items")
# This error should be fatal
 #       except IOError as e:
 #           smores.Smores.state.logger.error(f"Error loading file {self.file_path}")

    def check_items(self):
        filtered_items = {item: value for item,value in self.items \
                      if smores.Smores.state.items.exists_item(item)}
        return filtered_items

    def train(self):
        pass
    
    def isDatasetViable(self):
        return len(self.items) > 0
    
    def isProfileViable(self, user_id):
        return True
    
    def get_recommendations(self, user_id) -> ItemList:
        prior_interactions = self.get_user(user_id)
        if prior_interactions is not None and len(prior_interactions) > 0:
            usable_items = {item: value for item,value in self.items if item not in prior_interactions}
        else:
            usable_items = self.items

        slate_size = smores.Smores.state.slate_size
        if len(usable_items) < slate_size:
            slate_size = len(usable_items)
        
        items = smores.Smores.state.rand.choice(usable_items.keys(), slate_size, p=usable_items.values())
        
        scores = [1.0] * slate_size
        ranks = list(range(1, slate_size + 1))
        
        return ItemList(None, item_ids=items, scores=scores, rank=ranks)

RecommenderFactory.register('file_based', FileBasedRecommender)