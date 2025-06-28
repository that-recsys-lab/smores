from lenskit.data.items import ItemList
import csv
import random
import smores
from smores.recommender import Recommender, RecommenderFactory

class FileBasedRecommender(Recommender):
    """Recommender that load items from a file"""
    
    def __init__(self):
        super().__init__()
        self.items = []
    
    def setup(self, config):
        super().setup(config)
        
        file_path = config.params.get('file_path', '')
        
        self.load_items(file_path)
    
    def load_items(self, file_path):
        """Load item from a CSV file"""
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'item_id' in row:
                        self.items.append(int(row['item_id']))
                        
            smores.Smores.state.logger.info(f"Loaded {len(self.items)} items")
        except:
            smores.Smores.state.logger.error(f"Error loading file {file_path}")
    
    def train(self):
        pass
    
    def isDatasetViable(self):
        return len(self.items) > 0
    
    def isProfileViable(self, user_id):
        return True
    
    def get_recommendations(self, user_id) -> ItemList:
        slate_size = smores.Smores.state.slate_size
        if len(self.items) < slate_size:
            slate_size = len(self.items)
        
        items = random.sample(self.items, slate_size)
        
        scores = [1.0] * slate_size
        ranks = list(range(1, slate_size + 1))
        
        return ItemList(None, item_ids=items, scores=scores, rank=ranks)

RecommenderFactory.register('file_based', FileBasedRecommender)