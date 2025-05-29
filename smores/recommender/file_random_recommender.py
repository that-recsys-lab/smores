from smores.recommender.recommender import Recommender, RecommenderFactory
from lenskit.training import Trainable
from lenskit.pipeline import Component
from lenskit.data import ItemList
import random
import pandas as pd
from pathlib import Path
import os
import smores

class FileRandomScorer:
    """A simple scorer that assigns random scores to items loaded from a CSV file"""
    
    def __init__(self, data_directory, item_file):
        self.data_directory = data_directory
        self.item_file = item_file
        self.items = []
    
    def load_items(self):
        """Load item IDs from a CSV file"""
        file_path = os.path.join(self.data_directory, self.item_file)
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Item file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        id_column = 'itemId' if 'itemId' in df.columns else df.columns[0]
        
        self.items = df[id_column].astype(str).tolist()
        print(f"Loaded {len(self.items)} items from {file_path}")
    
    def train(self, dataset):
        """Required method for LensKit components"""
        self.load_items()
        return self

    def __call__(self, _, items: ItemList) -> ItemList:
        """Assign random scores to items"""
        scoreable_items = [i for i in items if str(i) in self.items]
        return {i: random.random() for i in scoreable_items}



class FileRandomRecommender(Recommender):
    """Recommender that randomly samples items from a CSV file"""
    
    def __init__(self):
        super().__init__()
        self.data_directory = None
        self.item_file = None
        self.scorer = None
    
    def setup(self, config):
        """Set up the recommender with data directory and item file"""
        global_config = smores.Smores.state.config
        
        self.data_directory = getattr(config, 'data_directory', global_config.data.directory)
        self.item_file = getattr(config, 'item_file', global_config.data.item_file)
        
        self.scorer = FileRandomScorer(self.data_directory, self.item_file)
        super().setup(config)
    
    def get_scorer(self):
        """Return the scorer component for the pipeline"""
        return self.scorer
    
    def isDatasetViable(self):
        """This recommender doesn't depend on the dataset"""
        return True
    
    def isProfileViable(self, user_id):
        """This recommender doesn't need user profiles"""
        return True


RecommenderFactory.register('file_random', FileRandomRecommender)