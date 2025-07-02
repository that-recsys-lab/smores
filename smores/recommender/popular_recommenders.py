from sys import maxsize
from icecream import ic

from lenskit.pipeline import topn_pipeline
from lenskit.basic.popularity import PopScorer, PopConfig
from lenskit.data import ID, ItemList
from lenskit import recommend

from .recommender import LKRecommender, RecommenderFactory

class PopularFromFileRecommender(Recommender):
    """
    A simplified version of the PopularRecommender specifically designed for cold start scenarios.
    This recommender has no minimum requirements for users or interactions and loads the popular items
    from a file.
    """
    def __init__(self):
        super().__init__()
        self.name = "Popular From File"

    def setup(self, config):
        # Override parent setup to avoid parameter requirements
        super().__init__()  # Call Recommender.__init__, not PopularRecommender.setup
        
        # Initialize with lenient defaults
        self.min_user_count = 0  # Accept any number of users
        self.min_interaction_count = 0  # Accept any number of interactions
        
        # Set up PopConfig and scorer
        self.lk_config = PopConfig(score='count')
        self.scorer = PopScorer(self.lk_config)
        
        # Call Recommender's setup to initialize dataset
        Recommender.setup(self, config)
        
        # Build the pipeline
        self.pipeline = self.build_pipeline()

    def isDatasetViable(self):
        # Always return True - this is a fallback
        return True

    def isProfileViable(self, user_id):
        # Always return True - this is a fallback
        return True
        
    def get_recommendations(self, user_id: ID) -> ItemList:
        try:
            # Try to use the normal pipeline
            return recommend(self.pipeline, user_id, n=smores.Smores.state.slate_size)
        except Exception as e:
            # If anything fails, fall back to a simple approach
            ic(f"Fallback recommender encountered error: {str(e)}")
            
            # Get all items
            all_items = list(smores.Smores.state.items.all_items())
            slate_size = smores.Smores.state.slate_size
            
            # If no items, return empty list
            if not all_items:
                return ItemList(None, item_ids=[], scores=[], rank=[])
            
            # Take first slate_size items
            recs = all_items[:slate_size] if len(all_items) >= slate_size else all_items
            scores = [5.0] * len(recs)
            ranks = list(range(1, len(recs)+1))
            
            return ItemList(None, item_ids=recs, scores=scores, rank=ranks)
        
class PopularRecommender(LKRecommender):
    def __init__(self):
        super().__init__()
        self.min_user_count: int = maxsize
        self.min_interaction_count: int = maxsize

    def setup(self, config):
        params = config.params
        self.min_user_count = int(params['min_user_count'])
        self.min_interaction_count = int(params['min_interaction_count'])
        self.lk_config = PopConfig(score='count')
        self.scorer = PopScorer(self.lk_config)
        super().setup(config)

    def train(self):
        if self.dataset.interaction_count >= self.min_interaction_count and \
                self.dataset.user_count >= self.min_user_count:
            super().train()

    def build_pipeline(self):
        scorer = self.get_scorer()
        slate_size = smores.Smores.state.slate_size
        return topn_pipeline(scorer, n=slate_size)

    def isDatasetViable(self):
        user_count = self.dataset.user_count
        interaction_count = self.dataset.interaction_count
        if user_count >= self.min_user_count and interaction_count >= self.min_interaction_count:
            return True
        else:
            return False
        
    def isProfileViable(self, _):
        return True # Because we ignore the profile, everyone is viable
        
RecommenderFactory.register('popular_file', PopularFromFileRecommender)
RecommenderFactory.register('popular', PopularRecommender)