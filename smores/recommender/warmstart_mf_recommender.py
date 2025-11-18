import pickle
from pathlib import Path
from sys import maxsize

from lenskit.als import ImplicitMFConfig, ImplicitMFScorer
from lenskit.pipeline import PipelineBuilder
from lenskit.basic import UserTrainingHistoryLookup, TopNRanker
from lenskit.basic.candidates import UnratedTrainingItemsCandidateSelector
from lenskit.data import ID

from .lk_recommenders import ImplicitMFRecommender
from .recommender import RecommenderFactory
import smores


class WarmStartImplicitMFRecommender(ImplicitMFRecommender):
    """
    Matrix Factorization recommender that loads pre-trained embeddings from a file.
    This allows the recommender to start with knowledge from a large training dataset
    instead of training from scratch during the simulation.
    """
    
    def __init__(self):
        super().__init__()
        self.pretrained_model_path = None
        
    def setup(self, config):
        """Setup the recommender with config parameters"""
        super(ImplicitMFRecommender, self).setup(config)
        
        params = config.params
        
        self.embedding_size = int(params['embedding_size'])
        self.epochs = int(params.get('epochs', 1))
        self.regularization = float(params['regularization'])
        self.positive_weight = float(params['positive_weight'])
        
        # Thresholds
        self.min_user_count = int(params.get('min_user_count', 0))
        self.min_interaction_count = int(params.get('min_interaction_count', 0))
        self.min_profile_size = int(params.get('min_profile_size', 1))
        
        # Path to pre-trained model file
        self.pretrained_model_path = params['pretrained_model_path']
        
        # Create LensKit config
        self.lk_config = ImplicitMFConfig(
            embedding_size=self.embedding_size,
            epochs=self.epochs,
            regularization=self.regularization,
            weight=self.positive_weight,
            user_embeddings=True,
            use_ratings=False
        )
        
        self.scorer = ImplicitMFScorer(self.lk_config)
        self.pipeline = self.build_pipeline()

    def build_pipeline(self):
        """Build the recommendation pipeline - same as parent class"""
        scorer = self.get_scorer()
        slate_size = smores.Smores.state.slate_size

        pipe = PipelineBuilder()
        query = pipe.create_input('query', ID)
        history = pipe.add_component('history-lookup', UserTrainingHistoryLookup, query=query)
        default_candidates = pipe.add_component('candidate-selector',
            UnratedTrainingItemsCandidateSelector, query=history)
        score = pipe.add_component('scorer', scorer, query=query, items=default_candidates)
        recommend = pipe.add_component('ranker', TopNRanker, {'n': slate_size}, items=score)
        pipe.alias('recommender', recommend)
        pipe.default_component('recommender')
        return pipe.build()
    
    def train(self):
        """Load pre-trained model instead of training from scratch"""
        # Check if we have enough data to train
        if self.get_dataset().interaction_count < self.min_interaction_count or \
                self.get_dataset().user_count < self.min_user_count:
            smores.Smores.state.logger.debug(
                f"Insufficient data for {self.name}. "
                f"Users: {self.get_dataset().user_count}/{self.min_user_count}, "
                f"Interactions: {self.get_dataset().interaction_count}/{self.min_interaction_count}"
            )
            cold_start_rec = self.get_cold_start_fallback()
            cold_user_rec = self.get_cold_user_fallback()
            if cold_start_rec is not None:
                cold_start_rec.train()
            if cold_user_rec is not None:
                cold_user_rec.train()
            return
        
        model_path = Path(self.pretrained_model_path)
        
        if model_path.exists():
            smores.Smores.state.logger.info(f"Loading pre-trained model from {self.pretrained_model_path}")
            
            try:
                with open(model_path, 'rb') as f:
                    pretrained_scorer = pickle.load(f)
                
                if not isinstance(pretrained_scorer, ImplicitMFScorer):
                    raise TypeError(f"Loaded object is {type(pretrained_scorer)}, expected ImplicitMFScorer")
                
                self.scorer = pretrained_scorer
                
                self.pipeline = self.build_pipeline()
                
                self.pipeline.train(self.get_dataset())
                
                self.trained = True
                smores.Smores.state.logger.info(
                    f"Pre-trained model loaded successfully for {self.name}"
                )
                
                cold_start_rec = self.get_cold_start_fallback()
                cold_user_rec = self.get_cold_user_fallback()
                if cold_start_rec is not None:
                    cold_start_rec.train()
                if cold_user_rec is not None:
                    cold_user_rec.train()
                    
            except Exception as e:
                smores.Smores.state.logger.error(f"Error loading pre-trained model: {e}")
                smores.Smores.state.logger.info("Falling back to training from scratch")
                # Call the parent's parent train method
                super(ImplicitMFRecommender, self).train()
        else:
            smores.Smores.state.logger.warning(
                f"Pre-trained model not found at {self.pretrained_model_path}"
            )
            smores.Smores.state.logger.info(f"Training {self.name} from scratch")
            # Call the parent's parent train method
            super(ImplicitMFRecommender, self).train()

    def isDatasetViable(self):
        """Check if we have enough data to use the recommender"""
        if not self.trained:
            return False
        else:
            user_count = self.dataset_active_users()
            interaction_count = self.get_dataset().interaction_count
            if user_count >= self.min_user_count and interaction_count >= self.min_interaction_count:
                return True
            else:
                return False
        
    def isProfileViable(self, user_id: ID):
        """Check if user has enough interactions in their profile"""
        items = self.get_dataset().user_row(user_id)
        if items is None:
            return False
        else:
            profile_size = items.ids().size
            if profile_size < self.min_profile_size:
                return False
            else:
                return True


# Register the recommender with the factory
RecommenderFactory.register('warmstart_implicit_mf', WarmStartImplicitMFRecommender)