import pickle
import numpy as np
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
    Implicit MF recommender that initializes item embeddings from a pre-trained model.
    
    Loads pre-trained item embeddings once during the first training cycle, then
    allows normal adaptation in subsequent cycles. This gives the model a head start
    on item representations while still learning user preferences from simulation data.
    
    Config params:
        pretrained_model_path: Path to pickled ImplicitMFScorer with item_embeddings
        embedding_size, epochs, regularization, positive_weight: Standard MF params
        min_user_count, min_interaction_count, min_profile_size: Training thresholds
    """
    
    def __init__(self):
        super().__init__()
        self.pretrained_model_path = None
        self.embeddings_injected = False  
        self.pretrained_scorer = None
        
    def setup(self, config):
        """Setup the recommender with config parameters"""
        super(ImplicitMFRecommender, self).setup(config)
        
        params = config.params
        
        self.embedding_size = int(params['embedding_size'])
        self.epochs = int(params.get('epochs', 5))
        self.regularization = float(params['regularization'])
        self.positive_weight = float(params['positive_weight'])
        
        self.min_user_count = int(params.get('min_user_count', 0))
        self.min_interaction_count = int(params.get('min_interaction_count', 0))
        self.min_profile_size = int(params.get('min_profile_size', 1))
        
        self.pretrained_model_path = params['pretrained_model_path']
        
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
        
        # Load pre-trained scorer at setup time
        self._load_pretrained_scorer()

    def _load_pretrained_scorer(self):
        """Load pre-trained scorer from file"""
        model_path = Path(self.pretrained_model_path)
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    self.pretrained_scorer = pickle.load(f)
                
                if isinstance(self.pretrained_scorer, ImplicitMFScorer):
                    if hasattr(self.pretrained_scorer, 'item_embeddings') and \
                       hasattr(self.pretrained_scorer, 'items'):
                        smores.Smores.state.logger.info(
                            f"Loaded pre-trained scorer: {self.pretrained_scorer.item_embeddings.shape[0]} items, "
                            f"{self.pretrained_scorer.item_embeddings.shape[1]} dimensions"
                        )
                    else:
                        smores.Smores.state.logger.warning(
                            f"Pre-trained scorer missing item_embeddings or items attribute"
                        )
                else:
                    smores.Smores.state.logger.warning(
                        f"Pre-trained model is not ImplicitMFScorer: {type(self.pretrained_scorer)}"
                    )
                    self.pretrained_scorer = None
                        
            except Exception as e:
                smores.Smores.state.logger.error(f"Error loading pre-trained scorer: {e}")
                self.pretrained_scorer = None
        else:
            smores.Smores.state.logger.warning(f"Pre-trained model not found: {model_path}")

    def build_pipeline(self):
        """Build the recommendation pipeline"""
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
    
    def _inject_pretrained_embeddings(self):
        """
        Inject pre-trained item embeddings into the current scorer.
        Called ONLY ONCE after first training.
        """
        if self.pretrained_scorer is None:
            return 0
        
        if not hasattr(self.scorer, 'items') or self.scorer.items is None:
            return 0
        
        if not hasattr(self.scorer, 'item_embeddings') or self.scorer.item_embeddings is None:
            return 0
        
        if not hasattr(self.pretrained_scorer, 'items') or self.pretrained_scorer.items is None:
            return 0
            
        if not hasattr(self.pretrained_scorer, 'item_embeddings') or self.pretrained_scorer.item_embeddings is None:
            return 0
        
        # Build lookup for pre-trained items
        pretrained_item_ids = list(self.pretrained_scorer.items.ids())
        pretrained_lookup = {int(item_id): idx for idx, item_id in enumerate(pretrained_item_ids)}
        
        # Get current scorer's items
        current_item_ids = list(self.scorer.items.ids())
        
        # Inject pre-trained embeddings
        injected_count = 0
        for current_idx, item_id in enumerate(current_item_ids):
            item_id_int = int(item_id)
            if item_id_int in pretrained_lookup:
                pretrained_idx = pretrained_lookup[item_id_int]
                self.scorer.item_embeddings[current_idx] = \
                    self.pretrained_scorer.item_embeddings[pretrained_idx]
                injected_count += 1
        
        return injected_count
    
    def train(self):
        """
        Train with warm-start.
        
        KEY DIFFERENCE: Only inject embeddings on FIRST training.
        Subsequent cycles train normally, allowing adaptation.
        """
        # Check data thresholds
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
        
        # Train the pipeline
        if self.get_dataset().interaction_count > 0:
            self.pipeline.train(self.get_dataset())
        
        self.trained = True
        
        # ONLY inject embeddings on FIRST training cycle
        if not self.embeddings_injected:
            injected = self._inject_pretrained_embeddings()
            total_items = len(self.scorer.items.ids()) if hasattr(self.scorer, 'items') and self.scorer.items else 0
            
            smores.Smores.state.logger.info(
                f"Warm-start {self.name}: ONE-TIME injection of {injected}/{total_items} pre-trained item embeddings"
            )
            self.embeddings_injected = True
        else:
            smores.Smores.state.logger.info(
                f"Warm-start {self.name}: Normal training (embeddings already injected, now adapting)"
            )
        
        # Train fallbacks
        cold_start_rec = self.get_cold_start_fallback()
        cold_user_rec = self.get_cold_user_fallback()
        if cold_start_rec is not None:
            cold_start_rec.train()
        if cold_user_rec is not None:
            cold_user_rec.train()

    def isDatasetViable(self):
        if not self.trained:
            return False
        user_count = self.dataset_active_users()
        interaction_count = self.get_dataset().interaction_count
        return user_count >= self.min_user_count and interaction_count >= self.min_interaction_count
        
    def isProfileViable(self, user_id: ID):
        items = self.get_dataset().user_row(user_id)
        if items is None:
            return False
        return items.ids().size >= self.min_profile_size


# Register the recommender
RecommenderFactory.register('warmstart_implicit_mf', WarmStartImplicitMFRecommender)