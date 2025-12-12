from pydantic import BaseModel
from sys import maxsize
from icecream import ic
from abc import abstractmethod
import numpy as np

import smores
from lenskit.pipeline import Pipeline, PipelineBuilder, Component, topn_pipeline
from lenskit.basic.candidates import UnratedTrainingItemsCandidateSelector
from lenskit.basic import UserTrainingHistoryLookup, TopNRanker
from lenskit.basic.popularity import PopScorer, PopConfig
from lenskit.knn import ItemKNNConfig, ItemKNNScorer
from lenskit.als import ImplicitMFConfig, ImplicitMFScorer
from lenskit.funksvd import FunkSVDConfig, FunkSVDScorer
from lenskit.data import ID
from lenskit import recommend

from .recommender import Recommender, RecommenderFactory

import smores


class LKRecommender(Recommender):
    def __init__(self):
        super().__init__()
        self.pipeline: Pipeline = None
        self.lk_config: BaseModel
        self.scorer: Component = None

    def setup(self, config):
        super().setup(config)

    def get_scorer(self):
        return self.scorer

    def train(self):
        super().train()
        # Don't train on an empty dataset
        if self.get_dataset().interaction_count > 0:
            self.pipeline.train(self.get_dataset())
        self.trained = True
    
    @abstractmethod
    def build_pipeline(self):
        pass
    
    def _should_train(self, min_users: int, min_interactions: int) -> bool:
        dataset = self.get_dataset()
        if dataset is None:
            return False
        return dataset.interaction_count >= min_interactions and dataset.user_count >= min_users

    def _dataset_viable(self, min_users: int, min_interactions: int) -> bool:
        if not self.trained:
            return False
        user_count = self.dataset_active_users()
        interaction_count = self.get_dataset().interaction_count
        return user_count >= min_users and interaction_count >= min_interactions

    def _profile_viable(self, user_id: ID, min_profile_size: int) -> bool:
        items = self.get_dataset().user_row(user_id)
        if items is None:
            return False
        return items.ids().size >= min_profile_size

    def _build_history_pipeline(
        self,
        scorer: Component,
        candidate_selector=UnratedTrainingItemsCandidateSelector,
        selector_config=None,
        ranker_kwargs: dict | None = None,
    ):
        candidate_count = max(self._candidate_request_count(), 1)
        ranker_args = ranker_kwargs or {'n': candidate_count}

        pipe = PipelineBuilder()
        query = pipe.create_input('query', ID)
        history = pipe.add_component('history-lookup', UserTrainingHistoryLookup, query=query)
        if selector_config is not None:
            default_candidates = pipe.add_component('candidate-selector',
                candidate_selector, selector_config, query=history)
        else:
            default_candidates = pipe.add_component('candidate-selector',
                candidate_selector, query=history)
        score = pipe.add_component('scorer', scorer, query=query, items=default_candidates)
        recommend_comp = pipe.add_component('ranker', TopNRanker, ranker_args, items=score)
        pipe.alias('recommender', recommend_comp)
        pipe.default_component('recommender')
        return pipe.build()

    def get_recommendations(self, user_id: ID):
        if not self.isDatasetViable():
            cold_start_rec = smores.Smores.state.recommenders_fallback.get_recommender(self.cold_start_fallback)
            self._last_used_fallback = True
            result = cold_start_rec.get_recommendations(user_id)
            self._last_sampled_count = getattr(cold_start_rec, '_last_sampled_count', 0)
            return result
        elif not self.isProfileViable(user_id):
            cold_user_rec = smores.Smores.state.recommenders_fallback.get_recommender(self.cold_user_fallback)
            self._last_used_fallback = True
            result = cold_user_rec.get_recommendations(user_id)
            self._last_sampled_count = getattr(cold_user_rec, '_last_sampled_count', 0)
            return result
        else:
            recs = recommend(self.pipeline, user_id)
            self._last_used_fallback = False
            return self.apply_item_sampling(user_id, recs)

class PopularRecommender(LKRecommender):
    def __init__(self):
        super().__init__()
        self.min_user_count: int = maxsize
        self.min_interaction_count: int = maxsize

    def setup(self, config):
        super().setup(config)
        params = config.params
        self.min_user_count = int(params['min_user_count'])
        self.min_interaction_count = int(params['min_interaction_count'])
        self.lk_config = PopConfig(score='count')
        self.scorer = PopScorer(self.lk_config)
        self.pipeline = self.build_pipeline()

    # No minimum for training the popular recommender
    def _select_core_items(
        self,
        user_id: int,
        ids: list[int],
        scores: list[float] | None,
        ranks: list[int] | None,
        desired_count: int,
    ) -> tuple[list[int], list[float] | None, list[int] | None]:
        if desired_count <= 0 or not ids:
            return super()._select_core_items(user_id, ids, scores, ranks, desired_count)

        prior = self.get_user(user_id)
        history = set(int(i) for i in prior.ids()) if prior is not None else set()
        blocked = self._current_blocked_items(user_id)

        candidates: list[int] = []
        weights: list[float] = []
        index_lookup: list[int] = []
        for idx, item_id in enumerate(ids):
            item_int = int(item_id)
            if item_int in history or item_int in blocked:
                continue
            candidates.append(item_int)
            index_lookup.append(idx)
            if scores is not None:
                weights.append(max(float(scores[idx]), 0.0))
            else:
                weights.append(1.0)

        if not candidates:
            return super()._select_core_items(user_id, ids, scores, ranks, desired_count)

        limit = min(desired_count, len(candidates))
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.sum() > 0:
            probs = weight_array / weight_array.sum()
            chosen_pos = smores.Smores.state.rand.choice(len(candidates), size=limit, replace=False, p=probs)
        else:
            chosen_pos = smores.Smores.state.rand.choice(len(candidates), size=limit, replace=False)
        chosen_indices = [index_lookup[pos] for pos in chosen_pos]

        selected_ids = [ids[idx] for idx in chosen_indices]
        selected_scores = None
        if scores is not None:
            selected_scores = [scores[idx] for idx in chosen_indices]
        selected_ranks = None
        if ranks is not None:
            selected_ranks = [ranks[idx] for idx in chosen_indices]

        return selected_ids, selected_scores, selected_ranks

    def train(self):
        super().train()
    
    def build_pipeline(self):
        scorer = self.get_scorer()
        candidate_count = max(self._candidate_request_count(), 1)
        return topn_pipeline(scorer, n=candidate_count)

    # def build_pipeline(self):
    #     scorer = self.get_scorer()
    #     slate_size = smores.Smores.state.slate_size

    #     pipe = PipelineBuilder()
    #     # define an input parameter for the user ID (the 'query')
    #     query = pipe.create_input('query', ID)
    #     # find candidates from the training data. Because some users do not have data yet
    #     # (cold user case) we can't use the UnratedTrainingItemsVersion. This may cause some problems
    #     # if we can't build up enough of a history for a user.
    #     default_candidates = pipe.add_component('candidate-selector',
    #         AllTrainingItemsCandidateSelector)
    #     # score the candidate items using the specified scorer
    #     score = pipe.add_component('scorer', scorer, query=query, items=default_candidates)
    #     # rank the items by score
    #     recommend = pipe.add_component('ranker', TopNRanker, {'n': slate_size}, items=score)
    #     pipe.alias('recommender', recommend)
    #     pipe.default_component('recommender')
    #     return pipe.build()

    def isDatasetViable(self):
        # If the model hasn't been trained, it can't be used
        if not self.trained:
            return False
        else:
            user_count = self.dataset_active_users()
            interaction_count = self.get_dataset().interaction_count
            if user_count >= self.min_user_count and interaction_count >= self.min_interaction_count:
                return True
            else:
                return False

        
    def isProfileViable(self, _):
        return True # Because we ignore the profile, everyone is viable

class ItemKnnRecommender(LKRecommender):
    def __init__(self):
        super().__init__()
        self.min_user_count = maxsize
        self.min_interaction_count = maxsize
        self.min_profile_size = maxsize

    def setup(self, config):
        super().setup(config)
        # get data from config
        params = config.params
        max_nbrs = int(params['max_neighbors'])
        min_nbrs = int(params['min_neighbors'])
        min_sim = float(params['min_similarity'])
        self.min_user_count = int(params['min_user_count'])
        self.min_interaction_count = int(params['min_interaction_count'])
        self.min_profile_size = int(params['min_profile_size'])
        # create ItemKNNConfig object
        self.lk_config = ItemKNNConfig(max_nbrs=max_nbrs, min_nbrs=min_nbrs, 
                                       min_sim=min_sim, feedback='implicit')
        # create ItemKNNScorer
        self.scorer = ItemKNNScorer(self.lk_config)
        self.pipeline = self.build_pipeline()

    def build_pipeline(self):
        return self._build_history_pipeline(self.get_scorer())

    def train(self):
        if self._should_train(self.min_user_count, self.min_interaction_count):
            super().train()

    def isDatasetViable(self):
        return self._dataset_viable(self.min_user_count, self.min_interaction_count)

        
    def isProfileViable(self, user_id: ID):
        return self._profile_viable(user_id, self.min_profile_size)


class ImplicitMFRecommender(LKRecommender):
    def __init__(self):
        super().__init__()
        self.min_user_count = maxsize
        self.min_interaction_count = maxsize
        self.min_profile_size = maxsize

    def setup(self, config):
        super().setup(config)
        # get data from config
        params = config.params
        self.embedding_size = int(params['embedding_size'])
        self.epochs = int(params['epochs'])
        self.regularization = float(params['regularization'])
        self.positive_weight = float(params['positive_weight'])
        self.min_user_count = int(params['min_user_count'])
        self.min_interaction_count = int(params['min_interaction_count'])
        self.min_profile_size = int(params['min_profile_size'])
        self.lk_config = ImplicitMFConfig(embedding_size=self.embedding_size,
                                          epochs=self.epochs,
                                          regularization=self.regularization,
                                          weight=self.positive_weight,
                                          user_embeddings=True,
                                          use_ratings=False)

        self.scorer = ImplicitMFScorer(self.lk_config)
        self.pipeline = self.build_pipeline()


    def build_pipeline(self):
        return self._build_history_pipeline(self.get_scorer())

    def train(self):
        if self._should_train(self.min_user_count, self.min_interaction_count):
            super().train()

    def isDatasetViable(self):
        return self._dataset_viable(self.min_user_count, self.min_interaction_count)
        
    def isProfileViable(self, user_id: ID):
        return self._profile_viable(user_id, self.min_profile_size)
    
class FunkSVDRecommender(LKRecommender):
    def __init__(self):
        super().__init__()
        self.min_user_count = maxsize
        self.min_interaction_count = maxsize
        self.min_profile_size = maxsize
    
    def setup(self, config):
        super().setup(config)
        params = config.params
        self.embedding_size = int(params['embedding_size'])
        self.epochs = int(params['epochs'])
        self.learning_rate = float(params['learning_rate'])
        self.regularization = float(params['regularization'])
        self.min_user_count = int(params['min_user_count'])
        self.min_interaction_count = int(params['min_interaction_count'])
        self.min_profile_size = int(params['min_profile_size'])
        # optional: damping = params.get('damping'), range = params.get('range')
        self.lk_config = FunkSVDConfig(
            embedding_size=self.embedding_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
        )
        self.scorer = FunkSVDScorer(self.lk_config)
        self.pipeline = self.build_pipeline()
    
    def build_pipeline(self):
        return self._build_history_pipeline(self.get_scorer())

    def train(self):
        if self._should_train(self.min_user_count, self.min_interaction_count):
            super().train()

    def isDatasetViable(self):
        return self._dataset_viable(self.min_user_count, self.min_interaction_count)
        
    def isProfileViable(self, user_id: ID):
        return self._profile_viable(user_id, self.min_profile_size)

RecommenderFactory.register('popular', PopularRecommender)
RecommenderFactory.register('item_knn', ItemKnnRecommender)
RecommenderFactory.register('implicit_mf', ImplicitMFRecommender)
RecommenderFactory.register('funk_svd', FunkSVDRecommender)
