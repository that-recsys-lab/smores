from pydantic import BaseModel
from sys import maxsize
from icecream import ic
from abc import abstractmethod

from lenskit.pipeline import Pipeline, PipelineBuilder, Component, topn_pipeline
from lenskit.basic.candidates import UnratedTrainingItemsCandidateSelector
from lenskit.basic import UserTrainingHistoryLookup, TopNRanker
from lenskit.basic.popularity import PopScorer, PopConfig
from lenskit.knn import ItemKNNConfig, ItemKNNScorer
from lenskit.als import ImplicitMFConfig, ImplicitMFScorer
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

    def get_recommendations(self, user_id: ID):
        if not self.isDatasetViable():
            cold_start_rec = smores.Smores.state.recommenders_fallback.get_recommender(self.cold_start_fallback)
            #smores.Smores.state.logger.debug(f'.       Insufficent interaction data. Fallback to {self.cold_start_fallback}')
            # Fallback recommenders already return fully sampled slates
            return cold_start_rec.get_recommendations(user_id)
        elif not self.isProfileViable(user_id):
            cold_user_rec = smores.Smores.state.recommenders_fallback.get_recommender(self.cold_user_fallback)
            #smores.Smores.state.logger.debug(f'.       Insufficent profile data. Fallback to {self.cold_user_fallback}')
            # Fallback recommenders already return fully sampled slates
            return cold_user_rec.get_recommendations(user_id)
        else:
            recs = recommend(self.pipeline, user_id)
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
    def train(self):
        super().train()
    
    def build_pipeline(self):
        scorer = self.get_scorer()
        slate_size = max(smores.Smores.state.slate_size - self.sampled_item_count, 1)
        return topn_pipeline(scorer, n=slate_size)

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
        scorer = self.get_scorer()
        slate_size = max(smores.Smores.state.slate_size - self.sampled_item_count, 1)

        pipe = PipelineBuilder()
        # define an input parameter for the user ID (the 'query')
        query = pipe.create_input('query', ID)
        # look up a user's history in the training data
        history = pipe.add_component('history-lookup', UserTrainingHistoryLookup, query=query)
        # find candidates from the training data
        default_candidates = pipe.add_component('candidate-selector',
            UnratedTrainingItemsCandidateSelector, query=history)
        # score the candidate items using the specified scorer
        score = pipe.add_component('scorer', scorer, query=query, items=default_candidates)
        # rank the items by score
        recommend = pipe.add_component('ranker', TopNRanker, {'n': slate_size}, items=score)
        pipe.alias('recommender', recommend)
        pipe.default_component('recommender')
        return pipe.build()

    def train(self):
        if self.get_dataset().interaction_count >= self.min_interaction_count and \
                self.get_dataset().user_count >= self.min_user_count:
            super().train()

    def isDatasetViable(self):
        if not self.trained:
            return False
        else:
            user_count = self.dataset_active_users()
            interaction_count = self.get_dataset().interaction_count
            if user_count >= self.min_user_count and interaction_count >= self.min_interaction_count:
                # smores.Smores.state.logger.debug(f"Recommender: {self.name} is viable. Interaction count {interaction_count}. User count {user_count}")
                return True
            else:
                return False

        
    def isProfileViable(self, user_id: ID):
        items = self.get_dataset().user_row(user_id)
        if items is None:
            return False
        else:
            profile_size = items.ids().size
            if profile_size < self.min_profile_size:
                return False
            else:
                # smores.Smores.state.logger.debug(f"Recommender: {self.name} user {user_id} is viable.")
                return True


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
        scorer = self.get_scorer()
        slate_size = max(smores.Smores.state.slate_size - self.sampled_item_count, 1)

        pipe = PipelineBuilder()
        # define an input parameter for the user ID (the 'query')
        query = pipe.create_input('query', ID)
        # look up a user's history in the training data
        history = pipe.add_component('history-lookup', UserTrainingHistoryLookup, query=query)
        # find candidates from the training data
        default_candidates = pipe.add_component('candidate-selector',
            UnratedTrainingItemsCandidateSelector, query=history)
        # score the candidate items using the specified scorer
        score = pipe.add_component('scorer', scorer, query=query, items=default_candidates)
        # rank the items by score
        recommend = pipe.add_component('ranker', TopNRanker, {'n': slate_size}, items=score)
        pipe.alias('recommender', recommend)
        pipe.default_component('recommender')
        return pipe.build()

    def train(self):
        if self.get_dataset().interaction_count >= self.min_interaction_count and \
                self.get_dataset().user_count >= self.min_user_count:
            super().train()

    def isDatasetViable(self):
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
        items = self.get_dataset().user_row(user_id)
        if items is None:
            return False
        else:
            profile_size = items.ids().size
            if profile_size < self.min_profile_size:
                return False
            else:
                return True

RecommenderFactory.register('popular', PopularRecommender)
RecommenderFactory.register('item_knn', ItemKnnRecommender)
RecommenderFactory.register('implicit_mf', ImplicitMFRecommender)
