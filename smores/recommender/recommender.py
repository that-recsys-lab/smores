from abc import ABC, abstractmethod
from pydantic import BaseModel
from pyarrow import Table
from sys import maxsize
from icecream import ic

from lenskit.data import Dataset, DatasetBuilder
from lenskit.training import Trainable, TrainingOptions
from lenskit.pipeline import Pipeline, PipelineBuilder, Component, topn_pipeline
from lenskit.basic.candidates import UnratedTrainingItemsCandidateSelector
from lenskit.basic import UserTrainingHistoryLookup, TopNRanker
from lenskit.basic.popularity import PopScorer, PopConfig
from lenskit.knn import ItemKNNConfig, ItemKNNScorer
from lenskit.data import ID, ItemList
from lenskit import recommend

import smores
from smores.utils import InteractionHistory, PythonClassConfig


class Recommender(ABC):
    def __init__(self):
        self.dataset: Dataset = None
        # Use this if there isn't enough data overall
        self.cold_start_fallback: Recommender = None
        # Use this if there isn't enough data for a particular user
        self.cold_user_fallback: Recommender = None

    @abstractmethod
    def setup(self, config):
        if type(config) is PythonClassConfig:
            params = config.params
            if params is not None:
                if 'cold_start_fallback' in params:
                    self.cold_start_fallback = params['cold_start_fallback']
                if 'cold_user_fallback' in params: 
                    self.cold_user_fallback = params['cold_user_fallback']

    @classmethod
    def name2recommender(cls, name: str):
        return smores.Smores.state.recommenders_available.get_recommender(name)        

    @abstractmethod
    def train(self):
        if self.cold_start_fallback is not None:
            self.cold_start_fallback.train()
        if self.cold_user_fallback is not None:
            self.cold_user_fallback.train()
        

    @abstractmethod
    def isDatasetViable(self):
        pass

    @abstractmethod
    def isProfileViable(self, user_id):
        pass

    @abstractmethod
    def get_recommendations(self, user_id) -> ItemList:
        pass

    def update_dataset(self, interaction_list: list):
        hist = InteractionHistory()
        hist.add_interactions(interaction_list)
        self.dataset = hist.to_dataset(self.dataset)




class LKRecommender(Recommender):
    def __init__(self):
        super().__init__()
        self.pipeline: Pipeline = None
        self.lk_config: BaseModel
        self.scorer: Component = None

    def setup(self, config):
        super().setup(config)
        self.pipeline = self.build_pipeline()

    def get_scorer(self):
        return self.scorer

    def train(self):
        super().train()
        self.scorer.train(self.dataset)
    
    def build_pipeline(self):
        scorer = self.get_scorer()
        slate_size = smores.Smores.state.slate_size

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

    def get_recommendations(self, user_id: ID):
        if not self.isDatasetViable():
            return self.cold_start_fallback.get_recommendations(user_id)
        elif not self.isProfileViable(user_id):
            return self.cold_user_fallback.get_recommendations(user_id)
        else:
            return recommend(self.pipeline, user_id, n=smores.Smores.state.slate_size)


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
        

class ItemKnnRecommender(LKRecommender):
    def __init__(self):
        super().__init__()
        self.min_user_count = maxsize
        self.min_interaction_count = maxsize
        self.min_profile_size = maxsize

    def setup(self, config):
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
        super().setup(config)

    def isDatasetViable(self):
        user_count = self.dataset.user_count
        interaction_count = self.dataset.interaction_count
        if user_count >= self.min_user_count and interaction_count >= self.min_interaction_count:
            return True
        else:
            return False
        
    def isProfileViable(self, user_id: ID):
        items = self.dataset.user_row(user_id)
        if items is None:
            return False
        else:
            profile_size = items.ids().size
            if profile_size < self.min_profile_size:
                return False
            else:
                return True


class RecommenderFactory():
    """
    The RecommenderFactory associates names with recommender objects so these can be passed to
    objects based on configuration information. A utility model must registered in the factory before it can be
    created.
    """

    _class_name_map = {}

    @classmethod
    def register(cls, rec_name, rec_class):
        if not issubclass(rec_class, Recommender):
            raise InvalidRecommenderError(rec_name)
        cls._class_name_map[rec_name] = rec_class

    @classmethod
    def register_all(cls, rec_specs):
        for rec_name, rec_class in rec_specs:
            cls.register(rec_name, rec_class)

    @classmethod
    def create(cls, rec_name):
        rec_class = cls._class_name_map.get(rec_name)
        if rec_class is None:
            raise UnregisteredRecommenderError(rec_name)
        return rec_class()

# Registering
RecommenderFactory.register('item_knn', ItemKnnRecommender)
RecommenderFactory.register('popular', PopularRecommender)


# Exceptions
class InvalidRecommenderError(Exception):
    def __init__(self, name):
        self.message = self.message = f'Cannot create recommender: Class {name} is not a subclass of Recommender.'
        super().__init__(self.message)


class UnregisteredRecommenderError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create recommender: Class {name} is not registered and may not exist.'
        super().__init__(self.message)


