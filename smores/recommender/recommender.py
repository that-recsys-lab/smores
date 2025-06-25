from abc import ABC, abstractmethod
from pydantic import BaseModel
from pyarrow import Table
from sys import maxsize
from icecream import ic
from pandas import DataFrame

from lenskit.data import Dataset, DatasetBuilder
from lenskit.training import Trainable, TrainingOptions
from lenskit.pipeline import Pipeline, PipelineBuilder, Component, topn_pipeline
from lenskit.basic.candidates import UnratedTrainingItemsCandidateSelector
from lenskit.basic import UserTrainingHistoryLookup, TopNRanker
from lenskit.basic.popularity import PopScorer, PopConfig
from lenskit.knn import ItemKNNConfig, ItemKNNScorer
from lenskit.data import ID, ItemList, from_interactions_df
from lenskit import recommend

import smores
from smores.utils import InteractionHistory, PythonClassConfig


class Recommender(ABC):
    def __init__(self):
        self.dataset: Dataset = None
        # Use this if there isn't enough data overall
        self.cold_start_fallback: str = None
        # Use this if there isn't enough data for a particular user
        self.cold_user_fallback: str = None
        self.name = None

    @abstractmethod
    def setup(self, config):
        if type(config) is PythonClassConfig:
            params = config.params
            if params is not None:
                if 'cold_start_fallback' in params:
                    self.cold_start_fallback = params['cold_start_fallback']
                if 'cold_user_fallback' in params: 
                    self.cold_user_fallback = params['cold_user_fallback']
        self.name = config.name
        self.dataset = self.setup_dataset()
    
    def setup_dataset(self):
        builder = DatasetBuilder(None)
        builder.add_entity_class('user')
        builder.add_relationship_class('interaction', ['user', 'item'], interaction=True)
        return builder.build()

    @classmethod
    def name2recommender(cls, name: str):
        return smores.Smores.state.recommenders_available.get_recommender(name)        

    @abstractmethod
    def train(self):
        if self.cold_start_fallback is not None:
            cold_start_rec = smores.Smores.state.recommenders_available.get_recommender(self.cold_start_fallback)
            cold_start_rec.train()
        if self.cold_user_fallback is not None:
            cold_user_rec = smores.Smores.state.recommenders_available.get_recommender(self.cold_user_fallback)
            cold_user_rec.train()
        

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
        self.update_fallback(self.cold_start_fallback, interaction_list)
        self.update_fallback(self.cold_user_fallback, interaction_list)

    def update_fallback(self, fallback_name, interaction_list: list):
        if fallback_name is not None:
            cold_rec = smores.Smores.state.recommenders_available.get_recommender(fallback_name)
            cold_rec.update_dataset(interaction_list)

    def update_fallback_itemlist(self, fallback_name, interaction_list: ItemList):
        if fallback_name is not None:
            cold_rec = smores.Smores.state.recommenders_available.get_recommender(fallback_name)
            cold_rec.update_dataset_itemlist(interaction_list)

    def update_dataset_itemlist(self, interaction_list: ItemList):
        if len(interaction_list) > 0:
            hist = InteractionHistory()
            hist.add_interactions_itemlist(interaction_list)
            self.dataset = hist.to_dataset(self.dataset)
            self.update_fallback_itemlist(self.cold_start_fallback, interaction_list)
            self.update_fallback_itemlist(self.cold_user_fallback, interaction_list)

    def get_user(self, user_id) -> ItemList | None:
        return self.dataset.user_row(user_id)
    
    def delete_user(self, user_id):
        builder = DatasetBuilder(self.dataset)
        builder.filter_interactions('interaction', remove={'user_id': [user_id]})
        self.dataset = builder.build()
        
    

class FixedItemRecommender(Recommender):
    def __init__(self):
        super().__init__()

    def setup(self, config):
        self.name = config.name
        self.dataset = self.setup_dataset()

    def isDatasetViable(self):
        return True
    
    def isProfileViable(self, user_id):
        return True
    
    def train(self):
        pass
    
    def get_recommendations(self, user_id) -> ItemList:
        prior_interactions = self.get_user(user_id)
        if prior_interactions is not None and len(prior_interactions) > 0:
            rec_pool = [item for item in list(smores.Smores.state.items.all_items()) if item not in prior_interactions.ids()]
        else:
            rec_pool = list(smores.Smores.state.items.all_items())
        recs = rec_pool[0:smores.Smores.state.slate_size]
        scores = [5.0] * len(recs)
        ranks = list(range(1, len(recs)+1))
        item_list = ItemList(None, item_ids=recs, scores=scores, rank=ranks)
        return item_list


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
        # Don't train on an empty dataset
        if self.dataset.interaction_count > 0:
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
            cold_start_rec = smores.Smores.state.recommenders_available.get_recommender(self.cold_start_fallback)
            return cold_start_rec.get_recommendations(user_id)
        elif not self.isProfileViable(user_id):
            cold_user_rec = smores.Smores.state.recommenders_available.get_recommender(self.cold_user_fallback)
            return cold_user_rec.get_recommendations(user_id)
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

    def train(self):
        if self.dataset.interaction_count >= self.min_interaction_count and \
                self.dataset.user_count >= self.min_user_count:
            super().train()

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
RecommenderFactory.register('fixed_recommender', FixedItemRecommender)


# Exceptions
class InvalidRecommenderError(Exception):
    def __init__(self, name):
        self.message = self.message = f'Cannot create recommender: Class {name} is not a subclass of Recommender.'
        super().__init__(self.message)


class UnregisteredRecommenderError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create recommender: Class {name} is not registered and may not exist.'
        super().__init__(self.message)


