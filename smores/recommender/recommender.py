from abc import ABC, abstractmethod
from icecream import ic

from lenskit.data import Dataset, DatasetBuilder
from lenskit.data import ItemList

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
        self.trained = False

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
        builder.add_entities('user', smores.Smores.state.consumers.get_consumer_ids())
        builder.add_entity_class('item')
        builder.add_entities('item', smores.Smores.state.items.all_items())
        builder.add_relationship_class('interaction', ['user', 'item'], interaction=True)
        return builder.build()

    @classmethod
    def name2base_recommender(cls, name: str):
        return smores.Smores.state.recommenders_base.get_recommender(name)        

    @abstractmethod
    def train(self):
        if self.cold_start_fallback is not None:
            cold_start_rec = smores.Smores.state.recommenders_fallback.get_recommender(self.cold_start_fallback)
            cold_start_rec.train()
        if self.cold_user_fallback is not None:
            cold_user_rec = smores.Smores.state.recommenders_fallback.get_recommender(self.cold_user_fallback)
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
            cold_rec = smores.Smores.state.recommenders_fallback.get_recommender(fallback_name)
            cold_rec.update_dataset(interaction_list)

    def update_fallback_itemlist(self, fallback_name, interaction_list: ItemList):
        if fallback_name is not None:
            cold_rec = smores.Smores.state.recommenders_fallback.get_recommender(fallback_name)
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
        # Needs no training
        self.trained = True

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


