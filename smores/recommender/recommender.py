from abc import ABC, abstractmethod
from icecream import ic
import pyarrow as pa

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
        self.parent = None
        self._cached_user_count = None

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
    
    def setup_dataset(self):
        builder = DatasetBuilder(None)
        builder.add_entity_class('user')
        builder.add_entities('user', smores.Smores.state.consumers.get_consumer_ids())
        builder.add_entity_class('item')
        builder.add_entities('item', smores.Smores.state.items.all_items())
        builder.add_relationship_class('interaction', ['user', 'item'], interaction=True)
        self.dataset = builder.build()

    def get_dataset(self):
        if self.dataset is None:
            return self.parent.get_dataset()
        return self.dataset

    def set_dataset(self, dataset: Dataset):
        if self.dataset is None:
            self.parent.set_dataset(dataset)
        else:
            self.dataset = dataset

    def dataset_active_users(self):
        if self._cached_user_count is None:
            interactions: pa.Table = self.get_dataset().interaction_table(format='arrow', original_ids=True)
            user_col = interactions.column('user_id')
            unique_users = user_col.unique()
            self._cached_user_count = len(unique_users)
        return self._cached_user_count

    @classmethod
    def name2base_recommender(cls, name: str):
        return smores.Smores.state.recommenders_base.get_recommender(name)        

    @abstractmethod
    def train(self):
        cold_start_rec = self.get_cold_start_fallback()
        cold_user_rec = self.get_cold_user_fallback()

        if cold_start_rec is not None:
            cold_start_rec.train()
        if cold_user_rec is not None:
           cold_user_rec.train()
        
        # Invalidate user count cache after training
        self._cached_user_count = None

    def get_cold_start_fallback(self):
        if self.cold_start_fallback is not None:
            cold_start_rec = smores.Smores.state.recommenders_fallback.get_recommender(
                self.cold_start_fallback)
            return cold_start_rec
        else:
            return None
        
    def get_cold_user_fallback(self):
        if self.cold_user_fallback is not None:
            cold_user_rec = smores.Smores.state.recommenders_fallback.get_recommender(
                self.cold_user_fallback)
            return cold_user_rec
        else:
            return None
        

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
        self.set_dataset(hist.to_dataset(self.get_dataset()))

    def update_dataset_itemlist(self, consumer_id, interaction_list: ItemList):
        if len(interaction_list) > 0:
            hist = InteractionHistory()
            hist.add_interactions_itemlist(consumer_id, interaction_list)
            self.set_dataset(hist.to_dataset(self.get_dataset()))
    
    def get_user(self, user_id) -> ItemList | None:
        return self.get_dataset().user_row(user_id)
    
    def delete_user(self, user_id):
        builder = DatasetBuilder(self.get_dataset())
        builder.filter_interactions('interaction', remove={'user_id': [user_id]})
        self.set_dataset(builder.build())           

class FixedItemRecommender(Recommender):
    def __init__(self):
        super().__init__()
        # Needs no training
        self.trained = True

    def setup(self, config):
        self.name = config.name

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


