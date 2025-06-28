import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

import smores

class RecommenderChoiceModel (ABC):
    '''
    RecommenderChoiceModel

    A recommender choice model uses the historical utility of a consumer for a recommendation algorithm to decide which algorithm
    to use.
    '''
    @abstractmethod
    def setup(self, config):
        pass

    @abstractmethod
    def update_recommender_utility(self, list_utility: float) -> float:
        return 0

    @abstractmethod
    def choose_recommender(self) -> str:
        pass

class FixedRecommenderChoiceModel(RecommenderChoiceModel):

    def setup(self, config):
        self.recommender_name = config.params['recommender_name']

    def choose_recommender(self):
        return self.recommender_name
    
    def update_recommender_utility(self, list_utility: float):
        return 0
    
class ThresholdRecommenderChoiceModel(RecommenderChoiceModel):

    def setup(self, config):
        self.recommender_name = config.params['recommender_name']
        self.threshold = config.params['threshold']
        self.beta = config.params['beta']
        self.recommender_utilities = defaultdict(int)

    def choose_recommender(self):
        current_utility = self.recommender_utilities[self.recommender_name] 
        if current_utility < self.threshold:
            for recommender_name, _ in smores.Smores.state.recommenders_available.items():
                utility = self.recommender_utilities[recommender_name] 
                if utility > current_utility:
                    self.recommender_name = recommender_name
                    current_utility = utility
        return self.recommender_name
    
    def update_recommender_utility(self, list_utility: float):
        prev_utility = self.recommender_utilities[self.recommender_name]
        new_utility = ((prev_utility * self.beta) + list_utility)/(1 + self.beta)
        self.recommender_utilities[self.recommender_name] = new_utility
        return new_utility
    
class UCBRecommenderChoiceModel(RecommenderChoiceModel):

    def setup(self, config):
        self.recommender_name = config.params['recommender_name']
        self.beta = config.params['beta']
        self.recommender_utilities = defaultdict(int)
        self.recommender_count = defaultdict(int)
        self.recommender_time = defaultdict(int)

    def choose_recommender(self):
        max_ucb = 0
        for recommender_name, _ in smores.Smores.state.recommenders_available.items():
            utility = self.recommender_utilities[recommender_name] 
            time = self.recommender_time[recommender_name]
            count = self.recommender_count[recommender_name]
            ucb = utility + np.sqrt(2*np.log(time)/count)/(1+time)
            if ucb > max_ucb:
                self.recommender_name = recommender_name
                max_ucb = ucb
        self.recommender_count[self.recommender_name] += 1
        return self.recommender_name
    
    def update_recommender_utility(self, list_utility: float):
        prev_utility = self.recommender_utilities[self.recommender_name]
        new_utility = ((prev_utility * self.beta) + list_utility)/(1 + self.beta)
        self.recommender_utilities[self.recommender_name] = new_utility
        self.recommender_time[self.recommender_name] += 1
        return new_utility

class RecommenderChoiceModelFactory():
    """
    The RecommenderChoiceModelFactory associates names with objects so these can be passed to
    objects based on configuration information. Choice models maintain a representation of the
    user's utility for each recommender and therefore must instances. 
    """

    _class_name_map = {}

    @classmethod
    def register(cls, model_name, model_class):
        if not issubclass(model_class, RecommenderChoiceModel):
            raise InvalidRecommenderChoiceModelError(model_name)
        cls._class_name_map[model_name] = model_class

    @classmethod
    def register_all(cls, model_specs):
        for model_name, model_class in model_specs:
            cls.register(model_name, model_class)

    @classmethod
    def create(cls, model_name):
        model_class = cls._class_name_map.get(model_name)
        if model_class is None:
            raise UnregisteredRecommenderChoiceModelError(model_name)
        return model_class()

# Registering
RecommenderChoiceModelFactory.register('fixed', FixedRecommenderChoiceModel)
RecommenderChoiceModelFactory.register('threshold', ThresholdRecommenderChoiceModel)
RecommenderChoiceModelFactory.register('ucb', UCBRecommenderChoiceModel)


# Exceptions
class InvalidRecommenderChoiceModelError(Exception):
    def __init__(self, name):
        self.message = self.message = f'Cannot create consumer utility model: Class {name} is not a subclass of RecommenderChoiceModel.'
        super().__init__(self.message)


class UnregisteredRecommenderChoiceModelError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create recommender choice model: Class {name} is not registered and may not exist.'
        super().__init__(self.message)
