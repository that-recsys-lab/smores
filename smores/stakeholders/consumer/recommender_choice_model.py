import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

import smores
import smores.stakeholders.consumer as consumer

class RecommenderChoiceModel (ABC):
    '''
    RecommenderChoiceModel

    A recommender choice model uses the historical utility of a consumer for a recommendation algorithm to decide which algorithm
    to use.
    '''
    def __init__(self):
        self.consumer = None

    @abstractmethod
    def setup(self, config):
        pass

    @abstractmethod
    def update_recommender_utility(self, list_utility: float) -> float:
        return 0

    @abstractmethod
    def choose_recommender(self) -> str:
        pass

    def set_consumer(self, consumer):
        self.consumer = consumer

class FixedRecommenderChoiceModel(RecommenderChoiceModel):

    def setup(self, config):
        self.recommender_name = config.params['recommender_name']

    def choose_recommender(self):
        return self.recommender_name
    
    def update_recommender_utility(self, list_utility: float):
        return 0
    
class ThresholdRecommenderChoiceModel(RecommenderChoiceModel):
    def __init__(self):
        super().__init__()

    def setup(self, config):
        self.threshold = config.params['threshold']
        self.beta = config.params['beta']
        self.recommender_utilities = defaultdict(int)
        self.recommender_ucbs = defaultdict(float)

    def choose_recommender(self):
        maybe_new_recommender = self.consumer.recommender.name
        current_utility = self.recommender_utilities[ maybe_new_recommender]
        if current_utility < self.threshold:
            for recommender_name in smores.Smores.state.recommenders_active:
                utility = self.recommender_utilities[recommender_name] 
                if utility > current_utility:
                    maybe_new_recommender = recommender_name
                    current_utility = utility
        return maybe_new_recommender
    
    def update_recommender_utility(self, list_utility: float):
        current_recommender_name = self.consumer.recommender.name
        prev_utility = self.recommender_utilities[current_recommender_name]
        new_utility = ((prev_utility * self.beta) + list_utility)/(1 + self.beta)
        self.recommender_utilities[current_recommender_name] = new_utility
        return new_utility
    
class UCBRecommenderChoiceModel(RecommenderChoiceModel):
    def __init__(self):
        super().__init__()

    def setup(self, config):
        self.beta = config.params['beta']
        self.recommender_utilities = defaultdict(int)
        self.recommender_count = defaultdict(int)
        self.recommender_time = defaultdict(int)

    def choose_recommender(self):
        max_ucb = 0
        maybe_new_recommender = self.consumer.recommender.name
        for recommender_name in smores.Smores.state.recommenders_active:
            utility = self.recommender_utilities[recommender_name] 
            time = self.recommender_time[recommender_name]
            count = self.recommender_count[recommender_name]
            if time == 0:
                continue
            ucb = utility + np.sqrt(2*np.log(time)/count)/(1+time)
            if ucb > max_ucb:
                maybe_new_recommender = recommender_name
                max_ucb = ucb
        self.recommender_count[maybe_new_recommender] += 1
        return maybe_new_recommender
    
    def update_recommender_utility(self, list_utility: float):
        current_recommender_name = self.consumer.recommender.name
        prev_utility = self.recommender_utilities[current_recommender_name]
        new_utility = ((prev_utility * self.beta) + list_utility)/(1 + self.beta)
        self.recommender_utilities[current_recommender_name] = new_utility
        self.recommender_time[current_recommender_name] += 1
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
