import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict

import smores
import smores.stakeholders.consumer as consumer
from smores.utils import ChoiceUtility

class RecommenderChoiceModel (ABC):
    '''
    RecommenderChoiceModel

    A recommender choice model uses the historical utility of a consumer for a recommendation algorithm to decide which algorithm
    to use.
    '''
    def __init__(self):
        self.consumer: Consumer = None
        self.next_recommender: str = '__None'

    @abstractmethod
    def setup(self, config):
        pass

    @abstractmethod
    def update_recommender_utility(self, list_utility: float) -> float:
        return 0

    # Must call log_utilities. Probably a better way to do this.
    @abstractmethod
    def choose_recommender(self) -> str:
        pass

    def set_consumer(self, consumer):
        self.consumer = consumer

    @abstractmethod
    def log_utilities(self):
        pass
        

class FixedRecommenderChoiceModel(RecommenderChoiceModel):

    def setup(self, config):
        self.recommender_name = config.params['recommender_name']
        self.next_recommender = self.recommender_name

    def choose_recommender(self):
        self.log_utilities()
        return self.recommender_name
    
    def update_recommender_utility(self, list_utility: float):
        return 0

    def log_utilities(self):
        logger = smores.Smores.state.logger
        time = smores.Smores.state.cycle_count
        recommenders = smores.Smores.state.recommenders_base.get_names()
        tuple = ChoiceUtility(self.consumer.id, self.consumer.type, self.recommender_name, \
                              self.recommender_name, [0] * len(recommenders), time)
        logger.log_recommender_choice(tuple)
    
class ThresholdRecommenderChoiceModel(RecommenderChoiceModel):
    def __init__(self):
        super().__init__()

    def setup(self, config):
        self.threshold = config.params['threshold']
        self.beta = config.params['beta']
        self.recommender_utilities = defaultdict(lambda: np.nan)

    def setup_recommender_utilities(self):
        self.recommender_utilities = {name: np.nan for name in smores.Smores.state.recommenders_base.get_names()}

    def choose_recommender(self):
        maybe_new_recommender = self.consumer.recommender.name
        current_utility = self.recommender_utilities[maybe_new_recommender]
        if current_utility < self.threshold:
            for recommender_name in smores.Smores.state.recommenders_active:
                if recommender_name != self.consumer.recommender.name:
                    utility = self.recommender_utilities[recommender_name] 
                    if np.isnan(utility):
                        maybe_new_recommender = recommender_name
                        self.recommender_utilities[recommender_name] = 0
                        break
                    else:
                        if utility >= current_utility:
                            maybe_new_recommender = recommender_name
                            current_utility = utility
        self.next_recommender = maybe_new_recommender
        self.log_utilities()
        return maybe_new_recommender
    
    def update_recommender_utility(self, list_utility: float):
        current_recommender_name = self.consumer.recommender.name
        prev_utility = self.recommender_utilities[current_recommender_name]
        if np.isnan(prev_utility):
            prev_utility = 0
        new_utility = ((prev_utility * self.beta) + list_utility)/(1 + self.beta)
        self.recommender_utilities[current_recommender_name] = new_utility
        return new_utility
    
    def log_utilities(self):
        logger = smores.Smores.state.logger
        time = smores.Smores.state.cycle_count
        recommenders = smores.Smores.state.recommenders_base.get_names()
        utilities = [self.recommender_utilities[name] for name in recommenders]
        current_recommender = self.consumer.recommender.name
        tuple = ChoiceUtility(self.consumer.id, self.consumer.type, current_recommender, \
                              self.next_recommender, utilities, time)
        logger.log_recommender_choice(tuple)
    
class UCBRecommenderChoiceModel(RecommenderChoiceModel):
    def __init__(self):
        super().__init__()

    def setup(self, config):
        self.beta = config.params['beta']
        self.recommender_utilities = defaultdict(float)
        self.recommender_count = defaultdict(int)
        self.recommender_ucbs = defaultdict(float)

    def choose_recommender(self):
        old_recommender = self.consumer.recommender.name
        self.recommender_count[old_recommender] += 1

        max_ucb = 0
        maybe_new_recommender = old_recommender
        for recommender_name in smores.Smores.state.recommenders_active:
            utility = self.recommender_utilities[recommender_name] 
            days_per_cycle = smores.Smores.state.day_limit
            time = smores.Smores.state.current_time()
            count = self.recommender_count[recommender_name]

            if count > 0:
                # This version of UCB counts the days within a cycle as "experience" relative
                # to the current recommender. Should enable faster convergence
                ucb = utility + np.sqrt(2*np.log(time+1) / (count * days_per_cycle))
                # ucb = utility + np.sqrt(2*np.log(cycle_count+1)/count)
                self.recommender_ucbs[recommender_name] = ucb
                if ucb > max_ucb:
                    maybe_new_recommender = recommender_name
                    max_ucb = ucb
            elif recommender_name != self.consumer.recommender.name:
                # Always sample untried options
                # Might be better to do this randomly if more than 2 options
                #smores.Smores.state.logger.debug(f"Switching from {maybe_new_recommender} to {recommender_name}")
                maybe_new_recommender = recommender_name
                break

        self.next_recommender = maybe_new_recommender
        self.log_utilities()
        
        return self.next_recommender
    
    def update_recommender_utility(self, list_utility: float):
        current_recommender_name = self.consumer.recommender.name
        prev_utility = self.recommender_utilities[current_recommender_name]
        new_utility = ((prev_utility * self.beta) + list_utility)/(1 + self.beta)
        self.recommender_utilities[current_recommender_name] = new_utility
        return new_utility
    
    def log_utilities(self):
        logger = smores.Smores.state.logger
        cycle = smores.Smores.state.cycle_count
        recommenders = smores.Smores.state.recommenders_base.get_names()
        utilities = [self.recommender_utilities[name] for name in recommenders]
        current_recommender = self.consumer.recommender.name
        tuple = ChoiceUtility(self.consumer.id, self.consumer.type, current_recommender, \
                              self.next_recommender, utilities, cycle)
        # logger.debug(f"UCBs: {self.recommender_ucbs['']}")
        logger.log_recommender_choice(tuple)

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
