from abc import ABC, abstractmethod
from numpy.linalg import norm
from numpy import dot, average

from smores.item import Item, ItemList

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
    def update_recommender_utility(self, rec_name, time: int, utility: float):
        pass

    @abstractmethod
    def choose_recommender(self) -> str:
        pass

class FixedRecommenderChoiceModel(RecommenderChoiceModel):

    def setup(self, config):
        self.recommender_name = config.recommender_name

    def choose_recommender(self):
        return self.recommender_name
    
    def update_recommender_utility(self, rec_name, time: int, utility: float):
        pass
    

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


# Exceptions
class InvalidRecommenderChoiceModelError(Exception):
    def __init__(self, name):
        self.message = self.message = f'Cannot create consumer utility model: Class {name} is not a subclass of RecommenderChoiceModel.'
        super().__init__(self.message)


class UnregisteredRecommenderChoiceModelError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create recommender choice model: Class {name} is not registered and may not exist.'
        super().__init__(self.message)
