from abc import ABC, abstractmethod
from numpy.linalg import norm
from numpy import dot, average
from lenskit.data.items import ItemList

from smores.item import Item, ItemCollection
# would like to import but circular issue needs to be resolved
#from .consumer import Consumer

class ConsumerUtilityModel (ABC):
    '''
    ConsumerUtilityModel

    A consumer utility model computes the utility for a consumer of a recommended item, and of a recommendation
    list as a whole.
    '''
    @abstractmethod
    def setup(self, config):
        pass

    @abstractmethod
    def compute_item_utility (self, consumer, item: Item) -> float:
        pass

    def compute_item_utilities (self, consumer, item_list: ItemList) -> list[float]:
        utils = [self.compute_item_utility(consumer, item) for item in item_list]
        return utils

    @abstractmethod
    def compute_list_utility (self, consumer, item_list: ItemList) -> float:
        pass

class ConsumerFixedUtilityModel (ConsumerUtilityModel):
    def __init__(self):
        self.utility: float  = 0.0

    def setup(self, config):
        self.utility = config['value']

    def compute_item_utility(self, consumer, item):
        return self.utility
    
    def compute_list_utility(self, consumer, item):
        return self.utility

class ConsumerPrefCosineUtilityModel (ConsumerUtilityModel):
    def setup(self, config):
        # No configuration information for this model
        pass

    def compute_item_utility(self, consumer, item: Item) -> float:
        
        pref_vector = consumer.preference_vector
        item_vector = item.features
        norm_pref = norm(pref_vector)
        norm_item = norm(item_vector)
        denom = norm_pref * norm_item
        if denom == 0:
            return 0.0
        cos_value = dot(pref_vector, item_vector) / denom
        return cos_value

    @abstractmethod
    def compute_list_utility(self, consumer, item_list: ItemList) -> float:
        pass

class ConsumerPrefCosineAvgUtilityModel (ConsumerPrefCosineUtilityModel):

    def compute_list_utility(self, consumer, item_list: ItemList):
        if len(item_list) == 0:
            return 0.0
        else:
            return average(self.compute_item_utilities(consumer, item_list))


class ConsumerUtilityModelFactory():
    """
    The ConsumerUtilityModelFactory associates names with class objects so these can be passed to
    objects based on configuration information. 
    """

    _class_name_map = {}

    @classmethod
    def register(cls, model_name, model_class):
        if not issubclass(model_class, ConsumerUtilityModel):
            raise InvalidConsumerUtilityModelError(model_name)
        cls._class_name_map[model_name] = model_class

    @classmethod
    def register_all(cls, model_specs):
        for model_name, model_class in model_specs:
            cls.register(model_name, model_class)

    @classmethod
    def create(cls, model_name):
        model_class = cls._class_name_map.get(model_name)
        if model_class is None:
            raise UnregisteredConsumerUtilityModelError(model_name)
        return model_class()

# Registering
ConsumerUtilityModelFactory.register('fixed_utility', ConsumerFixedUtilityModel)
ConsumerUtilityModelFactory.register('list_average', ConsumerPrefCosineAvgUtilityModel)



# Exceptions
class InvalidConsumerUtilityModelError(Exception):
    def __init__(self, name):
        self.message = self.message = f'Cannot create consumer utility model: Class {name} is not a subclass of ConsumerUtilityModel.'
        super().__init__(self.message)


class UnregisteredConsumerUtilityModelError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create consumer utility model: Class {name} is not registered and may not exist.'
        super().__init__(self.message)

