from abc import ABC, abstractmethod
from numpy.linalg import norm
from numpy import dot, average

from smores.item import Item, ItemList
# would like to import but circular issue needs to be resolved
#from smores.stakeholders.consumer import Consumer

class ConsumerUtilityModel (ABC):
    '''
    ConsumerUtilityModel

    A consumer utility model computes the utility for a consumer of a recommended item, and of a recommendation
    list as a whole.

    All methods are static because there shouldn't be anything user-specific about these functions.
    '''
    @classmethod
    @abstractmethod
    def setup(cls, config):
        pass

    @classmethod
    @abstractmethod
    def compute_item_utility (cls, consumer, item: Item) -> float:
        raise NotImplementedError()

    @classmethod
    def compute_item_utilities (cls, consumer, item_list: ItemList) -> list[float]:
        utils = [cls.compute_item_utility(consumer, item) for item in item_list]
        return utils

    @classmethod
    @abstractmethod
    def compute_list_utility (cls, consumer, item_list: ItemList) -> float:
        raise NotImplementedError()

class ConsumerFixedUtilityModel (ConsumerUtilityModel):
    utility: float  = 0.0

    @classmethod
    def setup(cls, config):
        cls.utility = config.utility_model.value

    @classmethod
    def compute_item_utility(cls, consumer, item):
        return cls.utility

class ConsumerPrefCosineUtilityModel (ConsumerUtilityModel):

    @classmethod
    def compute_item_utility(cls, consumer, item: Item) -> float:
        pref_vector = consumer.preference_vector
        item_vector = item.genre_vector
        norm_pref = norm(pref_vector)
        norm_item = norm(item_vector)
        denom = norm_pref * norm_item
        if denom == 0:
            return 0.0
        cos_value = dot(pref_vector, item_vector) / denom
        return cos_value

class ConsumerPrefCosineAvgUtilityModel (ConsumerPrefCosineUtilityModel):

    @classmethod
    def compute_list_utility(cls, consumer, item_list: ItemList) -> float:
        if item_list.size() == 0:
            return 0
        else:
            return average(cls.compute_item_utilities(cls, consumer, item_list))


class ConsumerUtilityModelLookup():
    """
    The ConsumerUtilityModelFactory associates names with class objects so these can be passed to
    objects based on configuration information. Note that a utility model is just a collection of
    functions so there is never a need to create an associated object.
    A utility model must registered in the factory before it can be
    created. Note these are all class methods, so an instance of this object never needs to be created.
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
    def get_class(cls, model_name):
        model_class = cls._class_name_map.get(model_name)
        if model_class is None:
            raise UnregisteredConsumerUtilityModelError(model_name)
        return model_class

# Registering
ConsumerUtilityModelLookup.register('fixed', ConsumerFixedUtilityModel)
ConsumerUtilityModelLookup.register('list_average', ConsumerPrefCosineAvgUtilityModel)



# Exceptions
class InvalidConsumerUtilityModelError(Exception):
    def __init__(self, name):
        self.message = self.message = f'Cannot create consumer utility model: Class {name} is not a subclass of ConsumerUtilityModel.'
        super().__init__(self.message)


class UnregisteredConsumerUtilityModelError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create consumer utility model: Class {name} is not registered and may not exist.'
        super().__init__(self.message)

