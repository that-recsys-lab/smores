from abc import ABC, abstractmethod
from numpy.linalg import norm
from numpy import dot, average
from icecream import ic

from lenskit.data import ItemList
from lenskit.data import ID


class ProviderUtilityModel (ABC):
    """
    ProviderUtilityModel

    A provider utility model computes the utility of an action taken by a consumer relative to one of their
    items.

    All methods are static because there shouldn't be anything provider-specific about these functions.
   """
    @abstractmethod
    def setup(self, config):
        pass

    @abstractmethod
    def compute_item_utility (self, consumer, item: ID) -> float:
        """
        This function is called when the consumer has clicked on an item belonging to the provider.

        Args:
            consumer: the consumer (probably not used in most cases)
            item: should belong to provider but checks for this

        Returns:
            the calculated utility
        """
        pass

    @abstractmethod
    def compute_list_utility (self, consumer, item_list: ItemList) -> float:
        """
        This function is called when the consumer is shown a list.

        Args:
            consumer: the consumer (probably not used in most cases)
            item_list: may contain items from any provider

        Returns:
            the calculated utility
        """
        pass

class ProviderClickFixedUtilityModel (ProviderUtilityModel):
    """
    Only clicks count under this utility model.
    """
    utility: float  = 0.0

    def setup(self, config):
        self.utility = config['value']

    def compute_item_utility(self, consumer, item):
        return self.utility

    def compute_list_utility(self, consumer, item_list: ItemList) -> float:
        # Only clicks count under this model
        return 0.0


class ProviderUtilityModelFactory():
    """
    The ProviderUtilityModelFactory associates names with class objects so these can be passed to
    objects based on configuration information. .
    """

    _class_name_map = {}

    @classmethod
    def register(cls, model_name, model_class):
        if not issubclass(model_class, ProviderUtilityModel):
            raise InvalidProviderUtilityModelError(model_name)
        cls._class_name_map[model_name] = model_class

    @classmethod
    def register_all(cls, model_specs):
        for model_name, model_class in model_specs:
            cls.register(model_name, model_class)

    @classmethod
    def create(cls, model_name):
        model_class = cls._class_name_map.get(model_name)
        if model_class is None:
            raise UnregisteredProviderUtilityModelError(model_name)
        return model_class()


# Registering
ProviderUtilityModelFactory.register('click_fixed', ProviderClickFixedUtilityModel)




# Exceptions
class InvalidProviderUtilityModelError(Exception):
    def __init__(self, name):
        self.message = self.message = f'Cannot create consumer utility model: Class {name} is not a subclass of ConsumerUtilityModel.'
        super().__init__(self.message)


class UnregisteredProviderUtilityModelError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create consumer utility model: Class {name} is not registered and may not exist.'
        super().__init__(self.message)

