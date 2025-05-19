from abc import ABC, abstractmethod
from numpy.linalg import norm
from numpy import dot, average
from icecream import ic

from smores.item import Item, ItemList


class ProviderUtilityModel (ABC):
    """
    ProviderUtilityModel

    A provider utility model computes the utility of an action taken by a consumer relative to one of their
    items.

    All methods are static because there shouldn't be anything provider-specific about these functions.
   """
    @classmethod
    @abstractmethod
    def setup(cls, config):
        pass

    @classmethod
    @abstractmethod
    def compute_item_utility (cls, consumer, item: Item) -> float:
        """
        This function is called when the consumer has clicked on an item belonging to the provider.

        Args:
            consumer: the consumer (probably not used in most cases)
            item: should belong to provider but checks for this

        Returns:
            the calculated utility
        """
        pass

    @classmethod
    @abstractmethod
    def compute_list_utility (cls, consumer, item_list: ItemList) -> float:
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

    @classmethod
    def setup(cls, config):
        cls.utility = config.value

    @classmethod
    def compute_item_utility(cls, consumer, item):
        return cls.utility

    @classmethod
    def compute_list_utility(cls, consumer, item_list: ItemList) -> float:
        # Only clicks count under this model
        return 0.0


class ProviderUtilityModelFactory():
    """
    The ProviderUtilityModelFactory associates names with class objects so these can be passed to
    objects based on configuration information. Note that a utility model is just a collection of
    functions so there is never a need to create an associated object.
    A utility model must registered in the factory before it can be
    created. Note these are all class methods, so an instance of this object never needs to be created.
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
    def get_class(cls, model_name):
        model_class = cls._class_name_map.get(model_name)
        if model_class is None:
            raise UnregisteredProviderUtilityModelError(model_name)
        return model_class


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

