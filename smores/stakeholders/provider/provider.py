from icecream import ic

from smores.utils import UtilityHistory
from .provider_utility_model import ProviderUtilityModelFactory


class Provider:
    _id: int = -1

    @classmethod
    def next_id(cls) -> int:
        cls._id += 1
        return cls._id

    def __init__(self):
        self.id = None
        self.items = None
        self.history = None
        self.utility_model = None
        self.recommenders = None
        self.items = None

    def __str__(self):
        return f'<Provider {self.id}>'

    def setup(self, config):
        self.id = Provider.next_id()

        self.history = UtilityHistory()

        utility_model_config = config.provider.utility_model
        self.utility_model = ProviderUtilityModelFactory.get_class(utility_model_config.class_name)
        self.utility_model.setup(utility_model_config)


class ProviderCollection():
    def __init__(self):
        self.collection = []

    def add_provider(self, cons: Provider):
        self.collection.append(cons)

    def __iter__(self):
        return self.collection.__iter__()

class ProviderFactory():
    """
    The ProviderFactory associates provider types with class names and allows appropriate instances
    to be created. A provider class must registered in the factory before it can be
    created.
    """

    _class_name_map = {}

    @classmethod
    def register(cls, type_name, provider_class):
        if not issubclass(provider_class, Provider):
            raise InvalidProviderError(type_name)
        cls._class_name_map[type_name] = provider_class

    @classmethod
    def register_all(cls, type_specs):
        for type_name, provider_class in type_specs:
            cls.register(type_name, provider_class)

    @classmethod
    def make_object(cls, type_name):
        provider_class = cls._class_name_map.get(type_name)
        if provider_class is None:
            raise UnregisteredProviderError(type_name)
        return provider_class()



# Exceptions
class InvalidProviderError(Exception):
    def __init__(self, name):
        self.message = self.message = f'Cannot create Provider object: Class {name} is not a subclass of Provider.'
        super().__init__(self.message)


class UnregisteredProviderError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create Provider object: Class {name} is not registered and may not exist.'
        super().__init__(self.message)

