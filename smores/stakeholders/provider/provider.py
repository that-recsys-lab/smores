from icecream import ic
from pydantic import BaseModel, NonNegativeInt
from csv import DictReader
from pathlib import Path

from smores.utils import ProviderTypeConfig, ProviderUtility
from .provider_utility_model import ProviderUtilityModel
import smores


class ProviderInfo (BaseModel):
    provider_id: NonNegativeInt
    provider_type: str


class Provider:
    def __init__(self):
        self.id = -1
        self.type = None
        self.utility_model: ProviderUtilityModel = None
        self.recommenders = None

    def __str__(self):
        return f'<Provider {self.id}>'

    def setup(self, config_type: ProviderTypeConfig, config: ProviderInfo):
        self.id = config.provider_id
        self.type = config.provider_type
        provider_models = smores.Smores.state.provider_models

        # Get utility model
        utility_model_name = config_type.utility_model
        self.utility_model = provider_models.get_utility_model(utility_model_name)

    def update_utility_item(self, consumer, recommender, item, time):
        utility_value = self.utility_model.compute_item_utility(consumer, item)
        if utility_value > 0:
            smores.Smores.state.logger.log_provider(ProviderUtility(self.id, self.type, recommender.name, utility_value, time))

    def update_utility_list(self, consumer, recommender, item_list, time):
        utility_value = self.utility_model.compute_list_utility(consumer, item_list)
        if utility_value > 0:
            smores.Smores.state.logger.log_provider(ProviderUtility(self.id, self.type, recommender.name, utility_value, time))


class ProviderCollection():
    def __init__(self):
        self.collection: dict[int, Provider] = {}
        self.types: dict[str, ProviderTypeConfig] = {}

    def setup(self, config: list[ProviderTypeConfig]):
        for type_config in config:
            self.types[type_config.name] = type_config

    def add_provider(self, provider: Provider):
        self.collection[provider.id] = provider

    def get_provider(self, provider_id: NonNegativeInt):
        return self.collection[provider_id]

    def __iter__(self):
        return iter(self.collection.values())
    
    def get_relevant_providers(self, item_list):
        """Get only providers that have items in the given item_list"""
        provider_ids = set()
        for item_id in item_list.ids():
            item = smores.Smores.state.items.get_item(item_id)
            provider_ids.add(item.provider_id)
        
        return [self.get_provider(provider_id) for provider_id in provider_ids]
    
    def update_utility_list(self, consumer, recommender, item_list, time):
        # Only process providers that have items in this recommendation list
        relevant_providers = self.get_relevant_providers(item_list)
        for provider in relevant_providers:
            provider.update_utility_list(consumer, recommender, item_list, time)

    def update_utility_item(self, consumer, recommender, item_id, time):
        item = smores.Smores.state.items.get_item(item_id)
        provider = self.get_provider(item.provider_id)
        provider.update_utility_item(consumer, recommender, item_id, time)
    
    def load_providers(self, provider_data_path: Path):
        with open(provider_data_path, 'r') as consumer_file:
            reader = DictReader(consumer_file)
            for row in reader:

                provider_config: ProviderInfo = ProviderInfo.model_validate(row)
                provider_type_config = self.types[provider_config.provider_type]

                provider = Provider()
                provider.setup(provider_type_config, provider_config)

                self.add_provider(provider)

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


