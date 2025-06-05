from typing import Dict

from .provider_utility_model import ProviderUtilityModel, ProviderUtilityModelFactory

from smores.utils import ProviderModelsConfig

class ProviderModelComponents:
    def __init__(self):
        self.utility_models: Dict[str | None, ProviderUtilityModel] = {}

    def setup(self, config: ProviderModelsConfig):
        for class_config in config.utility:
            um_object = ProviderUtilityModelFactory.create(class_config.class_name)
            um_object.setup(class_config.params)
            self.utility_models[class_config.name] = um_object
            

    def get_utility_model(self, name: str):
        return self.utility_models[name]
    
    
     