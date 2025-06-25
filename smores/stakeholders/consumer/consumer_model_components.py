from typing import Dict

from .consumer_utility_model import ConsumerUtilityModel, ConsumerUtilityModelFactory
from .item_selection_model import ItemSelectionModel, ItemSelectionModelFactory

from smores.utils import ConsumerModelsConfig

class ConsumerModelComponents:
    def __init__(self):
        self.utility_models: Dict[str | None, ConsumerUtilityModel] = {}
        self.item_selection_models: Dict[str | None, ItemSelectionModel] = {}

    def setup(self, config: ConsumerModelsConfig):
        for class_config in config.utility:
            um_object = ConsumerUtilityModelFactory.create(class_config.class_name)
            um_object.setup(class_config.params)
            self.utility_models[class_config.name] = um_object
            
        for class_config in config.item_selection:
            ism_object = ItemSelectionModelFactory.create(class_config.class_name)
            ism_object.setup(class_config.params)
            self.item_selection_models[class_config.name] = ism_object

    def get_utility_model(self, name: str):
        return self.utility_models[name]
    
    def get_item_selection_model(self, name: str):
        return self.item_selection_models[name]
    
                