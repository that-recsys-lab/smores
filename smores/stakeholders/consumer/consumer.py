from icecream import ic
from pathlib import Path
from csv import DictReader
from json import loads
from pydantic import BaseModel, PositiveInt

from .consumer_utility_model import ConsumerUtilityModelFactory
from .item_selection_model import ItemSelectionModelFactory
from .recommender_choice_model import RecommenderChoiceModelFactory
from smores.utils import UtilityHistory, ConsumerTypeConfig
import smores

class ConsumerInfo (BaseModel):
    consumer_id: PositiveInt
    consumer_type: str
    preferences: list[float]

class Consumer:
    def __init__(self):
        self.id = -1
        self.type = None
        self.preference_vector = None
        self.recommender: str = None
        self.utility_model = None
        self.item_selection_model = None
        self.recommender_choice_model = None
        self.history: UtilityHistory = None

    def __str__(self):
        return f'<Consumer {self.id} {self.type}>'

    def setup(self, config_type: ConsumerTypeConfig, config_instance: ConsumerInfo):
        # Instance-specific
        self.id = config_instance.consumer_id
        self.type = config_instance.consumer_type
        self.preference_vector = config_instance.preferences

        # Consumer-type specific
        self.history = UtilityHistory()
        consumer_models = smores.Smores.state.consumer_models

        # Get utility model
        utility_model_name = config_type.utility_model
        self.utility_model = consumer_models.get_utility_model(utility_model_name)

        # Get selection model
        selection_model_name = config_type.item_selection_model
        self.item_selection_model = consumer_models.get_item_selection_model(selection_model_name)

        # Setup recommender choice model
        choice_model_config = config_type.recommender_choice_model
        self.recommender_choice_model = RecommenderChoiceModelFactory.create(choice_model_config.class_name)
        self.recommender_choice_model.setup(choice_model_config)
        self.recommender_choice_model.set_consumer(self)


class ConsumerCollection():
    def __init__(self):
        self.collection: dict[int, Consumer] = {}
        self.types: dict[str, ConsumerTypeConfig] = {}

    def setup(self, config: list[ConsumerTypeConfig]):
        for type_config in config:
            self.types[type_config.name] = type_config

    def add_consumer(self, consumer: Consumer):
        self.collection[consumer.id] = consumer

    def get_consumer(self, consumer_id):
        return self.collection[consumer_id]

    def __iter__(self):
        return iter(self.collection.values())
    
    def load_consumers(self, consumer_data_path: Path):
        with open(consumer_data_path, 'r') as consumer_file:
            reader = DictReader(consumer_file)
            for row in reader:
                feature_list_str = row['preferences']
                feature_list = loads(feature_list_str)
                row['preferences'] = feature_list
                consumer_config: ConsumerInfo = ConsumerInfo.model_validate(row)
                consumer_type_config = self.types[consumer_config.consumer_type]

                consumer = Consumer()
                consumer.setup(consumer_type_config, consumer_config)

                self.add_consumer(consumer)


class ConsumerFactory():
    """
    The ConsumerFactory associates consumer types with class names and allows appropriate instances
    to be created. A consumer class must registered in the factory before it can be
    created.
    """

    _class_name_map = {}

    @classmethod
    def register(cls, type_name, consumer_class):
        if not issubclass(consumer_class, Consumer):
            raise InvalidConsumerError(type_name)
        cls._class_name_map[type_name] = consumer_class

    @classmethod
    def register_all(cls, type_specs):
        for type_name, consumer_class in type_specs:
            cls.register(type_name, consumer_class)

    @classmethod
    def make_object(cls, type_name):
        consumer_class = cls._class_name_map.get(type_name)
        if consumer_class is None:
            raise UnregisteredConsumerError(type_name)
        return consumer_class()



# Exceptions
class InvalidConsumerError(Exception):
    def __init__(self, name):
        self.message = self.message = f'Cannot create Consumer object: Class {name} is not a subclass of Provider.'
        super().__init__(self.message)


class UnregisteredConsumerError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create Consumer object: Class {name} is not registered and may not exist.'
        super().__init__(self.message)


