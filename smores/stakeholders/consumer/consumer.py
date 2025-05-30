from icecream import ic

from smores.stakeholders.consumer import ConsumerUtilityModelLookup, ItemSelectionModelLookup, RecommenderChoiceModelFactory
from smores.utils import UtilityHistory

class Consumer:

    _id: int = -1

    @classmethod
    def next_id(cls) -> int:
        cls._id += 1
        return cls._id

    def __init__(self):
        self.id = None
        self.preference_vector = None
        self.recommender = None
        self.utility_model = None
        self.item_selection_model = None
        self.recommender_choice_model = None
        self.history = None

    def __str__(self):
        return f'<Consumer {self.id}>'

    def setup(self, config):
        self.id = Consumer.next_id()

        self.history = UtilityHistory()

        utility_model_config = config.consumer.utility_model
        ic(utility_model_config)
        self.utility_model = ConsumerUtilityModelLookup.get_class(utility_model_config.class_name)
        self.utility_model.setup(utility_model_config)

        # Set up selection model
        selection_model_config = config.consumer.item_selection_model
        ic(selection_model_config)
        self.item_selection_model = ItemSelectionModelLookup.get_class(selection_model_config.class_name)
        self.item_selection_model.setup(selection_model_config)

        choice_model_config = config.consumer.recommender_choice_model
        self.recommender_choice_model = RecommenderChoiceModelFactory.create(choice_model_config.class_name)
        self.recommender_choice_model.setup(choice_model_config)


class ConsumerCollection():
    def __init__(self):
        self.collection = []

    def add_consumer(self, cons: Consumer):
        self.collection.append(cons)

    def __iter__(self):
        return self.collection.__iter__()


class ConsumerFactory():
    """
    The ConsumerFactory associates consumer types with class names and allows appropriate instances
    to be created. A consumer class must registered in the factory before it can be
    created.
    """

    _class_name_map = {}

    @classmethod
    def register(cls, type_name, consumer_class):
        if not issubclass(consumer_class, Provider):
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


