import random
from icecream import ic
from pathlib import Path

from smores.stakeholders.consumer import ConsumerModelComponents, ConsumerCollection, Consumer
from smores.stakeholders.provider import ProviderModelComponents, ProviderCollection
from smores.item import ItemMap
from smores.recommender import RecommenderMap
from smores.trigger import TriggerCollection, DayEvent, CycleEvent
from smores.utils import SmoresConfig

class Smores:

    class SmoresState:

        def __init__(self, config: SmoresConfig):
            if config is not None:
                self.config: SmoresConfig = config
                self.rand = random.Random(config.simulation.seed)

            self.cycle_count: int = 0
            self.cycle_limit: int = config.simulation.num_cycles

            self.day_count: int = 0
            self.day_limit: int = config.simulation.num_days

            self.slate_size: int  = config.simulation.slate_size

            # paths
            self.data_directory: Path = Path(config.data.directory)
            self.consumer_file: Path = self.data_directory / config.data.consumer_file
            self.item_file: Path = self.data_directory / config.data.item_file
            self.provider_file: Path = self.data_directory / config.data.provider_file

            # init consumer collection
            self.consumer_models: ConsumerModelComponents = ConsumerModelComponents()
            self.consumers: ConsumerCollection = ConsumerCollection()
            # init provider collection
            self.provider_models: ProviderModelComponents = ProviderModelComponents()
            self.providers: ProviderCollection = ProviderCollection()
            # init item collection
            self.items: ItemMap = ItemMap()
            # init recommender collections
            self.recommenders_available = RecommenderMap()
            self.recommenders_active: list[str] = []
            self.initial_recommenders = config.recommender.initial

            # init trigger collections
            self.time_triggers = TriggerCollection()

        # Helper function
        # t = days in current cycle + number of cycles * days in cycle
        def current_time(self):
            return self.day_count + self.day_limit * self.cycle_count
        

    state: SmoresState = None

    def __init__(self, config):
        Smores.state = Smores.SmoresState(config)

    def setup(self):
        state = Smores.state
        config = state.config

        # Setup consumers
        state.consumer_models.setup(config.consumer.models)
        state.consumers.setup(config.consumer.types)
        state.consumers.load_consumers(state.consumer_file)

        # Setup providers
        state.provider_models.setup(config.provider.models)
        state.providers.setup(config.provider.types)
        state.providers.load_providers(state.provider_file)

        # Setup items
        state.items.load_items(state.item_file)

        # Setup recommenders
        state.recommenders_active = config.recommender.initial
        state.recommenders_available.setup(config.recommender.definitions)

        # Setup triggers
        state.time_triggers.setup(config.triggers)

        # Connect consumers with initial recommenders
        self.setup_initial_recommenders()

        # Ignoring provider/recommender connections
        return
    
    def setup_initial_recommenders(self):
        initial_rec_policy = Smores.state.config.consumer.initial_recommender
        if Smores.state.recommenders_available.is_recommender(initial_rec_policy):
            if initial_rec_policy in Smores.state.recommenders_active:
                initial_recommender = Smores.state.recommenders_available.get_recommender(initial_rec_policy)
                for consumer in Smores.state.consumers:
                    consumer.recommender = initial_recommender
            else:
                raise InactiveInitialRecommenderException(initial_rec_policy)
        else:
            raise UnknownInitialRecommenderException(initial_rec_policy)


    def run_experiment(self):
        self.setup()
        self.run_cycles()
        self.cleanup()

    def run_cycles(self):
        while self.state.cycle_count < self.state.cycle_limit:
            self.run_cycle()
            self.state.cycle_count += 1

    def run_cycle(self):
        while self.state.day_count < self.state.day_limit:
            self.run_day()
            self.state.day_count += 1
        self.cycle_actions()
        self.state.day_count = 0

    def run_day(self):
        # Run consumer actions
        for consumer in Smores.state.consumers:
            self.run_consumer_day(consumer)
        
        # Run day triggers
        for trigger in Smores.state.time_triggers.get_iterator('day'):
            event = DayEvent(self.state.day_count, self.state.current_time())
            trigger.apply_trigger(event)

    def run_consumer_day(self, consumer: Consumer):
        # Get recommendations from associated recommender
        # Apply list utility for associated provider
        # Apply item selection model
        # Apply item utility for associated provider
        # Update recommender database with interaction
        # Update recommender choice model
        pass

    def cycle_actions(self):
        # Run consumer actions
        for consumer in iter(Smores.state.consumers):
            self.run_consumer_cycle(consumer)
        # Run day triggers
        for trigger in Smores.state.time_triggers.get_iterator('cycle'):
            event = CycleEvent(self.state.day_count, self.state.current_time())
            trigger.apply_trigger(event)

    def run_consumer_cycle(consumer: Consumer):
        # Apply recommender choice model
        pass


    def cleanup(self):
        # Save files, etc.
        return

# Exceptions
class InactiveInitialRecommenderException(Exception):
    def __init__(self, name):
        self.message = self.message = f'Cannot use recommender {name} as initial recommender. It is not active.'
        super().__init__(self.message)

class UnknownInitialRecommenderException(Exception):
    def __init__(self, name):
        self.message = self.message = f'Recommender {name} is unknown. Cannot be set as initial recommender.'
        super().__init__(self.message)

