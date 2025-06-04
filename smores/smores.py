import random
from icecream import ic

from smores.stakeholders.consumer import ConsumerModelComponents
from smores.recommender import RecommenderMap
from smores.trigger import TriggerCollection
from smores.utils import SmoresConfig

class Smores:

    class SmoresState:

        def __init__(self, config: SmoresConfig):
            if config is not None:
                self.config = config
                self.rand = random.Random(config.simulation.seed)

            self.cycle_count = 0
            self.cycle_limit = config.simulation.num_cycles

            self.day_count = 0
            self.day_limit = config.simulation.num_days

            self.slate_size  = config.simulation.slate_size

            # init consumer collection
            self.consumer_models = ConsumerModelComponents()
            # init provider collection
            # init item collection
            # init recommender collections
            self.recommenders_available = RecommenderMap()
            self.recommenders_active = RecommenderMap()
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
        # Setup providers
        # Setup items
        # Setup recommenders
        state.recommenders_available.setup(config.recommender.definitions)
        # Setup triggers
        state.time_triggers.setup(config.triggers)
        return

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
        # Run user actions
        # Run provider actions
        # Run platform actions
        self.day_actions()

    def cycle_actions(self):
        # Whatever happens at the end of a cycle
        # Run user choice actions
        return

    def day_actions(self):
        # Whatever happens at the end of a day
        return

    def cleanup(self):
        # Save files, etc.
        return

