import numpy as np
from icecream import ic
from pathlib import Path
from collections import defaultdict

from lenskit.data.items import ItemList

from smores.stakeholders.consumer import ConsumerModelComponents, ConsumerCollection, Consumer, ItemSelectionModel
from smores.stakeholders.provider import ProviderModelComponents, ProviderCollection
from smores.item import ItemMap
from smores.recommender import Recommender, RecommenderMap
from smores.trigger import TriggerCollection, DayEvent, CycleEvent, SwitchEvent, InteractionBatchEvent
from smores.utils import SmoresConfig, SmoresLogger, ConsumerUtility, ProviderUtility

class Smores:

    class SmoresState:

        def __init__(self, config: SmoresConfig):
            self.config: SmoresConfig = config
            self.rand = np.random.default_rng(config.simulation.seed)

            self.cycle_count: int = 0
            self.cycle_limit: int = config.simulation.num_cycles

            self.day_count: int = 0
            self.day_limit: int = config.simulation.num_days

            self.slate_size: int  = config.simulation.slate_size

            # input
            self.data_directory: Path = Path(config.data.directory)
            self.consumer_file: Path = self.data_directory / config.data.consumer_file
            self.item_file: Path = self.data_directory / config.data.item_file
            self.provider_file: Path = self.data_directory / config.data.provider_file

            # output
            self.logger = SmoresLogger(config.output)

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
            self.triggers = TriggerCollection()

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

        # Setup log
        state.logger.setup(config.output)

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
        state.triggers.setup(config.triggers)

        # Connect consumers with initial recommenders
        self.setup_initial_recommenders()

        # Ignoring provider/recommender connections
        self.state.logger.info('Completed setup')
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
            self.state.logger.info(f'Started cycle {self.state.cycle_count}')
            self.run_cycle()
            self.state.logger.info(f'Completed cycle {self.state.cycle_count}')
            self.state.day_count = 0
            self.state.cycle_count += 1

    def run_cycle(self):
        self.train_recommenders()

        while self.state.day_count < self.state.day_limit:
            self.state.logger.info(f'  Started day {self.state.day_count}')
            self.run_day()
            self.state.logger.info(f'  Completed day {self.state.day_count}')
            self.state.day_count += 1
        self.cycle_actions()

    def train_recommenders(self):
        for rec_name in Smores.state.recommenders_active:
            recommender = Smores.state.recommenders_available.get_recommender(rec_name)
            recommender.train()
        self.state.logger.info(f'  Completed recommender training')

    # Interaction format: (consumer.id, selected_id, consumer.recommender.name, rating, Smores.state.current_time)
    def process_interactions(self, interactions: list):
        interaction_dict = defaultdict(list)
        for (user_id, item_id, rec_name, rating, time) in interactions:
            if rating is not None and item_id != ItemSelectionModel.NO_ITEM_SELECTED:
                interaction_dict[rec_name].append((user_id, int(item_id), rating, time))
        
        for rec_name in interaction_dict.keys():
            rec: Recommender = Recommender.name2recommender(rec_name)
            rec.update_dataset(interaction_dict[rec_name])

        # Run interaction triggers
        for trigger in Smores.state.triggers.get_iterator('interactions'):
            event = InteractionBatchEvent(interaction_dict)
            trigger.apply_trigger(event)
        

    def run_day(self):
        # Run consumer actions
        interactions = []
        for consumer in Smores.state.consumers:
            interaction = self.run_consumer_day(consumer)
            if interaction is not None:
                interactions.append(interaction)

        self.process_interactions(interactions)
        
        # Run day triggers
        for trigger in Smores.state.triggers.get_iterator('day'):
            event = DayEvent(self.state.day_count, self.state.current_time())
            trigger.apply_trigger(event)

    def run_consumer_day(self, consumer: Consumer):
        state = Smores.state
        time = state.current_time()

        # Get recommendations from associated recommender
        if consumer.recommender is not None:
            recs: ItemList = consumer.recommender.get_recommendations(consumer.id)
        else:
            raise RecommenderUnassignedException(consumer)
        
        # Update list utility for providers
        state.providers.update_utility_list(consumer, consumer.recommender, recs, time)

        # Apply item selection model
        if consumer.item_selection_model is not None and consumer is not None:
            result = consumer.item_selection_model.select_item(consumer, recs)
        else:
            raise ItemSelectionUnassignedException(consumer)
        
        # Update item utility for item provider
        if not ItemSelectionModel.is_empty_selection(result):
            (selected_id, score) = result
            state.providers.update_utility_item(consumer, consumer.recommender, selected_id, time)

        # Update recommender choice model
        if consumer.recommender_choice_model is not None:
            if ItemSelectionModel.is_empty_selection(result):
                selected_id = result[0]
            else:
                selected_id = ItemSelectionModel.NO_ITEM_SELECTED
            interaction_utility, recommender_utility = \
                consumer.recommender_choice_model.update_recommender_utility(consumer.recommender.name, time, selected_id, recs)
        else:
            raise RecommenderChoiceUnassignedException(consumer)
        
        # log the consumer utility
        Smores.state.logger.log_consumer(ConsumerUtility(consumer.id, consumer.recommender.name, interaction_utility,
                                                         recommender_utility))

        # construct the interaction and return
        if ItemSelectionModel.is_empty_selection(result):
            interaction = (consumer.id, None, consumer.recommender.name, None, time)
        else:
            interaction = (consumer.id, selected_id, consumer.recommender.name, 1, Smores.state.current_time())
        return interaction

    def cycle_actions(self):
        # Run consumer actions
        for consumer in iter(Smores.state.consumers):
            self.run_consumer_cycle(consumer)
        # Run cycle triggers
        for trigger in Smores.state.triggers.get_iterator('cycle'):
            event = CycleEvent(self.state.day_count, self.state.current_time())
            trigger.apply_trigger(event)

    def run_consumer_cycle(self, consumer: Consumer):
        # Apply recommender choice model
        if consumer.recommender_choice_model is not None:
            if consumer.recommender is not None:
                rec_name = consumer.recommender.name
                next_rec_name = consumer.recommender_choice_model.choose_recommender()
                if next_rec_name in Smores.state.recommenders_active:
                    consumer.recommender = Smores.state.recommenders_available.get_recommender(rec_name)
                                    
                    for trigger in Smores.state.triggers.get_iterator('switch'):
                        event = SwitchEvent(consumer, next_rec_name)
                        trigger.apply_trigger(event)
                else: 
                    raise RecommenderNotActiveException(consumer, next_rec_name)


    def cleanup(self):
        self.state.logger.cleanup()

# Exceptions
class InactiveInitialRecommenderException(Exception):
    def __init__(self, name):
        self.message = self.message = f'Cannot use recommender {name} as initial recommender. It is not active.'
        super().__init__(self.message)

class UnknownInitialRecommenderException(Exception):
    def __init__(self, name):
        self.message = self.message = f'Recommender {name} is unknown. Cannot be set as initial recommender.'
        super().__init__(self.message)

class RecommenderUnassignedException(Exception):
    def __init__(self, consumer):
        self.message = self.message = f'Consumer {consumer.id} has no assigned recommender.'
        super().__init__(self.message)

class ItemSelectionUnassignedException(Exception):
    def __init__(self, consumer):
        self.message = self.message = f'Consumer {consumer.id} has no assigned item selection model.'
        super().__init__(self.message)

class RecommenderChoiceUnassignedException(Exception):
    def __init__(self, consumer):
        self.message = self.message = f'Consumer {consumer.id} has no assigned recommender choice model.'
        super().__init__(self.message)

class RecommenderNotActiveException(Exception):
    def __init__(self, consumer, rec_name):
        self.message = self.message = f'Consumer {consumer.id} trying to switch to Recommender {rec_name}, which is not active.'
        super().__init__(self.message)
