import numpy as np
from icecream import ic
from pathlib import Path
from collections import defaultdict, Counter

from lenskit.data.items import ItemList

from smores.stakeholders.consumer import ConsumerModelComponents, ConsumerCollection, Consumer, ItemSelectionModel
from smores.stakeholders.provider import ProviderModelComponents, ProviderCollection
from smores.item import ItemMap
from smores.recommender import Recommender, RecommenderMap
from smores.trigger import TriggerCollection, DayEvent, CycleEvent, InteractionBatchEvent
from smores.trigger.switch_trigger import SwitchEvent
from smores.trigger.trigger_tester import TriggerTester
from smores.utils import SmoresConfig, SmoresLogger, ConsumerUtility, ProviderUtility, UserJourney, SummaryLogger

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
            self.summary_logger = SummaryLogger(enabled=config.summary_logger.enabled)

            # init consumer collection
            self.consumer_models: ConsumerModelComponents = ConsumerModelComponents()
            self.consumers: ConsumerCollection = ConsumerCollection()
            # init provider collection
            self.provider_models: ProviderModelComponents = ProviderModelComponents()
            self.providers: ProviderCollection = ProviderCollection()
            # init item collection
            self.items: ItemMap = ItemMap()
            # init recommender collections
            self.recommenders_base = RecommenderMap()
            self.recommenders_fallback = RecommenderMap()
            self.recommenders_active: list[str] = []
            self.initial_recommenders = config.recommender.initial
            # per-cycle per-user clicked blocklist to avoid repeat exposure
            self.cycle_clicked_blocklist: dict[int, set[int]] = defaultdict(set)
            # per-cycle recommender metrics
            self.recommender_metrics: defaultdict[str, dict[str, float]] = defaultdict(
                Smores.SmoresState.default_rec_metrics
            )
            # track user assignment sets for churn / growth calculations
            self.prev_recommender_users: defaultdict[str, set[int]] = defaultdict(set)
            # track trigger health for the current cycle
            self.trigger_success: bool = True
            # optional trigger tester
            self.trigger_tester = None

            # init trigger collections
            self.triggers = TriggerCollection()

        # Helper function
        # t = days in current cycle + number of cycles * days in cycle
        def current_time(self):
            return self.day_count + self.day_limit * self.cycle_count

        @staticmethod
        def default_rec_metrics():
            return {
                "recommendations": 0,
                "fallback_used": 0,
                "sampled_items": 0,
                "slate_items_total": 0,
                "unique_items": set(),
                "new_interactions": 0,
                "deleted_interactions": 0,
                "clicks": 0,
            }
        

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
        state.summary_logger.set_catalog_size(len(state.items.all_items()))

        # Setup recommenders
        state.recommenders_active = config.recommender.initial
        state.recommenders_base.setup(config.recommender.base_recommenders)
        state.recommenders_fallback.setup(config.recommender.fallback_recommenders)

        # Setup dataset. Has to be a separate step so the fallback can point to the base dataset
        state.recommenders_base.setup_datasets()
        state.recommenders_base.setup_fallbacks()

        # Setup triggers
        if config.triggers is not None:
            state.triggers.setup(config.triggers)
        # Setup trigger tester (optional)
        tt_cfg = getattr(config, "trigger_tester", None)
        if tt_cfg is not None and getattr(tt_cfg, "enabled", False):
            state.trigger_tester = TriggerTester(state, tt_cfg)

        # Connect consumers with initial recommenders
        self.setup_initial_recommenders()

        # Setup log. Must be last so that it can use the set up information
        state.logger.setup(config.output)

        # Ignoring provider/recommender connections
        self.state.logger.info('Completed setup')
        return
    
    def setup_initial_recommenders(self):
        initial_rec_policy = Smores.state.config.consumer.initial_recommender
        if Smores.state.recommenders_base.is_recommender(initial_rec_policy):
            if initial_rec_policy in Smores.state.recommenders_active:
                initial_recommender = Smores.state.recommenders_base.get_recommender(initial_rec_policy)
                for consumer in Smores.state.consumers:
                    consumer.recommender = initial_recommender
                    consumer.recommender_choice_model
            else:
                raise InactiveInitialRecommenderException(initial_rec_policy)
        else:
            raise UnknownInitialRecommenderException(initial_rec_policy)


    def run_experiment(self):
        import os, sys; print(f"[py-spy] PID: {os.getpid()}", file=sys.stderr, flush=True)
        self.setup()
        cfg = self.state.config
        scenario = getattr(getattr(cfg, "trigger_tester", None), "scenario", None) or "n/a"
        data_dir = getattr(getattr(cfg, "data", None), "directory", None) or "n/a"
        self.state.logger.info(f"Experiment start: scenario={scenario}, data_dir={data_dir}, time={self.state.current_time()}")
        self.state.summary_logger.start_timer()
        self.run_cycles()
        self.cleanup()

    def run_cycles(self):
        while self.state.cycle_count < self.state.cycle_limit:
            display_cycle = self.state.cycle_count + 1
            self.state.trigger_success = True
            self.state.logger.info(f'Started cycle {display_cycle}')
            self.run_cycle()
            self._log_cycle_stats(display_cycle)
            self.state.day_count = 0
            self.state.cycle_count += 1

    def run_cycle(self):
        # Reset per-cycle clicked blocklist
        self.state.cycle_clicked_blocklist.clear()
        # Reset trigger health flag
        self.state.trigger_success = True
        self.state.logger.debug(f'    Active recommenders are: {self.state.recommenders_active}')
        self.train_recommenders()

        while self.state.day_count < self.state.day_limit:
            display_cycle = self.state.cycle_count + 1
            display_day = self.state.day_count + 1
            self.state.logger.info(f'  Day {display_cycle}:{display_day}')
            self.run_day()
            self.state.day_count += 1
        display_cycle = self.state.cycle_count + 1
        self.state.logger.info(f'  Finished cycle {display_cycle}')
        self.cycle_actions()
        if self.state.trigger_tester is not None:
            tester_ok = self.state.trigger_tester.run_cycle_checks()
            self.state.trigger_success = self.state.trigger_success and tester_ok

    def train_recommenders(self):
        for rec_name in Smores.state.recommenders_active:
            recommender = Smores.state.recommenders_base.get_recommender(rec_name)
            recommender.train()
        self.state.logger.info(f'  Completed recommender training')

    # Interaction format: (consumer.id, selected_id, consumer.recommender.name, rating, Smores.state.current_time)
    def process_interactions(self, interactions: list):
        interaction_dict = defaultdict(list)
        for (user_id, item_id, rec_name, rating, time) in interactions:
            if rating is not None and item_id != None:
                interaction_dict[rec_name].append((user_id, int(item_id), rating, time))
        
        for rec_name in interaction_dict.keys():
            rec_metrics = self.state.recommender_metrics[rec_name]
            rec_metrics["new_interactions"] += len(interaction_dict[rec_name])
            rec: Recommender = Recommender.name2base_recommender(rec_name)
            rec.update_dataset(interaction_dict[rec_name])

        # Run interaction triggers
        if self.state.trigger_tester is not None:
            self.state.trigger_tester.set_baseline()
        for trigger in Smores.state.triggers.get_iterator('interactions'):
            event = InteractionBatchEvent(interaction_dict)
            self._apply_trigger_safe(trigger, event)
        if self.state.trigger_tester is not None:
            tester_ok = self.state.trigger_tester.run_cycle_checks()
            self.state.trigger_success = self.state.trigger_success and tester_ok
        

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
            self._apply_trigger_safe(trigger, event)


    def _log_cycle_stats(self, display_cycle: int) -> None:
        state = Smores.state
        include_trigger_tester = state.trigger_tester is not None
        assignment_counts: Counter[str] = Counter()
        assignment_users: defaultdict[str, set[int]] = defaultdict(set)
        for consumer in state.consumers:
            assigned_rec = getattr(consumer, 'recommender', None)
            if assigned_rec is not None and getattr(assigned_rec, 'name', None):
                rec_name = assigned_rec.name
                assignment_counts[rec_name] += 1
                assignment_users[rec_name].add(consumer.id)

        for rec_name in state.recommenders_active:
            rec = state.recommenders_base.get_recommender(rec_name)
            if rec is None:
                continue
            dataset = rec.get_dataset()
            if dataset is None:
                continue
            interactions = dataset.interaction_count
            user_count = getattr(dataset, "user_count", 0)
            item_count = getattr(dataset, "item_count", 0)
            avg_profile = interactions / user_count if user_count else 0
            assigned_users = assignment_counts.get(rec_name, 0)
            metrics = state.recommender_metrics[rec_name]
            rec_requests = metrics.get("recommendations", 0)
            rec_fallbacks = metrics.get("fallback_used", 0)
            sampled_items = metrics.get("sampled_items", 0)
            avg_sampled = (sampled_items / rec_requests) if rec_requests else 0
            delivered_slate = metrics.get("slate_items_total", 0)
            avg_slate_size = (delivered_slate / rec_requests) if rec_requests else 0
            unique_item_count = len(metrics.get("unique_items", set()))
            coverage_pct = (unique_item_count / item_count * 100) if item_count else 0
            new_interactions = metrics.get("new_interactions", 0)
            deleted_interactions = metrics.get("deleted_interactions", 0)
            clicks = metrics.get("clicks", 0)
            ctr = (clicks / rec_requests) if rec_requests else 0
            current_users = assignment_users.get(rec_name, set())
            prev_users = state.prev_recommender_users.get(rec_name, set())
            new_users = current_users - prev_users
            churned_users = prev_users - current_users
            state.prev_recommender_users[rec_name] = set(current_users)
            cycle_row = {
                "cycle": display_cycle,
                "recommender": rec_name,
                "interactions": interactions,
                "dataset_users": user_count,
                "dataset_items": item_count,
                "avg_profile_len": round(avg_profile, 2),
                "active_users": assigned_users,
                "new_users": len(new_users),
                "churned_users": len(churned_users),
                "new_interactions": new_interactions,
                "deleted_interactions": deleted_interactions,
                "avg_slate_size": round(avg_slate_size, 2),
                "unique_items": unique_item_count,
                "coverage_pct": round(coverage_pct, 1),
                "rec_requests": rec_requests,
                "fallback_used": rec_fallbacks,
                "avg_sampled": round(avg_sampled, 2),
                "ctr": round(ctr, 2),
            }
            if include_trigger_tester:
                cycle_row["trigger_success"] = state.trigger_success
            log_msg = (
                f"  Cycle {display_cycle}: {rec_name} interactions={interactions}, "
                f"dataset_users={user_count}, dataset_items={item_count}, "
                f"avg_profile_len={avg_profile:.2f}, active_users={assigned_users}, "
                f"new_users={len(new_users)}, churned_users={len(churned_users)}, "
                f"new_interactions={new_interactions}, deleted_interactions={deleted_interactions}, "
                f"avg_slate_size={avg_slate_size:.2f}, unique_items={unique_item_count} ({coverage_pct:.1f}%), "
                f"fallback_used={rec_fallbacks}/{rec_requests}, avg_sampled={avg_sampled:.2f}, ctr={ctr:.2f}"
            )
            if include_trigger_tester:
                note = getattr(state.trigger_tester, "last_result_msg", None)
                log_msg += f", trigger_success={state.trigger_success}"
                if note:
                    log_msg += f" ({note})"
            state.logger.info(log_msg)
            state.logger.log_cycle_metrics(cycle_row)
            state.summary_logger.set_active_users(rec_name, assigned_users)
            # reset per-cycle metrics
            state.recommender_metrics[rec_name] = state.default_rec_metrics()

    def run_consumer_day(self, consumer: Consumer):
        state = Smores.state
        # state.logger.debug(f'     Processing user {consumer.id}')
        # state.logger.debug(f'        Recommender: {consumer.recommender.name}')
        time = state.current_time()

        # Get recommendations from associated recommender
        if consumer.recommender is not None:
            recs: ItemList = consumer.recommender.get_recommendations(consumer.id)
        else:
            raise RecommenderUnassignedException(consumer)

        slate_items = [int(item_id) for item_id in recs.ids()]
        consumer.seen_items.update(slate_items)
        for item_id in slate_items:
            state.logger.log_item_appear(item_id, consumer.recommender.name, state.cycle_count)

        state.summary_logger.log_recommendation(
            consumer.recommender.name,
            consumer.id,
            slate_items,
            state.slate_size,
            sampled_count=getattr(consumer.recommender, '_last_sampled_count', 0),
            used_fallback=getattr(consumer.recommender, '_last_used_fallback', False),
        )
        # Track per-cycle recommender metrics
        rec_metrics = state.recommender_metrics[consumer.recommender.name]
        rec_metrics["recommendations"] += 1
        rec_metrics["fallback_used"] += 1 if getattr(consumer.recommender, "_last_used_fallback", False) else 0
        rec_metrics["sampled_items"] += getattr(consumer.recommender, "_last_sampled_count", 0)
        rec_metrics["slate_items_total"] += len(slate_items)
        rec_metrics["unique_items"].update(slate_items)

        # Update list utility for providers
        state.providers.update_utility_list(consumer, consumer.recommender, recs, time)

        # Apply item selection model
        if consumer.item_selection_model is not None and consumer is not None:
            result = consumer.item_selection_model.select_item(consumer, recs)
        else:
            raise ItemSelectionUnassignedException(consumer)

        interaction_utility = 0.0
        recommender_utility = 0.0
        
        if not ItemSelectionModel.is_empty_selection(result):
            selected_id = int(result[0])
            # Add to clicked items
            consumer.clicked_items.add(selected_id)
            state.logger.log_item_click(selected_id, consumer.recommender.name, state.cycle_count)
            # Update item utility for item provider
            state.providers.update_utility_item(consumer, consumer.recommender, selected_id, time)
            state.cycle_clicked_blocklist[consumer.id].add(selected_id)
        else:
            selected_id = None

        if selected_id is not None:
            rec_metrics["clicks"] += 1

        consumer.recommender.update_feedback(consumer.id, slate_items, selected_id)
        state.summary_logger.log_click(consumer.recommender.name, selected_id is not None)

        # Update recommender choice model
        if consumer.recommender_choice_model is not None:
            interaction_utility = consumer.utility_model.compute_list_utility(consumer, recs)
            recommender_utility = consumer.recommender_choice_model.update_recommender_utility(interaction_utility)
        else:
            raise RecommenderChoiceUnassignedException(consumer)
        
        # log the consumer utility
        Smores.state.logger.log_consumer(ConsumerUtility(consumer.id, consumer.type, consumer.recommender.name, interaction_utility,
                                                         recommender_utility, time))

        if state.logger.is_sampled_user(consumer.id):
            slate_utilities = []
            slate_scores_raw = recs.scores()
            if slate_scores_raw is not None:
                slate_scores = [float(score) for score in slate_scores_raw]
            else:
                slate_scores = [None] * len(slate_items)

            for item_id in slate_items:
                item = state.items.get_item(int(item_id))
                if consumer.preference_vector is not None:
                    slate_utilities.append(float(np.dot(item.features, consumer.preference_vector)))
                else:
                    slate_utilities.append(0.0)

            max_utility = max(slate_utilities) if slate_utilities else None

            journey_info = UserJourney(
                user_id=consumer.id,
                cycle=state.cycle_count,
                day=state.day_count,
                time=time,
                recommender=consumer.recommender.name,
                slate_size=len(slate_items),
                slate_items=str(slate_items),
                slate_scores=str([round(s, 4) if s is not None else None for s in slate_scores]),
                slate_utilities=str([round(u, 4) for u in slate_utilities]),
                selected_item=selected_id,
                selected_rank=slate_items.index(selected_id) + 1 if selected_id is not None and selected_id in slate_items else None,
                selected_utility=None,
                max_utility=round(max_utility, 4) if max_utility is not None else None,
                num_unique_items_clicked=len(consumer.clicked_items),
                num_unique_items_seen=len(consumer.seen_items),
            )
            state.logger.log_user_journey(journey_info)

        # construct the interaction and return
        interaction = (consumer.id, selected_id, consumer.recommender.name, 1, time)

        return interaction

    def cycle_actions(self):
        # Run consumer actions
        for consumer in iter(Smores.state.consumers):
            self.run_consumer_cycle(consumer)
        # Run cycle triggers
        for trigger in Smores.state.triggers.get_iterator('cycle'):
            event = CycleEvent(self.state.cycle_count, self.state.current_time())
            # Smores.state.logger.debug(f'checking trigger {trigger}. cycle count: {event.cycle_count}')
            self._apply_trigger_safe(trigger, event)

    def run_consumer_cycle(self, consumer: Consumer):
        # Apply recommender choice model
        if consumer.recommender_choice_model is not None:
            if consumer.recommender is not None:
                rec_name = consumer.recommender.name
                next_rec_name = consumer.recommender_choice_model.choose_recommender()
                if next_rec_name != rec_name:
                    if next_rec_name in Smores.state.recommenders_active:
                        consumer.recommender = Smores.state.recommenders_base.get_recommender(next_rec_name)
                                    
                        for trigger in Smores.state.triggers.get_iterator('switch'):
                            event = SwitchEvent(consumer, rec_name, next_rec_name)
                            self._apply_trigger_safe(trigger, event)
                    else: 
                        raise RecommenderNotActiveException(consumer, next_rec_name)
                # Else no change to the recommender

    def cleanup(self):
        self.state.summary_logger.print_summary()
        self.state.logger.cleanup()

    def _apply_trigger_safe(self, trigger, event):
        """Run a trigger and mark the cycle's trigger health on failure without stopping the sim."""
        try:
            trigger.apply_trigger(event)
        except Exception as exc:
            self.state.trigger_success = False
            trigger_name = getattr(trigger, 'name', type(trigger).__name__)
            self.state.logger.error(f"Trigger '{trigger_name}' failed: {exc}")


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
