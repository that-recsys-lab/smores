from abc import ABC, abstractmethod
from collections import defaultdict
from icecream import ic
import pyarrow as pa

from lenskit.data import Dataset, DatasetBuilder
from lenskit.data import ItemList

import smores
from smores.utils import InteractionHistory, PythonClassConfig


class Recommender(ABC):
    def __init__(self):
        self.dataset: Dataset = None
        # Use this if there isn't enough data overall
        self.cold_start_fallback: str = None
        # Use this if there isn't enough data for a particular user
        self.cold_user_fallback: str = None
        self.name = None
        self.trained = False
        self.parent = None
        self._cached_user_count = None
        # Optional item sampling support
        self.item_sampler = None
        self.sampled_item_count = 0
        self.candidate_multiplier: int = 1
        # Optional recency blocking configuration
        self.recency_threshold: int = 0
        self.cooldown_cycles: int = 0
        self._no_click_streaks: defaultdict[int, defaultdict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._cooldown_until_cycle: defaultdict[int, dict[int, int]] = defaultdict(dict)
        self._last_sampled_count: int = 0
        self._last_used_fallback: bool = False

    @abstractmethod
    def setup(self, config):
        if type(config) is PythonClassConfig:
            params = config.params or {}
            if 'cold_start_fallback' in params:
                self.cold_start_fallback = params['cold_start_fallback']
            if 'cold_user_fallback' in params:
                self.cold_user_fallback = params['cold_user_fallback']

            sampler_cfg = params.get('item_sampler')
            if sampler_cfg:
                sampler_params = sampler_cfg.get('params', {}) or {}
                sampler_class = sampler_cfg.get('class_name', 'rejection_sampler')

                file_name = sampler_params.get('file_name')
                if not file_name:
                    raise ValueError(
                        f"Recommender '{config.name}': item_sampler '{sampler_class}' requires params.file_name"
                    )

                file_path = smores.Smores.state.data_directory / file_name
                self.item_sampler = self._create_item_sampler(sampler_class)
                self.item_sampler.load_from_file(file_path)
                self.sampled_item_count = int(sampler_params.get('sampled_item_count', 0))
            self.candidate_multiplier = int(params.get('candidate_multiplier', 1) or 1)
            if self.candidate_multiplier < 1:
                self.candidate_multiplier = 1
            self.recency_threshold = int(params.get('recency_threshold', 0) or 0)
            self.cooldown_cycles = int(params.get('cooldown_cycles', 0) or 0)
        self.name = config.name
    
    def setup_dataset(self):
        builder = DatasetBuilder(None)
        builder.add_entity_class('user_id')
        builder.add_entities('user_id', smores.Smores.state.consumers.get_consumer_ids())
        builder.add_entity_class('item_id')
        builder.add_entities('item_id', smores.Smores.state.items.all_items())
        builder.add_relationship_class('interaction', ['user_id', 'item_id'], interaction=True)
        self.dataset = builder.build()

    def get_dataset(self):
        if self.dataset is None:
            return self.parent.get_dataset()
        return self.dataset

    def set_dataset(self, dataset: Dataset):
        if self.dataset is None:
            self.parent.set_dataset(dataset)
        else:
            self.dataset = dataset

    def dataset_active_users(self):
        if self._cached_user_count is None:
            interactions: pa.Table = self.get_dataset().interaction_table(format='arrow', original_ids=True)
            user_col = interactions.column('user_id')
            unique_users = user_col.unique()
            self._cached_user_count = len(unique_users)
        return self._cached_user_count

    @classmethod
    def name2base_recommender(cls, name: str):
        return smores.Smores.state.recommenders_base.get_recommender(name)        

    @abstractmethod
    def train(self):
        cold_start_rec = self.get_cold_start_fallback()
        cold_user_rec = self.get_cold_user_fallback()

        if cold_start_rec is not None:
            cold_start_rec.train()
        if cold_user_rec is not None:
            cold_user_rec.train()
        
        # Invalidate user count cache after training
        self._cached_user_count = None

    def get_cold_start_fallback(self):
        if self.cold_start_fallback is not None:
            cold_start_rec = smores.Smores.state.recommenders_fallback.get_recommender(
                self.cold_start_fallback)
            return cold_start_rec
        else:
            return None
        
    def get_cold_user_fallback(self):
        if self.cold_user_fallback is not None:
            cold_user_rec = smores.Smores.state.recommenders_fallback.get_recommender(
                self.cold_user_fallback)
            return cold_user_rec
        else:
            return None
    
    def _create_item_sampler(self, class_name: str):
        if class_name == 'rejection_sampler':
            from smores.samplers.rejection_sampler import RejectionSampler

            return RejectionSampler()

        raise ValueError(
            f"Recommender '{self.name}': unsupported item sampler '{class_name}'"
        )

    @abstractmethod
    def isDatasetViable(self):
        pass

    @abstractmethod
    def isProfileViable(self, user_id):
        pass

    @abstractmethod
    def get_recommendations(self, user_id) -> ItemList:
        pass

    def update_dataset(self, interaction_list: list):
        hist = InteractionHistory()
        hist.add_interactions(interaction_list)
        self.set_dataset(hist.to_dataset(self.get_dataset()))

    def update_dataset_itemlist(self, consumer_id, interaction_list: ItemList):
        if len(interaction_list) > 0:
            hist = InteractionHistory()
            hist.add_interactions_itemlist(consumer_id, interaction_list)
            self.set_dataset(hist.to_dataset(self.get_dataset()))
    
    def get_user(self, user_id) -> ItemList | None:
        return self.get_dataset().user_row(user_id)
    
    def delete_user(self, user_id):
        user_data = self.get_user(user_id)
        removed = user_data.ids().size if user_data is not None else 0
        builder = DatasetBuilder(self.get_dataset())
        builder.filter_interactions('interaction', remove={'user_id': [user_id]})
        self.set_dataset(builder.build())
        if removed and getattr(smores.Smores, "state", None) is not None:
            smores.Smores.state.recommender_metrics[self.name]["deleted_interactions"] += int(removed)

    def _candidate_request_count(self) -> int:
        """Return the number of candidates to request from the core recommender."""

        slate_size = int(getattr(smores.Smores.state, 'slate_size', 0) or 0)
        if slate_size <= 0:
            return 1

        multiplier = max(int(self.candidate_multiplier or 1), 1)
        requested = slate_size * multiplier
        return max(int(requested), slate_size, 1)

    def apply_item_sampling(self, user_id, recommendations: ItemList) -> ItemList:
        """Append sampled items for diversity if configured."""

        filtered_recs = self._filter_blocked_items(user_id, recommendations)
        self._last_sampled_count = 0
        self._last_used_fallback = False

        ids_array = filtered_recs.ids()
        scores_array = filtered_recs.scores()
        ranks_array = filtered_recs.ranks()

        ids = [int(item_id) for item_id in ids_array] if ids_array is not None else []
        scores = [float(score) for score in scores_array] if scores_array is not None else None
        ranks = [int(rank) for rank in ranks_array] if ranks_array is not None else None

        slate_size = int(getattr(smores.Smores.state, 'slate_size', 0) or 0)
        if slate_size <= 0:
            return ItemList(None, item_ids=[], scores=None, rank=None)

        sampler_slots = min(max(int(self.sampled_item_count or 0), 0), slate_size)
        desired_core = max(slate_size + sampler_slots, slate_size)

        core_ids, core_scores, core_ranks = self._select_core_items(user_id, ids, scores, ranks, desired_core)

        core_ids = list(core_ids)
        core_scores = list(core_scores) if core_scores is not None else None
        core_ranks = list(core_ranks) if core_ranks is not None else None

        take_from_core = min(len(core_ids), slate_size)
        final_ids = list(core_ids[:take_from_core])
        core_boundary = len(final_ids)

        final_scores = None
        if core_scores is not None:
            final_scores = list(core_scores[:core_boundary])
        if core_ranks is not None:
            final_ranks = list(core_ranks[:core_boundary])
        else:
            final_ranks = None

        if final_ranks is None and final_ids:
            final_ranks = list(range(1, len(final_ids) + 1))

        sampler_total = 0
        prior = self.get_user(user_id)
        exclusions = set(final_ids)
        if prior is not None:
            exclusions.update(prior.ids())
        blocked = self._current_blocked_items(user_id)
        exclusions.update(blocked)

        # Fill any remaining capacity with sampler items first.
        remaining_capacity = slate_size - len(final_ids)
        if remaining_capacity > 0 and self.item_sampler is not None:
            sampled_fill = self.item_sampler.sample(remaining_capacity, exclude_items=exclusions)
            if sampled_fill:
                sampler_total += len(sampled_fill)
                exclusions.update(sampled_fill)
                final_ids.extend(int(item_id) for item_id in sampled_fill)
                if final_scores is not None:
                    final_scores.extend([0.0] * len(sampled_fill))
                if final_ranks is not None:
                    start_rank = final_ranks[-1] + 1 if final_ranks else 1
                    final_ranks.extend(range(start_rank, start_rank + len(sampled_fill)))

        remaining_capacity = slate_size - len(final_ids)
        if remaining_capacity > 0 and take_from_core < len(core_ids):
            extra_needed = min(remaining_capacity, len(core_ids) - take_from_core)
            extra_ids = core_ids[take_from_core : take_from_core + extra_needed]
            final_ids.extend(extra_ids)
            if final_scores is not None and core_scores is not None:
                final_scores.extend(core_scores[take_from_core : take_from_core + extra_needed])
            if final_ranks is not None and core_ranks is not None:
                final_ranks.extend(core_ranks[take_from_core : take_from_core + extra_needed])
            core_boundary += extra_needed

        # Enforce sampler tail positions when possible.
        sampler_slots = min(sampler_slots, slate_size)
        if sampler_slots > 0 and self.item_sampler is not None and final_ids:
            current_sampler_count = len(final_ids) - core_boundary
            needed_tail = max(0, sampler_slots - current_sampler_count)

            if needed_tail > 0:
                tail_exclusions = exclusions.union(final_ids)
                additional_tail = self.item_sampler.sample(needed_tail, exclude_items=tail_exclusions)
                if additional_tail:
                    drop_count = min(len(additional_tail), core_boundary)
                    if drop_count:
                        del final_ids[core_boundary - drop_count : core_boundary]
                        if final_scores is not None:
                            del final_scores[core_boundary - drop_count : core_boundary]
                        if final_ranks is not None:
                            del final_ranks[core_boundary - drop_count : core_boundary]
                        core_boundary -= drop_count
                    sampler_total += len(additional_tail)
                    final_ids.extend(int(item_id) for item_id in additional_tail)
                    if final_scores is not None:
                        final_scores.extend([0.0] * len(additional_tail))
                    exclusions.update(additional_tail)

            # Ensure sampler items occupy the tail in order.
            if len(final_ids) > core_boundary:
                sampler_tail = final_ids[core_boundary:]
                final_ids = final_ids[:core_boundary] + sampler_tail
                if final_scores is not None:
                    sampler_tail_scores = final_scores[core_boundary:]
                    final_scores = final_scores[:core_boundary] + sampler_tail_scores

        # Adjust ranks to be sequential.
        if final_ids:
            final_ranks = list(range(1, len(final_ids) + 1))
        else:
            final_ranks = None

        self._last_sampled_count = sampler_total

        if final_scores is not None and len(final_scores) != len(final_ids):
            # In case score tracking was disabled mid-stream, fall back to None.
            final_scores = None

        return ItemList(None, item_ids=final_ids, scores=final_scores, rank=final_ranks)

    def _select_core_items(
        self,
        user_id: int,
        ids: list[int],
        scores: list[float] | None,
        ranks: list[int] | None,
        desired_count: int,
    ) -> tuple[list[int], list[float] | None, list[int] | None]:
        if desired_count <= 0 or not ids:
            empty_scores = None if scores is None else []
            empty_ranks = None if ranks is None else []
            return [], empty_scores, empty_ranks

        limit = min(desired_count, len(ids))
        selected_ids = list(ids[:limit])

        selected_scores = None
        if scores is not None:
            selected_scores = list(scores[:limit])

        selected_ranks = None
        if ranks is not None:
            selected_ranks = list(ranks[:limit])

        return selected_ids, selected_scores, selected_ranks

    def _current_blocked_items(self, user_id: int) -> set[int]:
        if self.recency_threshold <= 0:
            return set()
        cooldowns = self._cooldown_until_cycle.get(user_id)
        if not cooldowns:
            return set()
        current_cycle = smores.Smores.state.cycle_count
        blocked: set[int] = set()
        expired: list[int] = []
        for item_id, until_cycle in cooldowns.items():
            if current_cycle < until_cycle:
                blocked.add(item_id)
            else:
                expired.append(item_id)
        for item_id in expired:
            del cooldowns[item_id]
        if not cooldowns:
            self._cooldown_until_cycle.pop(user_id, None)
        return blocked

    def _filter_blocked_items(self, user_id: int, item_list: ItemList) -> ItemList:
        blocked = set(self._current_blocked_items(user_id))
        cycle_blocklist = getattr(smores.Smores.state, 'cycle_clicked_blocklist', None)
        if cycle_blocklist:
            blocked.update(cycle_blocklist.get(user_id, set()))
        ids_array = item_list.ids()
        if ids_array is None or len(ids_array) == 0 or not blocked:
            return item_list
        ids = [int(item_id) for item_id in ids_array]
        keep_indices = [idx for idx, item_id in enumerate(ids) if item_id not in blocked]
        if len(keep_indices) == len(ids):
            return item_list
        if not keep_indices:
            return ItemList(None, item_ids=[], scores=None, rank=None)
        scores_array = item_list.scores()
        ranks_array = item_list.ranks()

        filtered_ids = [ids[idx] for idx in keep_indices]
        if scores_array is not None:
            filtered_scores = [float(scores_array[idx]) for idx in keep_indices]
        else:
            filtered_scores = None
        if ranks_array is not None:
            filtered_ranks = [int(ranks_array[idx]) for idx in keep_indices]
        else:
            filtered_ranks = list(range(1, len(filtered_ids) + 1))
        return ItemList(None, item_ids=filtered_ids, scores=filtered_scores, rank=filtered_ranks)

    def update_feedback(self, user_id: int, slate_items: list[int], selected_item) -> None:
        if self.recency_threshold <= 0:
            return

        selected = int(selected_item) if selected_item is not None else None
        current_cycle = smores.Smores.state.cycle_count
        streaks = self._no_click_streaks[user_id]
        cooldowns = self._cooldown_until_cycle[user_id]

        if selected is not None:
            streaks[selected] = 0
            cooldowns.pop(selected, None)

        for item in slate_items:
            item_int = int(item)
            if selected is not None and item_int == selected:
                continue
            until_cycle = cooldowns.get(item_int)
            if until_cycle is not None and current_cycle < until_cycle:
                continue
            streak = streaks.get(item_int, 0) + 1
            streaks[item_int] = streak
            if streak >= self.recency_threshold:
                cooldowns[item_int] = current_cycle + self.cooldown_cycles + 1
                streaks[item_int] = 0

class FixedItemRecommender(Recommender):
    def __init__(self):
        super().__init__()
        # Needs no training
        self.trained = True

    def setup(self, config):
        self.name = config.name

    def isDatasetViable(self):
        return True
    
    def isProfileViable(self, user_id):
        return True
    
    def train(self):
        pass
    
    def get_recommendations(self, user_id) -> ItemList:
        prior_interactions = self.get_user(user_id)
        if prior_interactions is not None and len(prior_interactions) > 0:
            rec_pool = [item for item in list(smores.Smores.state.items.all_items()) if item not in prior_interactions.ids()]
        else:
            rec_pool = list(smores.Smores.state.items.all_items())
        recs = rec_pool[0:smores.Smores.state.slate_size]
        scores = [5.0] * len(recs)
        ranks = list(range(1, len(recs)+1))
        item_list = ItemList(None, item_ids=recs, scores=scores, rank=ranks)
        return item_list


class RecommenderFactory():
    """
    The RecommenderFactory associates names with recommender objects so these can be passed to
    objects based on configuration information. A utility model must registered in the factory before it can be
    created.
    """

    _class_name_map = {}

    @classmethod
    def register(cls, rec_name, rec_class):
        if not issubclass(rec_class, Recommender):
            raise InvalidRecommenderError(rec_name)
        cls._class_name_map[rec_name] = rec_class

    @classmethod
    def register_all(cls, rec_specs):
        for rec_name, rec_class in rec_specs:
            cls.register(rec_name, rec_class)

    @classmethod
    def create(cls, rec_name):
        rec_class = cls._class_name_map.get(rec_name)
        if rec_class is None:
            raise UnregisteredRecommenderError(rec_name)
        return rec_class()

# Registering
RecommenderFactory.register('fixed_recommender', FixedItemRecommender)


# Exceptions
class InvalidRecommenderError(Exception):
    def __init__(self, name):
        self.message = self.message = f'Cannot create recommender: Class {name} is not a subclass of Recommender.'
        super().__init__(self.message)


class UnregisteredRecommenderError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create recommender: Class {name} is not registered and may not exist.'
        super().__init__(self.message)
