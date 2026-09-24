import unittest
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from lenskit.data import ItemList

import smores
from smores.recommender.recommender import Recommender


class DummySampler:
    """Deterministic sampler for tests."""

    def __init__(self, values):
        self._values = list(values)
        self._next_index = 0
        self.calls = []

    def sample(self, count, exclude_items=None):
        exclude = set(exclude_items or [])
        picked: list[int] = []
        while len(picked) < count and self._next_index < len(self._values):
            candidate = self._values[self._next_index]
            self._next_index += 1
            if candidate in exclude or candidate in picked:
                continue
            picked.append(candidate)
        self.calls.append((count, list(picked)))
        return picked


class DummyRecommender(Recommender):
    """Minimal recommender to exercise sampling logic."""

    def __init__(self, core_ids, sampler, sampled_count=2, candidate_multiplier=1):
        super().__init__()
        self.name = "Dummy"
        self.core_ids = list(core_ids)
        self.item_sampler = sampler
        self.sampled_item_count = sampled_count
        self.candidate_multiplier = candidate_multiplier

    def setup(self, config):
        self.name = getattr(config, "name", "Dummy")

    def isDatasetViable(self):
        return True

    def isProfileViable(self, user_id):
        return True

    def get_user(self, user_id):
        return None

    def train(self):
        return None

    def get_recommendations(self, user_id):
        scores = [1.0] * len(self.core_ids)
        ranks = list(range(1, len(self.core_ids) + 1))
        item_list = ItemList(None, item_ids=self.core_ids, scores=scores, rank=ranks)
        return self.apply_item_sampling(user_id, item_list)


class SlateFillingTestCase(unittest.TestCase):
    """Verify slate filling behaviour with sampler integration."""

    def setUp(self) -> None:
        self._original_state = getattr(smores.Smores, "state", None)
        smores.Smores.state = SimpleNamespace(
            slate_size=10,
            cycle_count=0,
            rand=np.random.default_rng(0),
            cycle_clicked_blocklist=defaultdict(set),
        )

    def tearDown(self) -> None:
        smores.Smores.state = self._original_state

    def test_sampler_top_up_when_core_short(self):
        sampler = DummySampler([101, 102, 103, 104, 105])
        recommender = DummyRecommender(core_ids=[1, 2, 3, 4, 5, 6], sampler=sampler, sampled_count=2)

        slate = recommender.get_recommendations(user_id=42)
        slate_ids = list(slate.ids())

        self.assertEqual(len(slate_ids), 10, "Slate should be filled to requested size")
        self.assertEqual(slate_ids[:6], [1, 2, 3, 4, 5, 6])
        self.assertEqual(slate_ids[6:], [101, 102, 103, 104])
        self.assertEqual(recommender._last_sampled_count, 4)
        self.assertEqual(recommender._last_candidate_count_after_filters, 10)

    def test_candidate_count_tracks_short_and_empty_filtered_pools(self):
        smores.Smores.state.slate_size = 4
        smores.Smores.state.cycle_clicked_blocklist[42].update({2, 4})
        recommender = DummyRecommender(core_ids=[1, 2, 3, 4], sampler=DummySampler([]), sampled_count=0)

        slate = recommender.get_recommendations(user_id=42)

        self.assertEqual(list(slate.ids()), [1, 3])
        self.assertEqual(recommender._last_candidate_count_after_filters, 2)

        recommender.core_ids = []
        slate = recommender.get_recommendations(user_id=42)

        self.assertEqual(list(slate.ids()), [])
        self.assertEqual(recommender._last_candidate_count_after_filters, 0)

    def test_sampler_tail_replaces_excess_core_items(self):
        sampler = DummySampler([201, 202, 203])
        core_ids = list(range(1, 13))  # Core produces more than the slate size
        recommender = DummyRecommender(core_ids=core_ids, sampler=sampler, sampled_count=2)

        slate = recommender.get_recommendations(user_id=7)
        slate_ids = list(slate.ids())

        self.assertEqual(len(slate_ids), 10)
        self.assertEqual(slate_ids[:8], core_ids[:8], "Core items should occupy the head of the slate")
        self.assertEqual(slate_ids[8:], [201, 202], "Sampler must control the tail positions")
        self.assertEqual(recommender._last_sampled_count, 2)
        self.assertIn((2, [201, 202]), sampler.calls)

    def test_candidate_request_count_respects_multiplier(self):
        sampler = DummySampler([])
        recommender = DummyRecommender(core_ids=[], sampler=sampler, sampled_count=2, candidate_multiplier=3)

        self.assertEqual(recommender._candidate_request_count(), 30)
        recommender.candidate_multiplier = 1
        self.assertEqual(recommender._candidate_request_count(), 10)

    def test_cycle_blocklist_removes_clicked_items(self):
        sampler = DummySampler([901, 902, 903, 904])
        recommender = DummyRecommender(core_ids=[1, 2, 3, 4, 5], sampler=sampler, sampled_count=0)
        smores.Smores.state.cycle_clicked_blocklist[42].update({2, 4})

        slate = recommender.get_recommendations(user_id=42)
        slate_ids = set(slate.ids())

        self.assertNotIn(2, slate_ids)
        self.assertNotIn(4, slate_ids)


if __name__ == "__main__":
    unittest.main()
