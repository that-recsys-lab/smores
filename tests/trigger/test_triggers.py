import unittest
from types import SimpleNamespace

import smores
from smores.trigger.trigger import (
    InitialBurnInTrigger,
    CycleEvent,
    InteractionBatchEvent,
    UniversalProfileTrigger,
)
from smores.trigger.switch_trigger import (
    ProfileColdStartTrigger,
    ProfileUserOwnershipTrigger,
    SwitchEvent,
)

from .test_trigger_integration import TriggerIntegrationTestCase


def _first_consumer():
    state = smores.Smores.state
    return next(iter(state.consumers.collection.keys()))


def _first_items(count: int):
    state = smores.Smores.state
    iterator = iter(state.items.item_map.keys())
    return [next(iterator) for _ in range(count)]


def _simulate_days(system, rec_name: str, consumer_id: int, item_ids: list[int]):
    for day, item_id in enumerate(item_ids):
        system.process_interactions([(consumer_id, item_id, rec_name, 1, day)])


def _interaction_count(rec):
    dataset = rec.get_dataset()
    return dataset.interaction_count if dataset is not None else 0


class InitialBurnInTriggerTest(TriggerIntegrationTestCase):
    """Algorithm-specific helper: deferred activation after burn-in cycles."""

    def test_activates_configured_recommenders(self):
        self.build_system(["Generic"], [], include_niche=True)

        state = smores.Smores.state
        self.assertNotIn("Niche", state.recommenders_active)

        trigger = InitialBurnInTrigger()
        trigger.setup(
            type(
                "Config",
                (),
                {
                    "name": "BurnIn",
                    "params": {
                        "cycle_count": 1,
                        "repeating": "False",
                        "recommenders": ["Generic", "Niche"],
                    },
                },
            )
        )

        trigger.handle_event(CycleEvent(0, time=0))
        self.assertEqual(["Generic", "Niche"], state.recommenders_active)
        self.assertIn("Niche", state.recommenders_active)


class SwitchTriggerTests(TriggerIntegrationTestCase):
    """Switch triggers implement the four profile-portability regimes."""

    def test_algorithm_specific_baseline_retains_profile_locally(self):
        system = self.build_system(["Generic"], [], include_niche=True)

        consumer_id = _first_consumer()
        items = _first_items(2)
        _simulate_days(system, "Generic", consumer_id, items)

        base = smores.Smores.state.recommenders_base
        generic = base.get_recommender("Generic")
        niche = base.get_recommender("Niche")

        self.assertEqual(len(items), _interaction_count(generic))
        self.assertEqual(0, _interaction_count(niche))

    def test_monolithic_single_recommender_no_cross_feed(self):
        system = self.build_system(["Generic"], [], include_niche=False)

        consumer_id = _first_consumer()
        items = _first_items(2)
        _simulate_days(system, "Generic", consumer_id, items)

        base = smores.Smores.state.recommenders_base
        generic = base.get_recommender("Generic")

        self.assertEqual(len(items), _interaction_count(generic))

    def test_cold_start_deletes_user_profile(self):
        system = self.build_system(["Generic"], [], include_niche=True)

        consumer_id = _first_consumer()
        items = _first_items(2)
        _simulate_days(system, "Generic", consumer_id, items)

        base = smores.Smores.state.recommenders_base
        generic = base.get_recommender("Generic")

        self.assertEqual(len(items), _interaction_count(generic))
        self.assertIsNotNone(generic.get_user(consumer_id))

        trigger = ProfileColdStartTrigger()
        trigger.setup(type("Config", (), {"name": "Cold", "params": {}}))
        consumer = SimpleNamespace(id=consumer_id)
        trigger.handle_event(SwitchEvent(consumer, "Generic", "Niche"))

        self.assertEqual(0, _interaction_count(generic))
        profile = generic.get_user(consumer_id)
        self.assertTrue(profile is None or profile.ids().size == 0)

    def test_user_ownership_transfers_and_removes_profile(self):
        system = self.build_system(["Generic"], [], include_niche=True)

        consumer_id = _first_consumer()
        items = _first_items(2)
        _simulate_days(system, "Generic", consumer_id, items)

        base = smores.Smores.state.recommenders_base
        generic = base.get_recommender("Generic")
        niche = base.get_recommender("Niche")

        trigger = ProfileUserOwnershipTrigger()
        trigger.setup(type("Config", (), {"name": "Ownership", "params": {}}))
        consumer = SimpleNamespace(id=consumer_id)
        trigger.handle_event(SwitchEvent(consumer, "Generic", "Niche"))

        self.assertEqual(0, _interaction_count(generic))
        niche_profile = niche.get_user(consumer_id)
        self.assertIsNotNone(niche_profile)
        self.assertEqual(len(items), niche_profile.ids().size)


class UniversalProfileTriggerTest(TriggerIntegrationTestCase):
    """Universal regime: replicate every interaction to all other recommenders."""

    def test_replays_interactions_to_other_recommenders(self):
        self.build_system(["Generic"], [], include_niche=True)

        base = smores.Smores.state.recommenders_base
        generic = base.get_recommender("Generic")
        niche = base.get_recommender("Niche")

        consumer_id = _first_consumer()
        items = _first_items(2)
        day_batches = [
            [(consumer_id, items[0], 1, 0)],
            [(consumer_id, items[1], 1, 1)],
        ]

        for batch in day_batches:
            generic.update_dataset(batch)

        self.assertEqual(len(day_batches), _interaction_count(generic))
        self.assertEqual(0, _interaction_count(niche))

        trigger = UniversalProfileTrigger()
        trigger.setup(type("Config", (), {"name": "Universal", "params": {}}))
        for batch in day_batches:
            trigger.handle_event(InteractionBatchEvent({"Generic": batch}))

        self.assertEqual(len(day_batches), _interaction_count(generic))
        self.assertEqual(len(day_batches), _interaction_count(niche))

        niche_profile = niche.get_user(consumer_id)
        self.assertIsNotNone(niche_profile)
        self.assertEqual(len(day_batches), niche_profile.ids().size)


if __name__ == '__main__':
    unittest.main()
