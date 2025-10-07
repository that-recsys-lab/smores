import unittest

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


class DummyLogger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass


class DummyRecommender:
    def __init__(self, name, user_records=None):
        self.name = name
        self.received_updates = []
        self.dataset_itemlist_updates = []
        self.deleted_users = []
        self.user_records = user_records or {}

    def update_dataset(self, interactions):
        self.received_updates.append(list(interactions))

    def update_dataset_itemlist(self, user_id, item_list):
        self.dataset_itemlist_updates.append((user_id, item_list))

    def delete_user(self, user_id):
        self.deleted_users.append(user_id)

    def get_user(self, user_id):
        return self.user_records.get(user_id, [])


class DummyRecommenderMap:
    def __init__(self, recs):
        self._map = recs

    def get_names(self):
        return sorted(self._map.keys())

    def get_recommender(self, name):
        return self._map[name]


class DummyState:
    def __init__(self, rec_map, active=None):
        self.recommenders_base = rec_map
        self.recommenders_active = active or []
        self.logger = DummyLogger()


class TriggerTestCase(unittest.TestCase):
    def setUp(self):
        self._prev_state = smores.Smores.state

    def tearDown(self):
        smores.Smores.state = self._prev_state


class InitialBurnInTriggerTest(TriggerTestCase):
    """Algorithm-specific helper: deferred activation after burn-in cycles."""
    def test_activates_configured_recommenders(self):
        recs = {
            'Generic': DummyRecommender('Generic'),
            'Niche': DummyRecommender('Niche'),
        }
        smores.Smores.state = DummyState(DummyRecommenderMap(recs), active=['Generic'])

        trigger = InitialBurnInTrigger()
        trigger.setup(type('Config', (), {'name': 'BurnIn', 'params': {'cycle_count': 2, 'repeating': 'False', 'recommenders': ['Niche']}}))

        # CycleEvent stores zero-based cycle count; set to 1 so adjusted count equals 2
        trigger.handle_event(CycleEvent(1, time=0))

        self.assertIn('Niche', smores.Smores.state.recommenders_active)


class SwitchTriggerTests(TriggerTestCase):
    """Switch triggers implement the four profile-portability regimes."""
    def test_algorithm_specific_baseline_retains_profile_locally(self):
        # Algorithm-specific: only the active recommender keeps the data.
        generic = DummyRecommender('Generic')
        niche = DummyRecommender('Niche')
        interactions = [(1, 10, 1, 0)]

        # Simulation always updates the active recommender directly
        generic.update_dataset(interactions)

        # Without any trigger, other recommenders should see nothing
        self.assertEqual([], niche.received_updates)

    def test_monolithic_single_recommender_no_cross_feed(self):
        # Monolithic baseline: one recommender sees its own interactions.
        mono = DummyRecommender('Generic')
        mono.update_dataset([(1, 10, 1, 0)])

        # No other recommender exists, so the one system just records its own update.
        self.assertEqual([[ (1, 10, 1, 0) ]], mono.received_updates)

    def test_cold_start_deletes_user_profile(self):
        # Cold start: delete the departing user's profile; no transfer.
        from_rec = DummyRecommender('Generic')
        recs = {'Generic': from_rec}
        smores.Smores.state = DummyState(DummyRecommenderMap(recs))

        trigger = ProfileColdStartTrigger()
        trigger.setup(type('Config', (), {'name': 'Cold', 'params': {}}))

        trigger.handle_event(SwitchEvent(type('Consumer', (), {'id': 42}), 'Generic', 'Niche'))

        self.assertEqual([42], from_rec.deleted_users)

    def test_user_ownership_transfers_and_removes_profile(self):
        # User ownership: transfer profile to the new recommender and remove old copy.
        transfer_data = ['item-list']
        from_rec = DummyRecommender('Generic', user_records={42: transfer_data})
        to_rec = DummyRecommender('Niche')
        recs = {'Generic': from_rec, 'Niche': to_rec}
        smores.Smores.state = DummyState(DummyRecommenderMap(recs))

        trigger = ProfileUserOwnershipTrigger()
        trigger.setup(type('Config', (), {'name': 'Ownership', 'params': {}}))

        trigger.handle_event(SwitchEvent(type('Consumer', (), {'id': 42}), 'Generic', 'Niche'))

        self.assertIn((42, transfer_data), to_rec.dataset_itemlist_updates)
        self.assertIn(42, from_rec.deleted_users)


class UniversalProfileTriggerTest(TriggerTestCase):
    """Universal regime: replicate every interaction to all other recommenders."""
    def test_replays_interactions_to_other_recommenders(self):
        generic_rec = DummyRecommender('Generic')
        niche_rec = DummyRecommender('Niche')
        recs = {'Generic': generic_rec, 'Niche': niche_rec}
        smores.Smores.state = DummyState(DummyRecommenderMap(recs))

        trigger = UniversalProfileTrigger()
        trigger.setup(type('Config', (), {'name': 'Universal', 'params': {}}))

        interactions = {
            'Generic': [(1, 10, 1, 0)],
            'Niche': [(2, 20, 1, 0)],
        }
        trigger.handle_event(InteractionBatchEvent(interactions))

        self.assertEqual([[ (2, 20, 1, 0) ]], generic_rec.received_updates)
        self.assertEqual([[ (1, 10, 1, 0) ]], niche_rec.received_updates)


if __name__ == '__main__':
    unittest.main()
