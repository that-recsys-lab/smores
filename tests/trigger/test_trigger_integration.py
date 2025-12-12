import shutil
import tempfile
import unittest
from csv import DictReader
from pathlib import Path

import smores
from smores.smores import Smores
from smores.trigger.switch_trigger import ProfileColdStartTrigger, ProfileUserOwnershipTrigger, SwitchEvent
from smores.utils import SmoresConfig


TEST_DATA_DIR = Path(__file__).parent / "../test_data"


def _build_config(initial_recs, triggers, include_niche=False):
    data_dir = (TEST_DATA_DIR).resolve()
    temp_output = tempfile.mkdtemp(prefix="smores-test-output-")

    include_niche_rec = (
        include_niche
        or "Niche" in initial_recs
        or any(t.get("name") == "Universal" for t in triggers)
    )

    def _ordered_types(type_names):
        return sorted(type_names, key=lambda name: (name != "Generic", name))

    with (data_dir / "consumers.csv").open(newline="") as consumer_file:
        consumer_type_names = {"Generic"}
        for row in DictReader(consumer_file):
            consumer_type_names.add(row["consumer_type"])

    with (data_dir / "providers.csv").open(newline="") as provider_file:
        provider_type_names = {"Generic"}
        for row in DictReader(provider_file):
            provider_type_names.add(row["provider_type"])

    base_recs = [
        {
            "name": "Generic",
            "class_name": "popular",
            "params": {
                "min_user_count": 1,
                "min_interaction_count": 1,
            },
        }
    ]

    if include_niche_rec:
        base_recs.append(
            {
                "name": "Niche",
                "class_name": "popular",
                "params": {
                    "min_user_count": 1,
                    "min_interaction_count": 1,
                },
            }
        )

    consumer_types = []
    for type_name in _ordered_types(consumer_type_names):
        recommender_name = type_name if type_name in initial_recs else initial_recs[0]
        consumer_types.append(
            {
                "name": type_name,
                "utility_model": "FixedUtility",
                "item_selection_model": "CategorySimilarity",
                "recommender_choice_model": {
                    "class_name": "fixed",
                    "params": {"recommender_name": recommender_name},
                },
            }
        )

    provider_types = [
        {"name": type_name, "utility_model": "ClickFixed"}
        for type_name in _ordered_types(provider_type_names)
    ]

    config_dict = {
        "simulation": {
            "experiment_name": "trigger_integration_test",
            "num_days": 1,
            "num_cycles": 1,
            "slate_size": 2,
            "seed": 20250101,
        },
        "data": {
            "directory": str(data_dir),
            "consumer_file": "consumers.csv",
            "item_file": "items.csv",
            "provider_file": "providers.csv",
        },
        "output": {
            "directory": temp_output,
            "consumer_file": "consumer_utility",
            "provider_file": "provider_utility",
            "choice_file": "choice_utility",
            "debug_file": "debug",
            "debug_level": "info",
            "use_timestamp": False,
            "use_parquet": False,
        },
        "summary_logger": {"enabled": False},
        "consumer": {
            "initial_recommender": initial_recs[0],
            "models": {
                "utility": [
                    {"name": "FixedUtility", "class_name": "fixed_utility", "params": {"value": 1.0}}
                ],
                "item_selection": [
                    {
                        "name": "CategorySimilarity",
                        "class_name": "category_similarity_logit",
                        "params": {"threshold": 0.0, "selection_utility": 1.0},
                    }
                ],
            },
            "types": consumer_types,
        },
        "provider": {
            "initial_recommender": "__all__",
            "models": {"utility": [{"name": "ClickFixed", "class_name": "click_fixed", "params": {"value": 1.0}}]},
            "types": provider_types,
        },
        "platform": {"utility_model": {"class_name": "null_model"}},
        "recommender": {
            "initial": list(initial_recs),
            "base_recommenders": base_recs,
            "fallback_recommenders": [],
        },
        "triggers": triggers,
    }

    return SmoresConfig.model_validate(config_dict), temp_output


def _get_first_consumer_and_item():
    state = smores.Smores.state
    consumer_id = next(iter(state.consumers.collection.keys()))
    item_id = next(iter(state.items.item_map.keys()))
    return consumer_id, item_id


def _interaction_count(rec):
    dataset = rec.get_dataset()
    if dataset is None:
        return 0
    return dataset.interaction_count


class TriggerIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.prev_state = smores.Smores.state
        self._temp_dirs = []

    def tearDown(self):
        state = smores.Smores.state
        logger = getattr(state, "logger", None)
        if logger is not None:
            logger.cleanup()
        smores.Smores.state = self.prev_state
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)

    def build_system(self, initial_recs, triggers, include_niche=False):
        config, out_dir = _build_config(initial_recs, triggers, include_niche=include_niche)
        self._temp_dirs.append(out_dir)
        system = Smores(config)
        system.setup()
        return system


class AlgorithmSpecificIntegrationTest(TriggerIntegrationTestCase):
    def test_only_active_recommender_retains_interactions(self):
        system = self.build_system(["Generic"], [], include_niche=True)

        consumer_id, item_id = _get_first_consumer_and_item()
        system.process_interactions([(consumer_id, item_id, "Generic", 1, 0)])

        base = smores.Smores.state.recommenders_base
        generic = base.get_recommender("Generic")
        self.assertEqual(1, _interaction_count(generic))
        if base.is_recommender("Niche"):
            niche = base.get_recommender("Niche")
            self.assertEqual(0, _interaction_count(niche))


class MonolithicIntegrationTest(TriggerIntegrationTestCase):
    def test_single_recommender_stores_interactions(self):
        system = self.build_system(["Generic"], [], include_niche=False)

        consumer_id, item_id = _get_first_consumer_and_item()
        system.process_interactions([(consumer_id, item_id, "Generic", 1, 0)])

        base = smores.Smores.state.recommenders_base
        generic = base.get_recommender("Generic")
        self.assertEqual(1, _interaction_count(generic))


class ColdStartIntegrationTest(TriggerIntegrationTestCase):
    def test_cold_start_resets_source_profile(self):
        system = self.build_system(["Generic"], [], include_niche=True)

        consumer_id, item_id = _get_first_consumer_and_item()
        base = smores.Smores.state.recommenders_base
        generic = base.get_recommender("Generic")

        system.process_interactions([(consumer_id, item_id, "Generic", 1, 0)])
        self.assertGreater(_interaction_count(generic), 0)
        self.assertIsNotNone(generic.get_user(consumer_id))

        trigger = ProfileColdStartTrigger()
        trigger.setup(type("Cfg", (), {"name": "Cold", "params": {}}))
        trigger.handle_event(SwitchEvent(type("Consumer", (), {"id": consumer_id}), "Generic", "Niche"))

        self.assertEqual(0, _interaction_count(generic))
        post_profile = generic.get_user(consumer_id)
        self.assertTrue(post_profile is None or post_profile.ids().size == 0)


class OwnershipIntegrationTest(TriggerIntegrationTestCase):
    def test_profile_moves_to_destination(self):
        system = self.build_system(["Generic"], [], include_niche=True)

        consumer_id, item_id = _get_first_consumer_and_item()
        base = smores.Smores.state.recommenders_base
        generic = base.get_recommender("Generic")
        niche = base.get_recommender("Niche")

        system.process_interactions([(consumer_id, item_id, "Generic", 1, 0)])
        trigger = ProfileUserOwnershipTrigger()
        trigger.setup(type("Cfg", (), {"name": "Ownership", "params": {}}))
        trigger.handle_event(SwitchEvent(type("Consumer", (), {"id": consumer_id}), "Generic", "Niche"))

        self.assertEqual(0, _interaction_count(generic))
        self.assertGreaterEqual(_interaction_count(niche), 1)


class UniversalIntegrationTest(TriggerIntegrationTestCase):
    def test_interactions_replicated_to_all_recommenders(self):
        system = self.build_system(
            ["Generic", "Niche"],
            triggers=[{"name": "Universal", "class_name": "universal_profile", "params": {}}],
            include_niche=True,
        )

        consumer_id, item_id = _get_first_consumer_and_item()
        interactions = [(consumer_id, item_id, "Generic", 1, 0)]
        system.process_interactions(interactions)

        base = smores.Smores.state.recommenders_base
        generic = base.get_recommender("Generic")
        niche = base.get_recommender("Niche")

        self.assertGreaterEqual(_interaction_count(generic), 1)
        self.assertGreaterEqual(_interaction_count(niche), 1)


if __name__ == "__main__":
    unittest.main()
