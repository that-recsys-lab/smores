import unittest
import yaml
from pathlib import Path
from icecream import ic

from smores.recommender import ItemKnnRecommender, PopularRecommender
from smores.stakeholders.consumer import CategorySimilarityLogitModel
from smores.stakeholders.provider import ProviderClickFixedUtilityModel
from smores.item import Item
from smores.trigger import InitialBurnInTrigger
from smores.utils import SmoresConfig
from smores import Smores
from tests.paths import TEST_CONFIG_PATH


class SmoresTestCase(unittest.TestCase):
    def setUp(self):
        self.config = SmoresConfig.model_validate(yaml.safe_load(TEST_CONFIG_PATH.read_text()))
        self.smores = Smores(self.config)

    def testInit(self):
        self.assertIsNotNone(self.smores.state)
        self.assertIsNotNone(self.smores.state.recommenders_active)
        self.assertIsNotNone(self.smores.state.recommenders_base)
        self.assertIsNotNone(self.smores.state.recommenders_fallback)
        self.assertIsNotNone(self.smores.state.initial_recommenders)
        self.assertIsNotNone(self.smores.state.triggers)

    def testSetup(self):
        self.smores.setup()

        # Add to this as more things get implemented.
        # CONSUMERS
        cmodels = self.smores.state.consumer_models
        self.assertIsInstance(cmodels.get_item_selection_model('Category Similarity'), CategorySimilarityLogitModel)
        ccoll = self.smores.state.consumers
        self.assertEqual(len(list(ccoll)), 3)

        consumer = next(iter(ccoll))
        self.assertIsInstance(consumer.recommender, ItemKnnRecommender)

        # ITEMS
        imap = self.smores.state.items
        item201 = imap.get_item(201)
        self.assertIsInstance(item201, Item)
        self.assertEqual(item201.provider_id, 300)
        self.assertEqual(item201.features[0], 0.1)

        # PROVIDERS
        pmodels = self.smores.state.provider_models
        self.assertIsInstance(pmodels.get_utility_model('Click Fixed 1.0'), ProviderClickFixedUtilityModel)
        pcoll = self.smores.state.providers
        self.assertEqual(len(list(pcoll)), 4)

        # RECOMMENDERS
        rec_map_base = self.smores.state.recommenders_base
        rec_map_fallback = self.smores.state.recommenders_fallback
        self.assertIsInstance(rec_map_base.get_recommender('Generic'), ItemKnnRecommender)
        self.assertIsInstance(rec_map_fallback.get_recommender('Popular Niche'), PopularRecommender)

        # TRIGGERS
        trigger_coll = self.smores.state.triggers
        self.assertEqual(len(trigger_coll.get_triggers('cycle')), 1)
        self.assertIsInstance(trigger_coll.get_triggers('cycle')[0], InitialBurnInTrigger)
        self.assertEqual(len(trigger_coll.get_triggers('day')), 0)

    def test_run_consumer_day(self):
        self.smores.setup()
        self.smores.train_recommenders()
        test_consumer = self.smores.state.consumers.get_consumer(101)
        self.smores.run_consumer_day(test_consumer)
        metrics = self.smores.state.recommender_metrics["Generic"]
        self.assertEqual(metrics["served_users"], {101})

    def test_cycle_metrics_separate_served_users_from_end_assignments(self):
        self.smores.setup()
        state = self.smores.state
        state.cycle_start_recommender_users = self.smores._assigned_users_by_recommender()

        consumers = list(state.consumers)
        generic_metrics = state.recommender_metrics["Generic"]
        generic_metrics["recommendations"] = len(consumers) * state.day_limit
        generic_metrics["served_users"] = {consumer.id for consumer in consumers}
        generic_metrics["clicks"] = 4
        generic_metrics["fallback_used"] = 2
        generic_metrics["sampled_items"] = 3
        generic_metrics["slate_items_total"] = 30
        generic_metrics["unique_items"] = {201, 202}

        # Simulate one consumer switching away after all cycle traffic was served.
        consumers[0].recommender = None
        rows = []
        state.logger.log_cycle_metrics = rows.append

        self.smores._log_cycle_stats(display_cycle=1)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["users_at_cycle_start"], 3)
        self.assertEqual(row["served_users"], 3)
        self.assertEqual(row["users_at_cycle_end"], 2)
        self.assertEqual(row["new_assignments"], 0)
        self.assertEqual(row["departures"], 1)
        self.assertEqual(row["rec_requests"], 6)
        self.assertEqual(row["clicks"], 4)
        self.assertEqual(row["fallback_used"], 2)
        self.assertEqual(row["sampled_items"], 3)
        self.assertEqual(row["slate_items_total"], 30)
        self.assertEqual(row["platform_unique_items"], 2)
        self.assertEqual(row["platform_coverage_pct"], 40.0)
        self.assertNotIn("active_users", row)

    def test_run_day(self):
        self.smores.setup()
        self.smores.train_recommenders()
        self.smores.run_day()

    def test_run_cycle(self):
        self.smores.setup()
        self.smores.train_recommenders()
        self.smores.run_cycle()
        self.assertEqual(self.smores.state.current_time(),2)

    def test_cycle_refreshes_representations_before_recommender_choice(self):
        self.smores.setup()
        events = []
        self.smores.update_recommender_representations = lambda: events.append("representations")
        self.smores.cycle_actions = lambda: events.append("choice")

        self.smores.run_cycle()

        self.assertEqual(events, ["representations", "choice"])

    
    def test_run_cycles(self):
        self.smores.setup()
        self.smores.train_recommenders()
        self.smores.run_cycles()   
        self.assertEqual(self.smores.state.current_time(),4)

    def test_profile_portability(self):
        self.smores.setup()
        self.smores.train_recommenders()
        # Switching does not happen in this run; attached fallback recommenders
        # should read from their parent recommender dataset.
        self.smores.state.triggers.clear_trigger_type('switch')
        self.smores.run_cycles()
        rec1 = self.smores.state.recommenders_fallback.get_recommender('Popular Fallback')
        rec2 = self.smores.state.recommenders_base.get_recommender('Generic')
        self.assertIs(rec1.parent, rec2)
        self.assertIsNone(rec1.dataset)
        self.assertEqual(rec1.get_dataset().interaction_count, rec2.get_dataset().interaction_count)

    def test_fallback_update(self):
        self.smores.setup()
        self.smores.train_recommenders()
        self.smores.state.triggers.clear_trigger_type('switch')
        self.smores.state.triggers.clear_trigger_type('interaction')
        self.smores.run_cycles()
        rec1 = self.smores.state.recommenders_fallback.get_recommender('Popular Fallback')
        rec2 = self.smores.state.recommenders_base.get_recommender('Generic')
        self.assertEqual(rec1.get_dataset().interaction_count, rec2.get_dataset().interaction_count)

        rec3 = self.smores.state.recommenders_fallback.get_recommender('Popular Niche')
        self.assertIsNone(rec3.parent)
        self.assertIsNone(rec3.dataset)
                         
if __name__ == '__main__':
    unittest.main()
    
