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


class SmoresTestCase(unittest.TestCase):
    def setUp(self):
        test_data_path = Path('tests/test_data')
        test_config_path = test_data_path / 'test_config.yaml'
        self.config = SmoresConfig.model_validate(yaml.safe_load(test_config_path.read_text()))
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

    def test_run_day(self):
        self.smores.setup()
        self.smores.train_recommenders()
        self.smores.run_day()

    def test_run_cycle(self):
        self.smores.setup()
        self.smores.train_recommenders()
        self.smores.run_cycle()
        self.assertEqual(self.smores.state.current_time(),2)

    
    def test_run_cycles(self):
        self.smores.setup()
        self.smores.train_recommenders()
        self.smores.run_cycles()   
        self.assertEqual(self.smores.state.current_time(),4)

    def test_profile_portability(self):
        self.smores.setup()
        self.smores.train_recommenders()
        # Switching doesn't happen so this shouldn't change anything
        # Universal profile should still be active so all recommenders get updated with
        # all interactions
        self.smores.state.triggers.clear_trigger_type('switch')
        self.smores.run_cycles()
        rec1 = self.smores.state.recommenders_fallback.get_recommender('Popular Niche')
        rec2 = self.smores.state.recommenders_base.get_recommender('Generic')
        self.assertEqual(rec1.dataset.interaction_count, rec2.dataset.interaction_count)

    def test_fallback_update(self):
        self.smores.setup()
        self.smores.train_recommenders()
        self.smores.state.triggers.clear_trigger_type('switch')
        self.smores.state.triggers.clear_trigger_type('interaction')
        self.smores.run_cycles()
        rec1 = self.smores.state.recommenders_fallback.get_recommender('Popular Fallback')
        rec2 = self.smores.state.recommenders_base.get_recommender('Generic')
        self.assertEqual(rec1.dataset.interaction_count, rec2.dataset.interaction_count)

        rec3 = self.smores.state.recommenders_fallback.get_recommender('Popular Niche')
        self.assertEqual(rec1.dataset.interaction_count, 0)
                         
if __name__ == '__main__':
    unittest.main()
    