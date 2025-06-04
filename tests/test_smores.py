import unittest
import yaml
from pathlib import Path
import os

from smores.recommender import ItemKnnRecommender, PopularRecommender
from smores.trigger import InitialBurnInTrigger
from smores.utils import SmoresConfig
from smores import Smores

from icecream import ic

class SmoresTestCase(unittest.TestCase):
    def setUp(self):
        test_data_path = Path('tests/test_data')
        test_config_path = test_data_path / 'test_config.yaml'
        self.config = SmoresConfig.model_validate(yaml.safe_load(test_config_path.read_text()))
        self.smores = Smores(self.config)

    def testInit(self):
        self.assertIsNotNone(self.smores.state)
        self.assertIsNotNone(self.smores.state.recommenders_active)
        self.assertIsNotNone(self.smores.state.recommenders_available)
        self.assertIsNotNone(self.smores.state.initial_recommenders)
        self.assertIsNotNone(self.smores.state.time_triggers)

    def testSetup(self):
        self.smores.setup()

        # Add to this as more things get implemented.
        rec_map = self.smores.state.recommenders_available
        self.assertIsInstance(rec_map.get_recommender('Generic'), ItemKnnRecommender)
        self.assertIsInstance(rec_map.get_recommender('Popular Niche'), PopularRecommender)

        trigger_coll = self.smores.state.time_triggers
        self.assertEqual(len(trigger_coll.get_triggers('cycle')), 1)
        self.assertIsInstance(trigger_coll.get_triggers('cycle')[0], InitialBurnInTrigger)
        self.assertEqual(len(trigger_coll.get_triggers('day')), 0)
                         
if __name__ == '__main__':
    unittest.main()
    