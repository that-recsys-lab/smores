import unittest
import yaml
import csv
from pathlib import Path

from smores.utils import SmoresConfig
from smores.recommender import RecommenderFactory, Recommender, PopularRecommender, RecommenderMap
from smores import Smores
from tests.paths import FIXTURE_DATA_DIR, TEST_CONFIG_PATH

from icecream import ic

class RecommenderTestCase(unittest.TestCase):
    def setUp(self):
        self.config = SmoresConfig.model_validate(yaml.safe_load(TEST_CONFIG_PATH.read_text()))
        self.smores = Smores(self.config)
        self.smores.setup()

        interactions_path = FIXTURE_DATA_DIR / 'interactions.csv'
        with open(interactions_path, ) as csvfile:
          reader = csv.reader(csvfile, delimiter=',')
          # skip header row
          reader.__next__()
          self.interactions = []
          for row in reader:
            row_int = [int(entry) for entry in row]
            self.interactions.append(row_int)
      

    def test_component_creation(self):
      rec_config = self.config.recommender.base_recommenders[0]
      rec: Recommender = RecommenderFactory.create(rec_config.class_name)
      self.assertIsNotNone(rec)
      rec.setup(rec_config)
      self.assertIsNotNone(rec.cold_user_fallback)

    def test_dataset_update(self):
      rec_config = self.config.recommender.base_recommenders[0]
      rec: Recommender = RecommenderFactory.create(rec_config.class_name)
      # Need to configure this so that lookups inside of the base recommender will succeed.
      fallback_config = self.config.recommender.fallback_recommenders
      self.smores.state.recommenders_fallback.setup(fallback_config)
      self.assertTrue(self.smores.state.recommenders_fallback.is_recommender('Popular Fallback'))
      self.assertIsNotNone(rec)
      rec.setup(rec_config)
      rec.setup_dataset()
      initial_items = set(self.smores.state.items.all_items())
      self.assertIsNotNone(self.interactions)
      time_step1 = [row for row in self.interactions if row[3] == 1]
      time_step2 = [row for row in self.interactions if row[3] == 2]
      time_step3 = [row for row in self.interactions if row[3] == 3]

      rec.update_dataset(time_step1)
      self.assertEqual(rec.dataset.interaction_count, 5)

      rec.update_dataset(time_step2)
      self.assertEqual(rec.dataset.interaction_count, 10)
      self.assertEqual(rec.dataset.user_count, 5)
      expected_items = initial_items.union({row[1] for row in time_step1 + time_step2})
      self.assertEqual(rec.dataset.item_count, len(expected_items))
      self.assertEqual(rec.dataset.user_row(100).ids().size, 2)

    def _build_popular_recommender(self) -> Recommender:
      rec_config = self.config.recommender.fallback_recommenders[0]
      rec: Recommender = RecommenderFactory.create(rec_config.class_name)
      rec.setup(rec_config)
      rec.setup_dataset()
      return rec

    def test_popular_items_representation_empty_dataset(self):
      rec = self._build_popular_recommender()
      self.assertEqual(rec.get_popular_items_representation(), [])

    def test_popular_items_representation_normalized(self):
      rec = self._build_popular_recommender()
      time_step1 = [row for row in self.interactions if row[3] == 1]
      rec.update_dataset(time_step1)

      representation = rec.get_popular_items_representation(items_count=3)
      self.assertGreater(len(representation), 0)
      self.assertEqual(len(representation), len(self.smores.state.items.get_item(200).features))
      self.assertAlmostEqual(sum(representation), 1.0, places=5)
      self.assertTrue(all(value >= 0 for value in representation))
      self.assertEqual(rec.get_popular_items_representation(items_count=0), [])

    def test_popular_items_representation_cycle_filter(self):
      rec = self._build_popular_recommender()
      rec.update_dataset(self.interactions)

      cycle0_representation = rec.get_popular_items_representation(items_count=3, cycle=0)
      self.assertGreater(len(cycle0_representation), 0)
      self.assertAlmostEqual(sum(cycle0_representation), 1.0, places=5)

      # In fixture data, later-cycle interactions include item IDs that are not in items.csv.
      cycle1_representation = rec.get_popular_items_representation(items_count=3, cycle=1)
      self.assertEqual(cycle1_representation, [])

    def fallback_creation(self):
      rec_config = self.config.recommender.definitions[0]
      rec: Recommender = RecommenderFactory.create(rec_config.class_name)
      self.assertIsNotNone(rec)
      rec.setup(rec_config)
      self.assertIsNotNone(rec.cold_user_fallback)
      self.assertIsInstance(rec.cold_user_fallback, PopularRecommender)
        

if __name__ == '__main__':
    unittest.main()
