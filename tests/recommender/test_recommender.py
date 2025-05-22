import unittest
import yaml
import csv

from smores.utils import SmoresConfig
from smores.recommender import RecommenderFactory, Recommender, PopularRecommender
from smores import Smores

from icecream import ic

SAMPLE_CONFIG1 = \
'''
simulation:
    experiment_name: test_experiment
    num_days: 10
    num_cycles: 10
    slate_size: 5
    seed: 20250513

data:
  directory: ../../../data/raw/ambar
  consumer_file: users.csv
  item_file: items.csv
  provider_file: artists.csv`

consumer:
  recommender_assignment:
    class_name: fixed
    name: Generic

  utility_model:
    class_name: list_average

  item_selection_model:
    class_name: category_similarity_logit
    threshold: 0.2

  recommender_choice_model:
    class_name: fixed
    recommender_name: Generic

provider:
  utility_model:
    class_name: click_fixed

platform:
  utility_model:
    class_name: null_model

recommenders:
  - name: Generic
    class_name: item_knn
    max_neighbors: 20
    min_neighbors: 2
    min_similarity: 0.0001
    min_user_count: 20
    min_interaction_count: 500
    min_profile_size: 5
    cold_user_fallback:
        class_name: popular
        min_user_count: 20
        min_interaction_count: 500

  - name: Niche
    class_name: niche

triggers:
  - name: Cycle5Freeze
    class_name: initial_burnin
    cycle_count: 5
'''

SAMPLE_INTERACTIONS = '''user_id,item_id,rating,time
100,200,1,1
101,201,1,1
102,202,1,1
103,203,1,1
104,200,1,1
100,210,1,2
101,211,1,2
102,212,1,2
103,213,1,2
104,210,1,2
100,220,1,3
101,221,1,3
102,222,1,3
103,223,1,3
'''


class RecommenderTestCase(unittest.TestCase):
    def setUp(self):
      config_raw = yaml.safe_load(SAMPLE_CONFIG1)
      self.config = SmoresConfig(**config_raw)
      Smores.state = Smores.SmoresState(self.config)

      reader = csv.reader(SAMPLE_INTERACTIONS.splitlines(), delimiter=',')
      # skip header row
      reader.__next__()
      self.interactions = []
      for row in reader:
        row_int = [int(entry) for entry in row]
        self.interactions.append(row_int)
      

    def test_component_creation(self):
      rec_config = self.config.recommenders[0]
      rec: Recommender = RecommenderFactory.create(rec_config.class_name)
      self.assertIsNotNone(rec)
      rec.setup(rec_config)
      self.assertIsNotNone(rec.cold_user_fallback)

    def test_dataset_update(self):
      rec_config = self.config.recommenders[0]
      rec: Recommender = RecommenderFactory.create(rec_config.class_name)
      self.assertIsNotNone(rec)
      rec.setup(rec_config)
      self.assertIsNotNone(self.interactions)
      time_step1 = [row for row in self.interactions if row[3] == 1]
      time_step2 = [row for row in self.interactions if row[3] == 2]
      time_step3 = [row for row in self.interactions if row[3] == 3]

      rec.update_dataset(time_step1)
      self.assertEqual(rec.dataset.interaction_count, 5)

      rec.update_dataset(time_step2)
      self.assertEqual(rec.dataset.interaction_count, 10)
      self.assertEqual(rec.dataset.user_count, 5)
      self.assertEqual(rec.dataset.item_count, 8)
      self.assertEqual(rec.dataset.user_row(100).ids().size, 2)

    def fallback_creation(self):
      rec_config = self.config.recommenders[0]
      rec: Recommender = RecommenderFactory.create(rec_config.class_name)
      self.assertIsNotNone(rec)
      rec.setup(rec_config)
      self.assertIsNotNone(rec.cold_user_fallback)
      self.assertIsInstance(rec.cold_user_fallback, PopularRecommender)
        

if __name__ == '__main__':
    unittest.main()