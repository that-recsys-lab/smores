import unittest
import yaml
from smores.utils import SmoresConfig

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
      params:
        threshold: 0.2

  recommender_choice_model:
      class_name: fixed
      params:
        recommender_name: Generic

provider:
  utility_model:
    class_name: click_fixed

platform:
  utility_model:
    class_name: null_model

recommenders:
  initial: ["Generic"]
  definitions:
    - name: Generic
      class_name: item_knn_cold
      min_neighbors: 1
      max_neighbors: 20
      min_similarity: 0.001

triggers:
  - name: Cycle5Freeze
    class_name: initial_burnin
    params:
      cycle_count: 5
      recommenders: ["Generic", "Popular Niche"]
'''


class ConfigTestCase(unittest.TestCase):
    def test_config_load(self):
        config_data = yaml.safe_load(SAMPLE_CONFIG1)
        config = SmoresConfig(**config_data)
        self.assertIsNotNone(config)


if __name__ == '__main__':
    unittest.main()