import unittest
import yaml
from smores import SmoresConfig

SAMPLE_CONFIG1 = \
'''
# Sample SMORES configuration

simulation:
    experiment_name: test_experiment
    num_days: 10
    num_cycles: 10
    slate_size: 5
    seed: 20250513

data:
  directory: ../data/raw/ambar
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
    class_name: list_stochastic

  recommender_choice_model:
    class_name: threshold
    value: 0.1

provider:
  utility_model:
    class_name: click_fixed

platform:
  utility_model:
    class_name: null_model

recommenders:
  - name: Generic
    class_name: svd_generic

  - name: Niche
    class_name: niche

triggers:
  - name: Cycle5Freeze
    class_name: initial_burnin
    cycle_count: 5
'''


class ConfigTestCase(unittest.TestCase):
    def test_config_load(self):
        config_data = yaml.safe_load(SAMPLE_CONFIG1)
        config = SmoresConfig(**config_data)
        self.assertIsNotNone(config)


if __name__ == '__main__':
    unittest.main()