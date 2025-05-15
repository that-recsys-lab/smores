import unittest
import yaml
from smores import SmoresConfig

from icecream import ic

from smores.stakeholders.consumer import Consumer, ConsumerUtilityModelFactory

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

  selection_model:
    class_name: list_stochastic

  choice_model:
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


class ConsumerTestCase(unittest.TestCase):
    def test_utility_model_creation(self):
        config_raw = yaml.safe_load(SAMPLE_CONFIG1)
        config = SmoresConfig(**config_raw)
        consumer = Consumer()
        consumer.setup(config)
        self.assertIsNotNone(consumer.utility_model)


if __name__ == '__main__':
    unittest.main()
