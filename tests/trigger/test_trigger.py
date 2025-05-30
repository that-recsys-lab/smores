import unittest
import yaml
import csv

from smores.utils import SmoresConfig
from smores.trigger import TriggerFactory, InitialBurnInTrigger, TimeTriggerEvent
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
    name: "Generic"

  utility_model:
    class_name: list_average

  item_selection_model:
    class_name: category_similarity_logit
    threshold: 0.2

  recommender_choice_model:
    class_name: fixed
    recommender_name: "Generic"

provider:
  utility_model:
    class_name: click_fixed

platform:
  utility_model:
    class_name: null_model

recommender:
  initial: ["Generic"]
  definitions:
    - name: "Generic"
      class_name: item_knn
      params:
        max_neighbors: 20
        min_neighbors: 2
        min_similarity: 0.0001
        min_user_count: 20
        min_interaction_count: 500
        min_profile_size: 5
        cold_user_fallback: "Popular Fallback"

    - name: "Popular Fallback"
      class_name: popular
      params:
          min_user_count: 20
          min_interaction_count: 500

    - name: "Popular Niche"
      class_name: popular
      params:
          min_user_count: 20
          min_interaction_count: 500

triggers:
  - name: Cycle5Freeze
    class_name: initial_burnin
    params:
      cycle_count: 5
      recommenders: ["Generic", "Popular Niche"]
'''


class TestTrigger(unittest.TestCase):
    def setUp(self):
        self.config = SmoresConfig.model_validate(yaml.safe_load(SAMPLE_CONFIG1))
        self.smores = Smores(self.config)

    def test_trigger_factory(self):
        trigger = TriggerFactory.make_object('initial_burnin')
        self.assertIsInstance(trigger, InitialBurnInTrigger)

    def test_initial_burnin_trigger(self):
        trigger = InitialBurnInTrigger()
        trigger_config = self.config.triggers[0]
        trigger.setup(trigger_config)

        self.assertEqual(trigger.cycle_count, 5)

        tev1 = TimeTriggerEvent(0)
        self.assertFalse(trigger.accept_event(tev1))

        tev2 = TimeTriggerEvent(10)
        self.assertFalse(trigger.accept_event(tev2))

                         


if __name__ == '__main__':
    unittest.main()