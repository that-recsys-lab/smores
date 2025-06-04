import unittest
import yaml
import tempfile
import pathlib

from smores.utils import SmoresConfig
from smores import Smores

from icecream import ic

from smores.stakeholders.consumer import Consumer, ConsumerCollection, ConsumerInfo

SAMPLE_CONFIG1 = \
'''
simulation:
    experiment_name: test_experiment
    num_days: 10
    num_cycles: 10
    slate_size: 5
    seed: 20250513

data:
  directory: data/raw/ambar
  consumer_file: users.csv
  item_file: items.csv
  provider_file: artists.csv

consumer:
  models:
      utility:
        - name: "Fixed utility 0.5"
          class_name: fixed_utility
          params:
            value: 0.5
        - name: "Fixed utility 0.3"
          class_name: fixed_utility
          params:
            value: 0.3
      item_selection:
        - name: "Category Similarity"
          class_name: category_similarity_logit
          params:
            threshold: 0.3        
  types:
    - name: "Generic"
      utility_model: "Fixed utility 0.5"
      item_selection_model: "Category Similarity"
      recommender_choice_model:
        class_name: fixed
        params:
          recommender_name: Generic

    - name: "Niche"
      utility_model: "Fixed utility 0.3"
      item_selection_model: "Category Similarity"
      recommender_choice_model:
        class_name: fixed
        params:
          recommender_name: Generic


provider:
  utility_model:
    class_name: click_fixed
    value: 1

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

CONSUMER_DATA = '''consumer_id,consumer_type,preferences
100,Niche,"[0.1, 0.63, 0.3, 0.2, 0.0]"
101,Niche,"[0.1, 0.63, 0.3, 0.2, 0.0]"
102,Generic,"[0.3, 0.01, 0.6, 0.1, 0.8]"
'''

TEST_CONSUMER_FILE = "test_consumer.csv"


class ConsumerTestCase(unittest.TestCase):
    def setUp(self):
        config_raw = yaml.safe_load(SAMPLE_CONFIG1)
        self.config = SmoresConfig(**config_raw)
        self.smores = Smores(self.config)
        self.smores.setup()

        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        # Get the path to the temporary directory
        self.temp_dir_path = pathlib.Path(self.temp_dir.name)

        self.file_path = self.temp_dir_path / TEST_CONSUMER_FILE
        with open(self.file_path, 'w') as feature_file:
            feature_file.write(CONSUMER_DATA)

    def test_component_creation(self):
        ccoll = ConsumerCollection()
        ccoll.setup(self.config.consumer.types)
        consumer_config: ConsumerInfo = ConsumerInfo.model_validate({"consumer_id": 100,
                                                                     "consumer_type": "Generic",
                                                                     "preferences": [0.1, 0.9]})
        consumer_type_config = ccoll.types["Generic"]

        consumer = Consumer()
        consumer.setup(consumer_type_config, consumer_config)

        self.assertIsNotNone(consumer.utility_model)
        self.assertIsNotNone(consumer.item_selection_model)
        self.assertIsNotNone(consumer.recommender_choice_model)

    def test_consumer_loading(self):
        ccoll = ConsumerCollection()
        ccoll.setup(self.config.consumer.types)
        ccoll.load_consumers(self.file_path)

    def tearDown(self):
        # Delete the temporary directory and all its contents
        self.temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()