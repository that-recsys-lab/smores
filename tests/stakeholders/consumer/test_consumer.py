import unittest
import yaml
from pathlib import Path

from smores.utils import SmoresConfig
from smores import Smores

from icecream import ic

from smores.stakeholders.consumer import Consumer, ConsumerCollection, ConsumerInfo

TEST_CONSUMER_FILE = "consumers.csv"


class ConsumerTestCase(unittest.TestCase):
    def setUp(self):
        test_data_path = Path('tests/test_data')
        test_config_path = test_data_path / 'test_config.yaml'
        self.config = SmoresConfig.model_validate(yaml.safe_load(test_config_path.read_text()))
        self.smores = Smores(self.config)
        self.smores.setup()

        self.consumer_data_path = test_data_path / TEST_CONSUMER_FILE

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
        ccoll.load_consumers(self.consumer_data_path)
        self.assertEqual(len(list(ccoll)), 3)


if __name__ == '__main__':
    unittest.main()