import unittest
from smores.stakeholders.consumer import ConsumerUtilityModelFactory
from smores.stakeholders.consumer.consumer_utility_model import ConsumerFixedUtilityModel, ConsumerPrefCosineAvgUtilityModel

class UtilityModelFactoryTestCase(unittest.TestCase):
    def test_factory_registration(self):
        fixed_model = ConsumerUtilityModelFactory.create('fixed_utility')
        self.assertIsInstance(fixed_model, ConsumerFixedUtilityModel)
        avg_model = ConsumerUtilityModelFactory.create('list_average')
        self.assertIsInstance(avg_model, ConsumerPrefCosineAvgUtilityModel)
        
    def test_invalid_model_name(self):
        with self.assertRaises(Exception):
            ConsumerUtilityModelFactory.create('nonexistent_model')

if __name__ == '__main__':
    unittest.main()