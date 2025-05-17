import unittest
from smores.stakeholders.consumer import ConsumerUtilityModelFactory
from smores.stakeholders.consumer.consumer_utility_model import ConsumerFixedUtilityModel, ConsumerPrefCosineAvgUtilityModel

class UtilityModelFactoryTestCase(unittest.TestCase):
    def test_factory_registration(self):
        fixed_model = ConsumerUtilityModelFactory.get_class('fixed')
        self.assertEqual(fixed_model, ConsumerFixedUtilityModel)
        avg_model = ConsumerUtilityModelFactory.get_class('list_average')
        self.assertEqual(avg_model, ConsumerPrefCosineAvgUtilityModel)
        
    def test_invalid_model_name(self):
        with self.assertRaises(Exception):
            ConsumerUtilityModelFactory.get_class('nonexistent_model')

if __name__ == '__main__':
    unittest.main()