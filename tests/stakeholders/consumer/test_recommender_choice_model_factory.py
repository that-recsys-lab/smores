import unittest
from smores.stakeholders.consumer import RecommenderChoiceModelFactory
from smores.stakeholders.consumer.recommender_choice_model import (
    EpsilonGreedyRecommenderChoiceModel,
    FixedRecommenderChoiceModel,
    ThresholdRecommenderChoiceModel,
    UCBRecommenderChoiceModel,
)

class RecommenderChoiceModelFactoryTestCase(unittest.TestCase):
    def test_factory_registration(self):
        fixed_model = RecommenderChoiceModelFactory.create('fixed')
        self.assertIsInstance(fixed_model, FixedRecommenderChoiceModel)
        threshold_model = RecommenderChoiceModelFactory.create('threshold')
        self.assertIsInstance(threshold_model, ThresholdRecommenderChoiceModel)
        ucb_model = RecommenderChoiceModelFactory.create('ucb')
        self.assertIsInstance(ucb_model, UCBRecommenderChoiceModel)
        epsilon_greedy_model = RecommenderChoiceModelFactory.create('epsilon_greedy')
        self.assertIsInstance(epsilon_greedy_model, EpsilonGreedyRecommenderChoiceModel)
        
    def test_invalid_model_name(self):
        with self.assertRaises(Exception):
            RecommenderChoiceModelFactory.create('nonexistent_model')

if __name__ == '__main__':
    unittest.main()
