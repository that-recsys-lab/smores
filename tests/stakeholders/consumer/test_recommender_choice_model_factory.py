import unittest
from types import SimpleNamespace

import numpy as np

import smores
from smores.stakeholders.consumer import RecommenderChoiceModelFactory
from smores.stakeholders.consumer.recommender_choice_model import (
    EpsilonGreedyRecommenderChoiceModel,
    FixedRecommenderChoiceModel,
    ThresholdRecommenderChoiceModel,
    UCBRecommenderChoiceModel,
)

class RecommenderChoiceModelFactoryTestCase(unittest.TestCase):
    def make_preview_model(self, representations, preference, current="Generic", epsilon=0.0, seed=123):
        logged_choices = []
        recommenders = {
            name: SimpleNamespace(name=name, representation=representation)
            for name, representation in representations.items()
        }
        smores.Smores.state = SimpleNamespace(
            recommenders_base=SimpleNamespace(
                get_names=lambda: list(recommenders),
                get_recommender=lambda name: recommenders[name],
            ),
            recommenders_active=list(recommenders),
            rand=np.random.default_rng(seed),
            cycle_count=0,
            logger=SimpleNamespace(log_recommender_choice=logged_choices.append),
        )
        model = EpsilonGreedyRecommenderChoiceModel()
        model.setup(SimpleNamespace(params={"epsilon": epsilon, "beta": 2}))
        model.consumer = SimpleNamespace(
            id=1,
            type="Generic",
            preference_vector=preference,
            recommender=recommenders[current],
        )
        return model, logged_choices

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

    def test_epsilon_greedy_empty_representation_has_no_preview(self):
        original_state = getattr(smores.Smores, "state", None)
        try:
            recommender = SimpleNamespace(representation=[])
            recommenders = SimpleNamespace(get_recommender=lambda name: recommender)
            smores.Smores.state = SimpleNamespace(recommenders_base=recommenders)

            model = EpsilonGreedyRecommenderChoiceModel()
            model.consumer = SimpleNamespace(preference_vector=[0.2, 0.8])

            self.assertTrue(np.isnan(model.get_expected_utility("empty")))
        finally:
            smores.Smores.state = original_state

    def test_epsilon_greedy_mismatched_representation_has_no_preview(self):
        original_state = getattr(smores.Smores, "state", None)
        try:
            recommender = SimpleNamespace(representation=[1.0])
            recommenders = SimpleNamespace(get_recommender=lambda name: recommender)
            smores.Smores.state = SimpleNamespace(recommenders_base=recommenders)

            model = EpsilonGreedyRecommenderChoiceModel()
            model.consumer = SimpleNamespace(preference_vector=[0.2, 0.8])

            self.assertTrue(np.isnan(model.get_expected_utility("short")))
        finally:
            smores.Smores.state = original_state

    def test_epsilon_greedy_keeps_current_recommender_when_its_preview_is_highest(self):
        original_state = getattr(smores.Smores, "state", None)
        try:
            model, _ = self.make_preview_model(
                {"Generic": [1.0, 0.0], "Niche": [0.0, 1.0]},
                preference=[1.0, 0.0],
            )
            self.assertEqual(model.choose_recommender(), "Generic")
        finally:
            smores.Smores.state = original_state

    def test_epsilon_greedy_selects_highest_preview_across_all_active_recommenders(self):
        original_state = getattr(smores.Smores, "state", None)
        try:
            model, logged_choices = self.make_preview_model(
                {
                    "Generic": [1.0, 0.0],
                    "Lower niche": [0.25, 0.75],
                    "Best niche": [0.0, 1.0],
                },
                preference=[0.0, 1.0],
            )
            self.assertEqual(model.choose_recommender(), "Best niche")
            self.assertEqual(logged_choices[-1].utilities, [0.0, 0.75, 1.0])
        finally:
            smores.Smores.state = original_state

    def test_epsilon_greedy_keeps_current_recommender_when_tied_for_best(self):
        original_state = getattr(smores.Smores, "state", None)
        try:
            model, _ = self.make_preview_model(
                {"Niche": [1.0, 0.0], "Generic": [1.0, 0.0]},
                preference=[1.0, 0.0],
            )
            self.assertEqual(model.choose_recommender(), "Generic")
        finally:
            smores.Smores.state = original_state

    def test_epsilon_greedy_ignores_missing_previews(self):
        original_state = getattr(smores.Smores, "state", None)
        try:
            model, _ = self.make_preview_model(
                {"Generic": [1.0, 0.0], "Unavailable": []},
                preference=[1.0, 0.0],
            )
            self.assertEqual(model.choose_recommender(), "Generic")
        finally:
            smores.Smores.state = original_state

    def test_epsilon_greedy_rejects_invalid_epsilon(self):
        model = EpsilonGreedyRecommenderChoiceModel()
        with self.assertRaisesRegex(ValueError, "epsilon must be between 0 and 1"):
            model.setup(SimpleNamespace(params={"epsilon": 1.1, "beta": 2}))

if __name__ == '__main__':
    unittest.main()
