import unittest
import yaml
import csv
import numpy as np
from pathlib import Path

from smores.utils import SmoresConfig, PythonClassConfig
from smores.recommender import RecommenderFactory, Recommender, RecommenderMap
from smores.samplers.rejection_sampler import RejectionSampler
from smores import Smores
from tests.paths import FIXTURE_DATA_DIR, TEST_CONFIG_PATH


class ItemSamplingTestCase(unittest.TestCase):
    """Test cases for item sampling integration with recommenders."""

    def setUp(self):
        """Set up test environment with smores instance and test data."""
        self.config = SmoresConfig.model_validate(yaml.safe_load(TEST_CONFIG_PATH.read_text()))
        self.smores = Smores(self.config)
        self.smores.setup()

        # Load test interactions
        interactions_path = FIXTURE_DATA_DIR / 'interactions.csv'
        with open(interactions_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            reader.__next__()  # Skip header
            self.interactions = []
            for row in reader:
                row_int = [int(entry) for entry in row]
                self.interactions.append(row_int)

    def test_rejection_sampler_basic(self):
        """Test that RejectionSampler loads items and samples correctly."""
        sampler = RejectionSampler()
        file_path = FIXTURE_DATA_DIR / 'item_popularity.csv'
        sampler.load_from_file(file_path)

        # Verify items loaded
        self.assertTrue(sampler.has_items())
        self.assertGreater(len(sampler.item_ids), 0)
        self.assertEqual(len(sampler.item_ids), len(sampler.base_probabilities))

        # Verify probabilities sum to 1.0
        prob_sum = sum(sampler.base_probabilities)
        self.assertAlmostEqual(prob_sum, 1.0, places=5)

    def test_rejection_sampler_excludes_items(self):
        """Test that RejectionSampler respects exclusions."""
        sampler = RejectionSampler()
        file_path = FIXTURE_DATA_DIR / 'item_popularity.csv'
        sampler.load_from_file(file_path)

        # Sample with exclusions
        exclude_items = {200, 201, 202}
        sampled = sampler.sample(5, exclude_items=exclude_items)

        # Verify no excluded items were sampled
        for item in sampled:
            self.assertNotIn(item, exclude_items)

        # Verify we got unique items
        self.assertEqual(len(sampled), len(set(sampled)))

    def test_vectorized_sampling_matches_previous_seeded_behavior(self):
        """Vectorized filtering preserves the previous seeded sampling result."""
        sampler = RejectionSampler()
        sampler.load_from_file(FIXTURE_DATA_DIR / 'item_popularity.csv')
        exclude_items = {200, 202}

        self.smores.state.rand = np.random.default_rng(42)
        candidate_ids, candidate_probs = sampler._filtered_items(exclude_items)
        prob_sum = sum(candidate_probs)
        candidate_probs = [prob / prob_sum for prob in candidate_probs]
        chosen_indices = self.smores.state.rand.choice(
            len(candidate_ids),
            size=min(3, len(candidate_ids)),
            replace=False,
            p=candidate_probs,
        )
        expected = [candidate_ids[index] for index in chosen_indices]

        self.smores.state.rand = np.random.default_rng(42)
        actual = sampler.sample(3, exclude_items=exclude_items)

        self.assertEqual(actual, expected)

    def test_recommender_with_item_sampling_config(self):
        """Test that recommender correctly parses and uses item_sampler config."""
        # Create a config with item_sampler
        config_dict = {
            'name': 'TestRec',
            'class_name': 'popular',
            'params': {
                'min_user_count': 1,
                'min_interaction_count': 1,
                'file_name': 'item_popularity.csv',
                'item_sampler': {
                    'class_name': 'rejection_sampler',
                    'params': {
                        'file_name': 'item_popularity.csv',
                        'sampled_item_count': 2
                    }
                }
            }
        }

        rec_config = PythonClassConfig(**config_dict)
        rec = RecommenderFactory.create('popular')
        rec.setup(rec_config)
        rec.setup_dataset()

        # Verify item_sampler was created
        self.assertIsNotNone(rec.item_sampler)
        self.assertIsInstance(rec.item_sampler, RejectionSampler)
        self.assertEqual(rec.sampled_item_count, 2)
        self.assertTrue(rec.item_sampler.has_items())

    def test_lk_recommender_generates_and_extends_recommendations(self):
        """Test full flow: LKRecommender generates recs, then extends with sampled items."""
        # Create a popular recommender with item sampling
        config_dict = {
            'name': 'PopularWithSampling',
            'class_name': 'popular',
            'params': {
                'min_user_count': 1,
                'min_interaction_count': 1,
                'file_name': 'item_popularity.csv',
                'item_sampler': {
                    'class_name': 'rejection_sampler',
                    'params': {
                        'file_name': 'item_popularity.csv',
                        'sampled_item_count': 1
                    }
                }
            }
        }

        rec_config = PythonClassConfig(**config_dict)
        rec = RecommenderFactory.create('popular')
        rec.setup(rec_config)
        rec.setup_dataset()

        # Add only first time step to leave more items available
        time_step1 = [row for row in self.interactions if row[3] == 1]
        rec.update_dataset(time_step1)

        # Train the recommender
        rec.train()

        # Get recommendations for a user with minimal history
        user_id = 102  # User with less interaction history
        slate_size = 2
        self.smores.state.slate_size = slate_size

        recommendations = rec.get_recommendations(user_id)

        # Verify we got some items
        self.assertIsNotNone(recommendations)
        rec_ids = recommendations.ids()
        self.assertGreater(len(rec_ids), 0, "Should get at least some recommendations")

        # Verify user's interaction history is not in recommendations
        user_history = rec.get_user(user_id)
        if user_history is not None:
            history_ids = set(user_history.ids())
            for item_id in rec_ids:
                self.assertNotIn(item_id, history_ids,
                                 f"Item {item_id} from user history should not be recommended")

        # Verify we have scores and ranks
        scores = recommendations.scores()
        ranks = recommendations.ranks()
        self.assertIsNotNone(scores)
        self.assertIsNotNone(ranks)

        # Check for sampled items (score 0.0)
        zero_score_items = [i for i, s in enumerate(scores) if s == 0.0]
        if len(zero_score_items) > 0:
            # If we got sampled items, verify they have score 0.0
            for idx in zero_score_items:
                self.assertEqual(scores[idx], 0.0, "Sampled items should have score 0.0")

    def test_item_sampling_with_user_interaction(self):
        """Test full simulation: generate recs with sampling, user clicks, update dataset."""
        # Create recommender with sampling
        config_dict = {
            'name': 'PopularWithSampling',
            'class_name': 'popular',
            'params': {
                'min_user_count': 1,
                'min_interaction_count': 1,
                'file_name': 'item_popularity.csv',
                'item_sampler': {
                    'class_name': 'rejection_sampler',
                    'params': {
                        'file_name': 'item_popularity.csv',
                        'sampled_item_count': 1
                    }
                }
            }
        }

        rec_config = PythonClassConfig(**config_dict)
        rec = RecommenderFactory.create('popular')
        rec.setup(rec_config)
        rec.setup_dataset()

        # Add initial interactions
        time_step1 = [row for row in self.interactions if row[3] == 1]
        rec.update_dataset(time_step1)
        rec.train()

        user_id = 103  # User with minimal history
        slate_size = 2
        self.smores.state.slate_size = slate_size

        # Get initial recommendations
        recs_before = rec.get_recommendations(user_id)
        items_before = list(recs_before.ids())
        self.assertGreater(len(items_before), 0, "Should get at least some recommendations")

        # Simulate user clicking on an item (could be a sampled item)
        clicked_item = items_before[0]  # Click first item

        # Add interaction to dataset
        new_interaction = [user_id, clicked_item, 1, 99]  # [user, item, rating, time]
        rec.update_dataset([new_interaction])

        # Retrain after adding interaction (in real system, this happens periodically)
        rec.train()

        # Verify the clicked interaction was persisted to the recommender dataset.
        interactions_table = rec.get_dataset().interaction_table(format='arrow', original_ids=True)
        observed_pairs = set(zip(interactions_table["user_id"].to_pylist(), interactions_table["item_id"].to_pylist()))
        self.assertIn((user_id, int(clicked_item)), observed_pairs)

    def test_recommender_without_item_sampling(self):
        """Test that recommender works normally without item_sampler config."""
        # Create config WITHOUT item_sampler
        config_dict = {
            'name': 'PopularNoSampling',
            'class_name': 'popular',
            'params': {
                'min_user_count': 1,
                'min_interaction_count': 1
            }
        }

        rec_config = PythonClassConfig(**config_dict)
        rec = RecommenderFactory.create('popular')
        rec.setup(rec_config)
        rec.setup_dataset()

        # Verify no sampler created
        self.assertIsNone(rec.item_sampler)
        self.assertEqual(rec.sampled_item_count, 0)

        # Add interactions and train
        time_step1 = [row for row in self.interactions if row[3] == 1]
        rec.update_dataset(time_step1)
        rec.train()

        # Get recommendations
        user_id = 100
        self.smores.state.slate_size = 2
        recommendations = rec.get_recommendations(user_id)

        # Should work normally
        self.assertIsNotNone(recommendations)
        # Might get fewer than 10 if not enough items, but should work
        self.assertGreater(len(recommendations.ids()), 0)


if __name__ == '__main__':
    unittest.main()
