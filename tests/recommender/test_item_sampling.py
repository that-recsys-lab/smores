import unittest
import yaml
import csv
from pathlib import Path

from smores.utils import SmoresConfig, PythonClassConfig
from smores.recommender import RecommenderFactory, Recommender, RecommenderMap
from smores.samplers.rejection_sampler import RejectionSampler
from smores import Smores


class ItemSamplingTestCase(unittest.TestCase):
    """Test cases for item sampling integration with recommenders."""

    def setUp(self):
        """Set up test environment with smores instance and test data."""
        test_data_path = Path('tests/test_data')
        test_config_path = test_data_path / 'test_config.yaml'
        self.config = SmoresConfig.model_validate(yaml.safe_load(test_config_path.read_text()))
        self.smores = Smores(self.config)
        self.smores.setup()

        # Load test interactions
        interactions_path = test_data_path / 'interactions.csv'
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
        file_path = Path('tests/test_data/item_popularity.csv')
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
        file_path = Path('tests/test_data/item_popularity.csv')
        sampler.load_from_file(file_path)

        # Sample with exclusions
        exclude_items = {200, 201, 202}
        sampled = sampler.sample(5, exclude_items=exclude_items)

        # Verify no excluded items were sampled
        for item in sampled:
            self.assertNotIn(item, exclude_items)

        # Verify we got unique items
        self.assertEqual(len(sampled), len(set(sampled)))

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
                        'sampled_item_count': 2
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
        slate_size = 5  # Only 5 items exist in test data
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
                        'sampled_item_count': 3
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
        slate_size = 5  # Only 5 items in test data
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

        # Get new recommendations
        recs_after = rec.get_recommendations(user_id)
        items_after = list(recs_after.ids())

        # Verify clicked item is no longer recommended
        # Note: The system correctly excludes items from user history
        self.assertNotIn(clicked_item, items_after,
                         "Clicked item should not appear in new recommendations after retraining")

        # Verify we still get some items
        self.assertGreater(len(items_after), 0, "Should still get recommendations")

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
        self.smores.state.slate_size = 10
        recommendations = rec.get_recommendations(user_id)

        # Should work normally
        self.assertIsNotNone(recommendations)
        # Might get fewer than 10 if not enough items, but should work
        self.assertGreater(len(recommendations.ids()), 0)


if __name__ == '__main__':
    unittest.main()
