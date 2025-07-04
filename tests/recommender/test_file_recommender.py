import unittest
import yaml
from pathlib import Path

from smores.utils import SmoresConfig, PythonClassConfig
from smores.recommender import FileBasedRecommender
from smores import Smores

class FileBasedRecommenderTestCase(unittest.TestCase):
    def setUp(self):
        test_data_path = Path('tests/test_data')
        test_config_path = test_data_path / 'test_config.yaml'
        self.config = SmoresConfig.model_validate(yaml.safe_load(test_config_path.read_text()))
        self.smores = Smores(self.config)
        self.smores.setup()
        
        self.recommender = FileBasedRecommender()
        self.rec_config = PythonClassConfig(
            name="TestFileRec", 
            class_name="file_based", 
            params={'file_name': "popular_niche_movies.csv"}
        )
        self.test_file_path = Path('tests/test_data/popular_niche_movies.csv')
        
    def test_load_file(self):
        """Test file loading"""
        self.recommender.file_path = self.test_file_path
        self.recommender.load_items()
        
        self.assertGreater(len(self.recommender.items), 0)
        self.assertIn(1219, self.recommender.items)
        self.assertIn(3081, self.recommender.items)
        
    def test_with_mocked_check_items(self):
        """Test setup with mocked check_items"""
        original_check_items = self.recommender.check_items
        FileBasedRecommender.check_items = lambda self: self.items
        
        try:
            self.recommender.setup(self.rec_config)
            self.assertGreater(len(self.recommender.items), 0)
            self.assertIn(1219, self.recommender.items)
        finally:
            FileBasedRecommender.check_items = original_check_items
            
    def test_recommendations_with_mocked_check(self):
        """Test recommendations with mocked check_items"""
        original_check_items = self.recommender.check_items
        FileBasedRecommender.check_items = lambda self: self.items
        
        try:
            self.recommender.setup(self.rec_config)
            test_user_id = 999
            recs = self.recommender.get_recommendations(test_user_id)
            
            self.assertIsNotNone(recs)
            expected_size = min(self.smores.state.slate_size, len(self.recommender.items))
            self.assertEqual(len(recs), expected_size)
        finally:
            FileBasedRecommender.check_items = original_check_items

if __name__ == '__main__':
    unittest.main()