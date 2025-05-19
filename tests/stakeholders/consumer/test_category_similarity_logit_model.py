import unittest
import numpy as np
from unittest.mock import MagicMock

from smores.stakeholders.consumer.category_similarity_logit_model import CategorySimilarityLogitModel


class CategorySimilarityLogitModelBehaviorTestCase(unittest.TestCase):
    
    def test_prohibited_genres(self):
        np.random.seed(20250517)
        test_items = []
        for i in range(3):
            item = MagicMock()
            item.genres = [f"genre{i}", "common_genre"]
            test_items.append(item)
            
        prohibited_item = MagicMock()
        prohibited_item.genres = ["prohibited_genre", "common_genre"]
        test_items.append(prohibited_item)
        
        category_preferences = {
            "genre0": 0.5, "genre1": 0.5, "genre2": 0.5,
            "common_genre": 0.5, "prohibited_genre": 0.9
        }
        
        prohibited_genres = {"prohibited_genre"}
        
        selected_index = CategorySimilarityLogitModel.select_item(
            test_items,
            threshold=0.2,
            category_preferences=category_preferences,
            prohibited_genres=prohibited_genres
        )
        
        self.assertNotEqual(selected_index, 3)
    
    def test_preference_based_selection(self):
        np.random.seed(20250517)
        test_items = []
        for i in range(3):
            item = MagicMock()
            item.genres = [f"genre{i}"]
            test_items.append(item)
        
        high_preference = {
            "genre0": 0.9,
            "genre1": 0.1,
            "genre2": 0.1
        }
        
        equal_preference = {
            "genre0": 0.5,
            "genre1": 0.5,
            "genre2": 0.5
        }
        
        high_pref_selection = CategorySimilarityLogitModel.select_item(
            test_items,
            threshold=0.2,
            category_preferences=high_preference,
            prohibited_genres=set()
        )
        
        self.assertEqual(high_pref_selection, 0)
        
        selections = []
        for _ in range(50):
            idx = CategorySimilarityLogitModel.select_item(
                test_items,
                threshold=0.2,
                category_preferences=equal_preference,
                prohibited_genres=set()
            )
            selections.append(idx)
        
        counts = {0: 0, 1: 0, 2: 0}
        for idx in selections:
            if idx is not None:
                counts[idx] += 1
        
        self.assertGreater(counts[0], 0)
        self.assertGreater(counts[1], 0) 
        self.assertGreater(counts[2], 0)
    
    def test_threshold_behavior(self):
        test_items = []
        for i in range(3):
            item = MagicMock()
            item.genres = [f"genre{i}"]
            test_items.append(item)
        
        low_preferences = {
            "genre0": 0.1,
            "genre1": 0.1, 
            "genre2": 0.1
        }
        
        high_threshold_result = CategorySimilarityLogitModel.select_item(
            test_items,
            threshold=5.0,
            category_preferences=low_preferences,
            prohibited_genres=set()
        )
        
        self.assertIsNone(high_threshold_result)


if __name__ == '__main__':
    unittest.main()