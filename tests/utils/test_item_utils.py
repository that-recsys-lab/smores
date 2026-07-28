import unittest
from lenskit.data import ItemList

from smores.utils import itemList2rankedTuples

class ItemUtilsTestCase(unittest.TestCase):
    def test_item_list_convert(self):
        item_list = ItemList(item_ids=[13, 12, 11, 14, 15], 
                             scores=[0.3, 0.2, 0.1, 0.4, 0.5],
                             rank=[3, 2, 1, 4, 5], ordered=True)
        tuples = itemList2rankedTuples(item_list)
        self.assertEqual(tuples[0][0], 11)
        self.assertAlmostEqual(tuples[0][1], 0.1, 3)

    def test_item_list_missing_scores_defaults_to_one(self):
        item_list = ItemList(item_ids=[1, 2, 3], scores=None, rank=[2, 3, 1], ordered=True)
        tuples = itemList2rankedTuples(item_list)
        self.assertEqual(tuples[0], (3, 1))
        self.assertTrue(all(score == 1 for _, score in tuples))

if __name__ == '__main__':
    unittest.main()
        
