import unittest
import yaml
from pathlib import Path
from lenskit.data.items import ItemList

from smores.utils import itemList2rankedTuples, itemListLen

class ItemUtilsTestCase(unittest.TestCase):
    def test_item_list_convert(self):
        item_list = ItemList(item_ids=[13, 12, 11, 14, 15], 
                             scores=[0.3, 0.2, 0.1, 0.4, 0.5],
                             rank=[3, 2, 1, 4, 5], ordered=True)
        tuples = itemList2rankedTuples(item_list)
        self.assertEqual(tuples[0][0], 11)
        self.assertAlmostEqual(tuples[0][1], 0.1, 3)

    def test_item_list_len(self):
        item_list = ItemList(item_ids=[13, 12, 11, 14, 15], 
                             scores=[0.3, 0.2, 0.1, 0.4, 0.5],
                             rank=[3, 2, 1, 4, 5], ordered=True)
        self.assertEqual(itemListLen(item_list), 5)

if __name__ == '__main__':
    unittest.main()
        
