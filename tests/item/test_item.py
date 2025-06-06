import unittest
import pathlib

from smores.utils import SmoresConfig
from smores.item import Item, ItemMap, ItemCollection
from smores import Smores

from icecream import ic


TEST_ITEM_FILE = 'items.csv'

class ItemTestCase(unittest.TestCase):
    def setUp(self):
        test_data_path = pathlib.Path('tests/test_data')
        self.item_data_path = test_data_path / TEST_ITEM_FILE


    def testItemLoad(self):
        item_map = ItemMap()
        item_map.load_items(self.item_data_path)
        items = item_map.all_items()
        self.assertEqual(len(items), 5)
        item201 = item_map.get_item(201)
        self.assertIsInstance(item201, Item)
        self.assertEqual(item201.provider_id, 300)


if __name__ == '__main__':
    unittest.main()