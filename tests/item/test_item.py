import unittest
import tempfile
import pathlib

from smores.utils import SmoresConfig
from smores.item import Item, ItemMap, ItemCollection
from smores import Smores

from icecream import ic

ITEM_DATA = '''item_id,provider_id,features
200,300,"[0.2,0.4,0.0,0.35,0.05]"
201,300,"[0.1,0.1,0.8,0.0,0.0]"
202,300,"[0.2,0.0,0.0,0.8,0.0]"
203,301,"[0.1,0.0,0.0,0.0,0.9]"
204,301,"[0.1,0.1,0.0,0.0,0.8]"
'''

TEST_ITEM_FILE = 'item_test.csv'

class ItemTestCase(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        # Get the path to the temporary directory
        self.temp_dir_path = pathlib.Path(self.temp_dir.name)

        self.file_path = self.temp_dir_path / TEST_ITEM_FILE
        with open(self.file_path, 'w') as feature_file:
            feature_file.write(ITEM_DATA)

    def testItemLoad(self):
        item_map = ItemMap()
        item_map.load_items(self.file_path)
        items = item_map.all_items()
        self.assertEqual(len(items), 5)
        item201 = item_map.get_item(201)
        self.assertIsInstance(item201, Item)
        self.assertEqual(item201.provider_id, 300)

if __name__ == '__main__':
    unittest.main()