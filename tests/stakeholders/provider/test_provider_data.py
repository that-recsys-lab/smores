import unittest
from pathlib import Path

from smores.stakeholders.provider.provider_data import ProviderData, BadRecommenderListError


class ProviderDataTestCase(unittest.TestCase):
    def setUp(self):
        test_data_path = Path('tests/test_data')
        self.provider_path = test_data_path / 'providers01.csv'

    def test_correct_data(self):
        with open(self.provider_path, 'r') as csvfile:
            data = ProviderData(csvfile)
            row1 = data.__next__()
            self.assertEqual(row1[0], 10)
            self.assertEqual(row1[1], "niche")
            self.assertEqual(row1[2], ['*'])

'''
Will be fixed with new implementation of ProviderData class
    def test_bad_rec_spec(self):
        data = ProviderData(BAD_PROVIDER_DATA.splitlines())
        with self.assertRaises(BadRecommenderListError):
            for _ in data:
                pass

    def test_non_int_id(self):
        data = ProviderData(BAD_PROVIDER_DATA2.splitlines())
        with self.assertRaises(Exception):
            row1 = data.__next__()
            print(row1)
'''



if __name__ == '__main__':
    unittest.main()
