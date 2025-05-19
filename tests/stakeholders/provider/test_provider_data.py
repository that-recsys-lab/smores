import unittest

from smores.stakeholders.provider.provider_data import ProviderData, BadRecommenderListError

SAMPLE_PROVIDER_DATA = \
'''
10, niche, all
11, general, all
12, general, all
13, niche, all
'''

BAD_PROVIDER_DATA = \
'''
10, niche, all
11, general, all
12, general, all
13, niche, foo
'''

BAD_PROVIDER_DATA2 = \
'''
prov10, niche, all
11, general, all
12, general, all
13, niche, foo
'''


class ProviderDataTestCase(unittest.TestCase):
    def test_correct_data(self):
        data = ProviderData(SAMPLE_PROVIDER_DATA.splitlines())
        row1 = data.__next__()
        self.assertEqual(row1[0], 10)
        self.assertEqual(row1[1], "niche")
        self.assertEqual(row1[2], ['*'])

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




if __name__ == '__main__':
    unittest.main()
