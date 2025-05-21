import unittest
from smores.utils import UtilityHistory, UtilityHistoryEntry

# Courtesy ChatGPT

class TestUtilityHistory(unittest.TestCase):
    def setUp(self):
        self.history = UtilityHistory()

    def test_initialization(self):
        self.assertEqual(self.history.get_history(), [])

    def test_add_entry(self):
        self.history.add_entry(1, "recA", 0.5)
        self.assertEqual(len(self.history.get_history()), 1)
        entry = self.history.get_history()[0]
        self.assertIsInstance(entry, UtilityHistoryEntry)
        self.assertEqual(entry.time, 1)
        self.assertEqual(entry.recommender, "recA")
        self.assertEqual(entry.value, 0.5)

    def test_get_history(self):
        self.history.add_entry(1, "recA", 0.5)
        self.history.add_entry(2, "recB", 0.7)
        entries = self.history.get_history()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].recommender, "recA")
        self.assertEqual(entries[1].recommender, "recB")

    def test_get_history_for_recommender(self):
        self.history.add_entry(1, "recA", 0.5)
        self.history.add_entry(2, "recB", 0.7)
        self.history.add_entry(3, "recA", 0.9)
        recA_entries = self.history.get_history_for_recommender("recA")
        self.assertEqual(len(recA_entries), 2)
        for entry in recA_entries:
            self.assertEqual(entry.recommender, "recA")
        recB_entries = self.history.get_history_for_recommender("recB")
        self.assertEqual(len(recB_entries), 1)
        self.assertEqual(recB_entries[0].recommender, "recB")

    def test_get_history_for_recommender_no_match(self):
        self.history.add_entry(1, "recA", 0.5)
        entries = self.history.get_history_for_recommender("recX")
        self.assertEqual(entries, [])

if __name__ == "__main__":
    unittest.main()