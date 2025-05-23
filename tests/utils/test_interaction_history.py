import unittest
import pyarrow as pa
from smores.utils.interaction_history import InteractionHistory

SAMPLE_INTERACTIONS = [
    [100,200,1,1],
    [101,201,1,1],
    [102,202,1,1],
    [100,210,1,2],
    [101,211,1,2],
    [102,212,1,2]]


class TestInteractionHistory(unittest.TestCase):
    def setUp(self):
        self.ih = InteractionHistory()

    def test_initial_state(self):
        self.assertIsNone(self.ih.interaction_table)

    def test_add_interaction(self):
        self.ih.add_interaction(*SAMPLE_INTERACTIONS[0])
        self.assertIsNotNone(self.ih.interaction_table)
        # Should have one row
        self.assertEqual(self.ih.interaction_table.num_rows, 1)

    def test_add_interactions(self):

        self.ih.add_interactions(SAMPLE_INTERACTIONS[:3])
        self.assertIsNotNone(self.ih.interaction_table)
        self.assertEqual(self.ih.interaction_table.num_rows, 3)

    def test_to_dataset(self):
        self.ih.add_interactions(SAMPLE_INTERACTIONS[:3])
        dataset = self.ih.to_dataset()
        self.assertIsNotNone(dataset)
        # Dataset should have at least the interactions we added
        self.assertTrue(hasattr(dataset, "interactions"))
        self.assertEqual(dataset.interaction_count, 3)

if __name__ == "__main__":
    unittest.main()

