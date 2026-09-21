import io
import unittest
from contextlib import redirect_stdout

from smores.utils import SummaryLogger


class SummaryLoggerTestCase(unittest.TestCase):
    def test_sampled_ratio_is_the_share_of_delivered_items(self):
        logger = SummaryLogger()
        logger.log_recommendation(
            "rec",
            user_id=1,
            slate_items=[10, 11, 12, 13],
            slate_size=4,
            sampled_count=1,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            logger.print_summary()

        self.assertIn("sampled_ratio=0.25", output.getvalue())

    def test_sampled_ratio_is_zero_when_no_items_are_delivered(self):
        logger = SummaryLogger()
        logger.log_recommendation(
            "rec",
            user_id=1,
            slate_items=[],
            slate_size=4,
            sampled_count=0,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            logger.print_summary()

        self.assertIn("sampled_ratio=0.00", output.getvalue())
