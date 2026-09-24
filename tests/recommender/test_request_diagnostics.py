from __future__ import annotations

import csv
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
from lenskit.data import ItemList

import smores
from smores.recommender.lk_recommenders import LKRecommender
from smores.smores import Smores
from smores.utils.logger import ConsumerUtility, SmoresLogger


class _FallbackRecommender:
    _last_sampled_count = 0
    _last_candidate_count_after_filters = 3

    def get_recommendations(self, _user_id):
        return ItemList(None, item_ids=[10, 11], scores=[1.0, 1.0], rank=[1, 2])


class _StubLKRecommender(LKRecommender):
    def __init__(self, dataset_viable: bool, profile_viable: bool, profile_size: int):
        super().__init__()
        self.dataset_viable = dataset_viable
        self.profile_viable = profile_viable
        self.profile_size = profile_size
        self.cold_start_fallback = 'cold_start'
        self.cold_user_fallback = 'cold_user'

    def build_pipeline(self):
        return None

    def train(self):
        return None

    def isDatasetViable(self):
        return self.dataset_viable

    def isProfileViable(self, _user_id):
        return self.profile_viable

    def get_user_item_ids(self, _user_id):
        return frozenset(range(self.profile_size))


class RequestDiagnosticsTestCase(unittest.TestCase):
    def setUp(self):
        self.original_state = getattr(smores.Smores, 'state', None)

    def tearDown(self):
        smores.Smores.state = self.original_state

    def test_fallback_reason_and_candidate_count(self):
        fallback = _FallbackRecommender()
        smores.Smores.state = SimpleNamespace(
            recommenders_fallback=SimpleNamespace(get_recommender=lambda _name: fallback)
        )

        cases = [
            (False, True, 'dataset_not_viable'),
            (True, False, 'profile_too_small'),
        ]
        for dataset_viable, profile_viable, expected_reason in cases:
            with self.subTest(reason=expected_reason):
                recommender = _StubLKRecommender(dataset_viable, profile_viable, profile_size=2)
                result = recommender.get_recommendations(user_id=7)

                self.assertEqual(len(result), 2)
                self.assertTrue(recommender._last_used_fallback)
                self.assertEqual(recommender._last_fallback_reason, expected_reason)
                self.assertEqual(len(recommender.get_user_item_ids(7)), 2)
                self.assertEqual(recommender._last_candidate_count_after_filters, 3)

    def test_request_rows_log_full_short_and_empty_slates(self):
        class FakeRecommender:
            name = 'test_rec'
            _last_sampled_count = 0
            _last_used_fallback = False
            _last_fallback_reason = ''

            def __init__(self, item_ids, candidate_count):
                self.item_ids = item_ids
                self._last_candidate_count_after_filters = candidate_count

            def get_user_item_ids(self, _user_id):
                return frozenset({101, 102})

            def get_recommendations(self, _user_id):
                return ItemList(
                    None,
                    item_ids=self.item_ids,
                    scores=[1.0] * len(self.item_ids),
                    rank=list(range(1, len(self.item_ids) + 1)),
                )

            def update_feedback(self, *_args):
                return None

        class FakeLogger:
            def __init__(self):
                self.rows = []

            def log_item_appear(self, *_args):
                return None

            def log_item_click(self, *_args):
                return None

            def log_consumer(self, row):
                self.rows.append(row)

            def is_sampled_user(self, _user_id):
                return False

        class FakeItemSelection:
            def select_item(self, *_args):
                return (-1, -1.0)

        cases = [
            ([1, 2, 3], 3, 3, None, ''),
            ([1, 2], 3, 2, 2, 'profile_too_small'),
            ([], 3, 0, 0, 'dataset_not_viable'),
        ]
        for item_ids, requested, delivered, candidate_count, fallback_reason in cases:
            with self.subTest(delivered=delivered):
                logger = FakeLogger()
                recommender = FakeRecommender(item_ids, candidate_count=delivered)
                recommender._last_fallback_reason = fallback_reason
                consumer = SimpleNamespace(
                    id=7,
                    type='test_type',
                    recommender=recommender,
                    seen_items=set(),
                    clicked_items=set(),
                    item_selection_model=FakeItemSelection(),
                    recommender_choice_model=SimpleNamespace(
                        update_recommender_utility=lambda _utility: 0.5
                    ),
                    utility_model=SimpleNamespace(
                        compute_list_utility=lambda _consumer, _recs: 1.0
                    ),
                    preference_vector=None,
                )
                metrics = {
                    'recommendations': 0,
                    'served_users': set(),
                    'fallback_used': 0,
                    'sampled_items': 0,
                    'slate_items_total': 0,
                    'unique_items': set(),
                    'clicks': 0,
                }
                smores.Smores.state = SimpleNamespace(
                    current_time=lambda: 'time-0',
                    cycle_count=0,
                    day_count=0,
                    slate_size=requested,
                    logger=logger,
                    summary_logger=SimpleNamespace(
                        log_recommendation=lambda *_args, **_kwargs: None,
                        log_click=lambda *_args: None,
                    ),
                    recommender_metrics={'test_rec': metrics},
                    providers=SimpleNamespace(
                        update_utility_list=lambda *_args: None,
                        update_utility_item=lambda *_args: None,
                    ),
                    cycle_clicked_blocklist=defaultdict(set),
                )

                Smores.run_consumer_day(None, consumer)

                row = logger.rows[0]
                self.assertEqual(row.profile_size, 2)
                self.assertEqual(row.requested_slate_size, requested)
                self.assertEqual(row.delivered_slate_size, delivered)
                self.assertEqual(row.candidate_count_after_filters, candidate_count)
                self.assertEqual(row.fallback_reason, fallback_reason)

    def test_consumer_log_csv_and_parquet_keep_named_columns(self):
        expected_fields = list(ConsumerUtility._fields)
        row = ConsumerUtility(
            7, 'type', 'rec', 1.0, 0.5, 'time-0', 'profile_too_small', 2, 5, 3, 3
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / 'consumer_utility.csv'
            logger = SmoresLogger.__new__(SmoresLogger)
            logger.consumer_output_file = csv_path.open('w', newline='')
            logger.consumer_writer = csv.DictWriter(
                logger.consumer_output_file, fieldnames=expected_fields
            )
            logger.consumer_writer.writeheader()
            logger.log_consumer(row)
            logger.consumer_output_file.close()

            with csv_path.open(newline='') as output:
                self.assertEqual(next(csv.reader(output)), expected_fields)

            smores.Smores.state = SimpleNamespace(logger=SimpleNamespace(debug=lambda _msg: None))
            logger.copy_parquet(csv_path)
            parquet_path = csv_path.with_suffix('.parquet')
            self.assertEqual(pq.read_table(parquet_path).schema.names, expected_fields)


if __name__ == '__main__':
    unittest.main()
