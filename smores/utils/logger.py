import logging
import os
import csv
import pyarrow.csv as pv
import pyarrow.parquet as pq
from datetime import datetime
from pathlib import Path
from collections import namedtuple, Counter

from smores.utils import LoggerConfig
import smores

ConsumerUtility = namedtuple('ConsumerUtility', ['consumer_id', 'consumer_type', 'recommender', 'utility', 'aggregate', 'time'])
ProviderUtility = namedtuple('ProviderUtility', ['provider_id', 'provider_type', 'recommender', 'utility', 'time'])
ChoiceUtility = namedtuple('ChoiceUtility', ['consumer_id', 'consumer_type', 'current_recommender', 'next_recommender', 'utilities', 'time'])
UserJourney = namedtuple('UserJourney', ['user_id', 'cycle', 'day', 'time', 'recommender', 'slate_size', 'slate_items', 'slate_scores', 'slate_utilities', 'selected_item', 'selected_rank', 'selected_utility', 'max_utility', 'num_unique_items_clicked', 'num_unique_items_seen'])


class SmoresLogger:
    CONSUMER_UTILITY_HEADERS = ['user_id', 'consumer_type', 'recommender', 'utility', 'aggregate', 'time']
    PROVIDER_UTILITY_HEADERS = ['provider_id', 'provider_type', 'recommender', 'utility', 'time']
    RECOMMENDER_CHOICE_HEADERS = ['consumer_id', 'consumer_type', 'time', 'current_recommender', 'next_recommender']

    def __init__(self, config: LoggerConfig):
        self.use_timestamp = config.use_timestamp
        self.use_parquet = config.use_parquet

        output_dir = Path(config.directory)
        os.makedirs(output_dir, exist_ok=True)

        # debug log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file_name = f'{config.debug_file}_{timestamp}.log'
        self.debug_log_path = output_dir / debug_file_name

        self.debug_logger = logging.getLogger('debug')
        if config.debug_level == 'debug':
            self.debug_logger.setLevel(logging.DEBUG)
        else:
            self.debug_logger.setLevel(logging.INFO)

        # cycle metrics output
        cycle_file = config.cycle_file or "cycle_metrics"
        if self.use_timestamp:
            cycle_file_name = f"{cycle_file}_{timestamp}.csv"
        else:
            cycle_file_name = f"{cycle_file}.csv"
        self.cycle_log_path = output_dir / cycle_file_name

        # consumer utility output
        if self.use_timestamp:
            consumer_file_name = f'{config.consumer_file}_{timestamp}.csv'
        else:
            consumer_file_name = f'{config.consumer_file}.csv'
        self.consumer_log_path = output_dir / consumer_file_name

        # provider utility output
        if self.use_timestamp:
            provider_file_name = f'{config.provider_file}_{timestamp}.csv'
        else:
            provider_file_name = f'{config.provider_file}.csv'
        self.provider_log_path = output_dir / provider_file_name

        # recommender choice output
        if self.use_timestamp:
            choice_file_name = f'{config.choice_file}_{timestamp}.csv'
        else:
            choice_file_name = f'{config.choice_file}.csv'
        self.choice_log_path = output_dir / choice_file_name

        # user journey configuration
        self.enable_journey_logging = config.enable_user_journeys
        self.sampled_user_ids = set(config.sampled_user_ids or [])
        self.sampled_user_count = config.sampled_user_count
        self.journey_log_path = None
        self.journey_output_file = None
        self.journey_writer = None
        if self.enable_journey_logging:
            if config.sampled_user_file is None:
                raise ValueError("enable_user_journeys is True but sampled_user_file is not provided")
            if self.use_timestamp:
                journey_filename = f"{config.sampled_user_file}_{timestamp}.csv"
            else:
                journey_filename = f"{config.sampled_user_file}.csv"
            self.journey_log_path = output_dir / journey_filename

        # item stats configuration
        self.enable_item_stats = config.enable_item_stats
        self.item_stats_path = None
        self.item_stats_output_file = None
        self.item_stats_writer = None
        self.item_appear_counts: Counter[tuple[int, str, int]] = Counter()
        self.item_click_counts: Counter[tuple[int, str, int]] = Counter()
        if self.enable_item_stats:
            if config.item_stats_file is None:
                raise ValueError("enable_item_stats is True but item_stats_file is not provided")
            if self.use_timestamp:
                stats_filename = f"{config.item_stats_file}_{timestamp}.csv"
            else:
                stats_filename = f"{config.item_stats_file}.csv"
            self.item_stats_path = output_dir / stats_filename

        self.cycle_output_file = None
        self.cycle_writer = None

    def setup(self, config):
        debug_file_handler = logging.FileHandler(self.debug_log_path)
        debug_console_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        debug_file_handler.setFormatter(formatter)
        debug_console_handler.setFormatter(formatter)

        self.debug_logger.addHandler(debug_file_handler)
        self.debug_logger.addHandler(debug_console_handler)

        self.cycle_output_file = open(self.cycle_log_path, 'w', newline='')
        cycle_fieldnames = [
            'cycle',
            'recommender',
            'interactions',
            'dataset_users',
            'dataset_items',
            'avg_profile_len',
            'active_users',
            'new_users',
            'churned_users',
            'new_interactions',
            'deleted_interactions',
            'avg_slate_size',
            'unique_items',
            'coverage_pct',
            'rec_requests',
            'fallback_used',
            'avg_sampled',
            'ctr',
        ]
        if smores.Smores.state.trigger_tester is not None:
            cycle_fieldnames.append('trigger_success')
        self.cycle_writer = csv.DictWriter(self.cycle_output_file, fieldnames=cycle_fieldnames)
        self.cycle_writer.writeheader()

        self.consumer_output_file = open(self.consumer_log_path, 'w', newline='')
        self.consumer_writer = csv.DictWriter(self.consumer_output_file, fieldnames=ConsumerUtility._fields)
        self.consumer_writer.writeheader()

        self.provider_output_file = open(self.provider_log_path, 'w', newline='')
        self.provider_writer = csv.DictWriter(self.provider_output_file, fieldnames=ProviderUtility._fields)
        self.provider_writer.writeheader()

        self.choice_output_file = open(self.choice_log_path, 'w', newline='')
        self.choice_writer = csv.writer(self.choice_output_file)
        rec_names = smores.Smores.state.recommenders_base.get_names()
        headers = self.RECOMMENDER_CHOICE_HEADERS + rec_names
        self.choice_writer.writerow(headers)

        if self.enable_journey_logging:
            self.journey_output_file = open(self.journey_log_path, 'w', newline='')
            self.journey_writer = csv.DictWriter(self.journey_output_file, fieldnames=UserJourney._fields)
            self.journey_writer.writeheader()

            if self.sampled_user_count and not self.sampled_user_ids:
                all_consumer_ids = smores.Smores.state.consumers.get_consumer_ids()
                sample_size = min(self.sampled_user_count, len(all_consumer_ids))
                sampled = smores.Smores.state.rand.choice(all_consumer_ids, size=sample_size, replace=False)
                self.sampled_user_ids.update(int(s) for s in sampled)

            self.info(
                f"Journey logging enabled for {len(self.sampled_user_ids)} users: {sorted(self.sampled_user_ids)}"
            )

        if self.enable_item_stats:
            self.item_stats_output_file = open(self.item_stats_path, 'w', newline='')
            fieldnames = ['item_id', 'recommender', 'cycle', 'appearances', 'clicks']
            self.item_stats_writer = csv.DictWriter(self.item_stats_output_file, fieldnames=fieldnames)
            self.item_stats_writer.writeheader()

    # Debug log
    def debug(self, message):
        self.debug_logger.debug(message)

    def info(self, message):
        self.debug_logger.info(message)

    def warning(self, message):
        self.debug_logger.warning(message)

    def error(self, message):
        self.debug_logger.error(message)

    def critical(self, message):
        self.debug_logger.critical(message)


    def log_consumer(self, utility_info: ConsumerUtility):
        """Write a row of simulation data to the CSV file."""
        self.consumer_writer.writerow(utility_info._asdict())
        self.consumer_output_file.flush()

    def log_provider(self, utility_info: ProviderUtility):
        """Write a row of simulation data to the CSV file."""
        self.provider_writer.writerow(utility_info._asdict())
        self.provider_output_file.flush()

    def log_recommender_choice(self, choice_info: ChoiceUtility):
        """Write a row of recommender choice data to the CSV file."""
        row = [choice_info.consumer_id, choice_info.consumer_type, choice_info.time,
               choice_info.current_recommender, choice_info.next_recommender] + \
                choice_info.utilities
        self.choice_writer.writerow(row)
        self.choice_output_file.flush()

    def log_cycle_metrics(self, metrics: dict):
        """Persist per-cycle recommender metrics to CSV."""
        if self.cycle_writer is None:
            return
        self.cycle_writer.writerow(metrics)
        self.cycle_output_file.flush()

    def log_user_journey(self, journey_info: UserJourney):
        if not self.enable_journey_logging or self.journey_writer is None:
            return
        self.journey_writer.writerow(journey_info._asdict())
        self.journey_output_file.flush()

    def is_sampled_user(self, user_id: int) -> bool:
        return self.enable_journey_logging and user_id in self.sampled_user_ids

    def log_item_appear(self, item_id: int, recommender: str, cycle: int):
        if not self.enable_item_stats:
            return
        key = (item_id, recommender, cycle)
        self.item_appear_counts[key] += 1

    def log_item_click(self, item_id: int, recommender: str, cycle: int):
        if not self.enable_item_stats:
            return
        key = (item_id, recommender, cycle)
        self.item_click_counts[key] += 1

    def copy_parquet(self, inpath: Path):
        outpath = inpath.with_suffix('.parquet')
        smores.Smores.state.logger.debug(f'Converting {inpath} to parquet.')
        table = pv.read_csv(inpath)
        pq.write_table(table, outpath)
        os.remove(inpath)
        smores.Smores.state.logger.debug(f'\tConversion complete. {inpath} deleted. {outpath} written')

    # TODO: convert to parquet
    def cleanup(self):
        """Close the data file when done."""
        if self.cycle_output_file is not None:
            self.cycle_output_file.close()
        self.consumer_output_file.close()
        self.provider_output_file.close()
        self.choice_output_file.close()

        if self.enable_journey_logging and self.journey_output_file is not None:
            self.journey_output_file.close()

        if self.enable_item_stats and self.item_stats_output_file is not None:
            self._write_item_stats()
            self.item_stats_output_file.close()

        if self.use_parquet:
            parquet_targets = [
                self.consumer_log_path,
                self.provider_log_path,
                self.choice_log_path,
                self.cycle_log_path,
            ]
            if self.enable_journey_logging and self.journey_log_path is not None:
                parquet_targets.append(self.journey_log_path)
            if self.enable_item_stats and self.item_stats_path is not None:
                parquet_targets.append(self.item_stats_path)

            for path in parquet_targets:
                if path is not None and path.exists():
                    self.copy_parquet(path)

    def _write_item_stats(self):
        keys = set(self.item_appear_counts.keys()) | set(self.item_click_counts.keys())
        for item_id, recommender, cycle in sorted(keys):
            row = {
                'item_id': item_id,
                'recommender': recommender,
                'cycle': cycle,
                'appearances': self.item_appear_counts.get((item_id, recommender, cycle), 0),
                'clicks': self.item_click_counts.get((item_id, recommender, cycle), 0),
            }
            self.item_stats_writer.writerow(row)
        self.item_stats_output_file.flush()
