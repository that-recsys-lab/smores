import logging
import os
import csv
import pyarrow.csv as pv
import pyarrow.parquet as pq
from datetime import datetime
from pathlib import Path
from collections import namedtuple

from smores.utils import LoggerConfig
import smores

ConsumerUtility = namedtuple('ConsumerUtility', ['consumer_id', 'consumer_type', 'recommender', 'utility', 'aggregate', 'time'])
ProviderUtility = namedtuple('ProviderUtility', ['provider_id', 'provider_type', 'recommender', 'utility', 'time'])
ChoiceUtility = namedtuple('ChoiceUtility', ['consumer_id', 'consumer_type', 'current_recommender', 'next_recommender', 'utilities', 'time'])


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

    def setup(self, config):
        debug_file_handler = logging.FileHandler(self.debug_log_path)
        debug_console_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        debug_file_handler.setFormatter(formatter)
        debug_console_handler.setFormatter(formatter)

        self.debug_logger.addHandler(debug_file_handler)
        self.debug_logger.addHandler(debug_console_handler)

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
        self.consumer_output_file.close()
        self.provider_output_file.close()
        self.choice_output_file.close()

        if self.use_parquet:
            self.copy_parquet(self.consumer_log_path)
            self.copy_parquet(self.provider_log_path)
            self.copy_parquet(self.choice_log_path)
