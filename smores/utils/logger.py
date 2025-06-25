import logging
import os
import csv
from datetime import datetime
from pathlib import Path
from collections import namedtuple

from smores.utils import LoggerConfig

ConsumerUtility = namedtuple('ConsumerUtility', ['consumer_id', 'recommender', 'utility', 'aggregate'])
ProviderUtility = namedtuple('ProviderUtility', ['provider_id', 'recommender', 'utility'])


class SmoresLogger:
    CONSUMER_UTILITY_HEADERS = ['user_id', 'recommender', 'utility', 'aggregate']
    PROVIDER_UTILITY_HEADERS = ['provider_id', 'recommender', 'utility']

    def __init__(self, config: LoggerConfig):
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
        consumer_file_name = f'{config.consumer_file}_{timestamp}.csv'
        self.consumer_log_path = output_dir / consumer_file_name

        # provider utility output
        provider_file_name = f'{config.provider_file}_{timestamp}.csv'
        self.provider_log_path = output_dir / provider_file_name

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

    def cleanup(self):
        """Close the data file when done."""
        self.consumer_output_file.close()
        self.provider_output_file.close()
