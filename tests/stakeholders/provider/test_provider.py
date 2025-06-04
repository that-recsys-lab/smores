import unittest
import yaml
from pathlib import Path
from icecream import ic

from smores.utils import SmoresConfig
from smores.stakeholders.provider import Provider


class ProviderTestCase(unittest.TestCase):
    def test_utility_model_creation(self):
        test_data_path = Path('tests/test_data')
        test_config_path = test_data_path / 'test_config.yaml'
        self.config = SmoresConfig.model_validate(yaml.safe_load(test_config_path.read_text()))

        provider = Provider()
        provider.setup(self.config)
        self.assertIsNotNone(provider.utility_model)


if __name__ == '__main__':
    unittest.main()