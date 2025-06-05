import unittest
import yaml
from pathlib import Path

from smores.utils import SmoresConfig

class ConfigTestCase(unittest.TestCase):
    def test_config_load(self):
        test_data_path = Path('tests/test_data')
        test_config_path = test_data_path / 'test_config.yaml'
        config = SmoresConfig.model_validate(yaml.safe_load(test_config_path.read_text()))
        self.assertIsNotNone(config)


if __name__ == '__main__':
    unittest.main()