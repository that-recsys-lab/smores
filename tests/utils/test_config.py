import unittest
import yaml
from pathlib import Path

from smores.utils import SmoresConfig
from tests.paths import TEST_CONFIG_PATH

class ConfigTestCase(unittest.TestCase):
    def test_config_load(self):
        config = SmoresConfig.model_validate(yaml.safe_load(TEST_CONFIG_PATH.read_text()))
        self.assertIsNotNone(config)


if __name__ == '__main__':
    unittest.main()
