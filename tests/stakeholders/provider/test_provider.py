import unittest
import yaml
from pathlib import Path
from icecream import ic

from smores.utils import SmoresConfig
from smores.stakeholders.provider import Provider, ProviderCollection, ProviderInfo
from smores import Smores

class ProviderTestCase(unittest.TestCase):
    def setUp(self):
        test_data_path = Path('tests/test_data')
        test_config_path = test_data_path / 'test_config.yaml'
        self.config = SmoresConfig.model_validate(yaml.safe_load(test_config_path.read_text()))
        self.smores = Smores(self.config)
        self.smores.setup()

        self.provider_data_path = test_data_path / 'providers.csv'

    def test_component_creation(self):
        pcoll = ProviderCollection()
        pcoll.setup(self.config.provider.types)
        provider_config: ProviderInfo = ProviderInfo.model_validate({"provider_id": 100,
                                                                     "provider_type": "Generic"})
        provider_type_config = pcoll.types["Generic"]

        provider = Provider()
        provider.setup(provider_type_config, provider_config)

        self.assertIsNotNone(provider.utility_model)

    def test_provider_loading(self):
        pcoll = ProviderCollection()
        pcoll.setup(self.config.provider.types)
        pcoll.load_providers(self.provider_data_path)

        self.assertEqual(len(list(pcoll)), 4)



if __name__ == '__main__':
    unittest.main()