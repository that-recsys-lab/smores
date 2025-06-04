import unittest
import yaml
import csv
from pathlib import Path

from smores.utils import SmoresConfig
from smores.trigger import TriggerEvent, TriggerFactory, InitialBurnInTrigger, CycleEvent, CycleTrigger, TriggerCollection
from smores import Smores

from icecream import ic


# A dummy implementation so that we have more than one.
# Might turn into a real thing depending on how we implement portfolio portability
class ForgetUnconnected(CycleTrigger):
    def __init__(self):
        super().__init__()

    def setup(self, config):
        config.params['repeating'] = "True"
        config.params['cycle_count'] = 1
        super().setup(config)

    def handle_event(self, event: TriggerEvent):
        # Tell recommender to delete users?
        pass
    
TriggerFactory.register('cold_start_memory', ForgetUnconnected)

class TestTrigger(unittest.TestCase):
    def setUp(self):
        test_data_path = Path('tests/test_data')
        test_config_path = test_data_path / 'trigger_config.yaml'
        self.config = SmoresConfig.model_validate(yaml.safe_load(test_config_path.read_text()))
        self.smores = Smores(self.config)


    def test_trigger_factory(self):
        trigger = TriggerFactory.make_object('initial_burnin')
        self.assertIsInstance(trigger, InitialBurnInTrigger)

    def test_initial_burnin_trigger(self):
        trigger = InitialBurnInTrigger()
        trigger_config = self.config.triggers[0]
        trigger.setup(trigger_config)

        self.assertEqual(trigger.cycle_count, 5)

        tev1 = CycleEvent(0)
        self.assertFalse(trigger.accept_event(tev1))

        tev2 = CycleEvent(10)
        self.assertFalse(trigger.accept_event(tev2))

    def test_trigger_collection(self):
        trigger_coll = TriggerCollection()
        trigger_coll.setup(self.config.triggers)
        ic(trigger_coll.collection)
        self.assertEqual(len(trigger_coll.collection['cycle']), 2)

                        

if __name__ == '__main__':
    unittest.main()