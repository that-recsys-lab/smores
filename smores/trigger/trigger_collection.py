from .trigger import Trigger, TriggerFactory
from collections import defaultdict

class TriggerCollection():
    def __init__(self):
        self.collection = defaultdict(list)

    def setup(self, config: list):
        for trigger_def in config:
            trigger = TriggerFactory.make_object(trigger_def.class_name)
            trigger.setup(trigger_def)
            self.add_trigger(trigger)

    def add_trigger(self, trigger: Trigger):
        self.collection[trigger.trigger_type].append(trigger)

    def get_iterator(self, trigger_type: str):
        return self.collection[trigger_type].__iter__()
    
    