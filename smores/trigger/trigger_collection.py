from .trigger import Trigger, TriggerFactory
from collections import defaultdict

class TriggerCollection():
    def __init__(self):
        self.collection = defaultdict(list)

    def setup(self, config: list):
        if config is not None:
            for trigger_def in config:
                trigger = TriggerFactory.make_object(trigger_def.class_name)
                trigger.setup(trigger_def)
                self.add_trigger(trigger)

    def add_trigger(self, trigger: Trigger):
        self.collection[trigger.trigger_type].append(trigger)

    def get_triggers(self, trigger_type: str):
        return self.collection[trigger_type]

    def get_iterator(self, trigger_type: str):
        return self.collection[trigger_type].__iter__()
    
    def delete_trigger(self, name, trigger_type):
        self.collection[trigger_type] = [trigger for trigger in self.collection[trigger_type] if trigger.name != name]

    def clear_trigger_type(self, trigger_type):
        self.collection[trigger_type] = []
        
    
    