from abc import ABC, abstractmethod 
from numpy.linalg import norm
from numpy import dot, average
from distutils.util import strtobool
from collections import defaultdict

import smores
from smores.recommender import Recommender, UnknownRecommenderError
# would like to import but circular issue needs to be resolved
#from smores.stakeholders.consumer import Consumer

class TriggerEvent:
    def __init__(self, event_type):
        self.event_type = event_type

class CycleEvent(TriggerEvent):
    def __init__(self, cycle_count, time):
        super().__init__('cycle')
        self.cycle_count = cycle_count
        self.time = time

class DayEvent(TriggerEvent):
    def __init__(self, day_count, time):
        super().__init__('day')
        self.day_count = day_count
        self.time = time

class InteractionBatchEvent(TriggerEvent):
    def __init__(self, interaction_dict):
        super().__init__('interaction')
        self.interaction_dict = interaction_dict


class Trigger (ABC):
    '''
    Trigger

    Take some action depending on the state of the simulation.
    '''
    def __init__(self, trigger_type: str):
        self.trigger_type = trigger_type

    @abstractmethod
    def setup(self, config):
        pass

    def apply_trigger(self, event: TriggerEvent):
        if self.accept_event(event):
            self.handle_event(event)

    @abstractmethod
    def accept_event(self, event: TriggerEvent) -> bool:
        return False

    @abstractmethod
    def handle_event(self, event: TriggerEvent):
        # Could potentially log here
        raise NotImplementedError


class CycleTrigger(Trigger):
    def __init__(self):
        super().__init__('cycle')

    def setup(self, config):
        self.name = config.name
        self.repeating = strtobool(config.params['repeating'])
        self.cycle_count = int(config.params['cycle_count'])

    def accept_event(self, event: CycleEvent):
        if not self.repeating and self.cycle_count == event.cycle_count:
            return True
        if self.repeating and self.cycle_count % event.cycle_count == 0:
            return True
        return False

class InitialBurnInTrigger(CycleTrigger):
    def setup(self, config):
        config.params['repeating'] = "False"
        super().setup(config)
        self.recommenders_to_activate = config.params['recommenders']

    def handle_event(self, event):
        # activate listed recommenders
        for name in self.recommenders_to_activate:
            rec = Recommender.name2recommender(name)
            if rec is not None:
                smores.Smores.state.recommenders_active.append(name)
            else:
                raise UnknownRecommenderError(name)


class InteractionBatchTrigger(Trigger):
    def __init__(self):
        super().__init__('interaction')

    def setup(self, config):
        self.name = config.name

class UniversalProfileTrigger(InteractionBatchTrigger):
    def accept_event(self, event: InteractionBatchEvent):
        return True

    # Should update all recommenders with the interaction, except the
    # recommender that the user is connected to. (That one gets updated
    # by default). Assemble in batches because that's more efficient for
    # dataset building.
    def handle_event(self, event: InteractionBatchEvent):
        other_interaction_dict = defaultdict(list)

        for name, recommender in smores.Smores.state.recommenders_available.items():
            for other_name, other_rec in smores.Smores.state.recommenders_available.items():
                rec_interactions = event.interaction_dict[name]
                if other_name != name:
                    other_interaction_dict[other_name] = other_interaction_dict[other_name] + rec_interactions

        for name, interactions in other_interaction_dict.items():
            rec = Recommender.name2recommender(name)
            if rec is not None:
                rec.update_interactions(interactions)
            else:
                raise UnknownRecommenderError(name)


            



class TriggerFactory():
    """
    The ProviderFactory associates provider types with class names and allows appropriate instances
    to be created. A provider class must registered in the factory before it can be
    created.
    """

    _class_name_map = {}

    @classmethod
    def register(cls, type_name, provider_class):
        if not issubclass(provider_class, Trigger):
            raise InvalidTriggerError(type_name)
        cls._class_name_map[type_name] = provider_class

    @classmethod
    def register_all(cls, type_specs):
        for type_name, provider_class in type_specs:
            cls.register(type_name, provider_class)

    @classmethod
    def make_object(cls, type_name):
        provider_class = cls._class_name_map.get(type_name)
        if provider_class is None:
            raise UnregisteredTriggerError(type_name)
        return provider_class()



# Exceptions
class InvalidTriggerError(Exception):
    def __init__(self, name):
        self.message = self.message = f'Cannot create Trigger object: Class {name} is not a subclass of Trigger.'
        super().__init__(self.message)


class UnregisteredTriggerError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create Trigger object: Class {name} is not registered and may not exist.'
        super().__init__(self.message)


TriggerFactory.register('initial_burnin', InitialBurnInTrigger)
TriggerFactory.register('universal_profile', UniversalProfileTrigger)
