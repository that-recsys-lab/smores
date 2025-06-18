from lenskit.data import ItemList

from smores.recommender import Recommender

from .trigger import Trigger, TriggerEvent, TriggerFactory


class SwitchEvent(TriggerEvent):
    def __init__(self, consumer, next_rec_name):
        super().__init__('switch')
        self.consumer_id = consumer.id
        self.from_rec = consumer.recommender.name
        self.next_rec = next_rec_name


class SwitchTrigger(Trigger):
    def __init__(self):
        super().__init__('switch')

    def setup(self, config):
        self.name = config.name

    def accept_event(self, event: SwitchEvent):
        return True


class SwitchSaveInfoTrigger(SwitchTrigger):
    def __init__(self):
        super().__init__()
        self.events: list[TriggerEvent] = []

    def setup(self, config):
        super().setup(config)

    def handle_event(self, event: TriggerEvent):
        self.events.append(event)

# User Ownership
# The profile information is deleted from the FROM recommender
# The profile information is added to the TO recommender
class ProfileUserOwnershipTrigger(SwitchTrigger):
    def __init__(self):
        super().__init__()

    def setup(self, config):
        super().setup(config)

    def handle_event(self, event: SwitchEvent):
        from_rec: Recommender = Recommender.name2recommender(event.from_rec)
        user_data = from_rec.get_user(event.consumer_id)

        if user_data is not None:
            to_rec: Recommender = Recommender.name2recommender(event.next_rec)
            to_rec.update_dataset_itemlist(user_data)

            from_rec.delete_user(event.consumer_id)

# Cold Start
# The profile information is deleted from the FROM recommender
# Nothing happens to the TO recommender
class ProfileColdStartTrigger(SwitchTrigger):
    def __init__(self):
        super().__init__()

    def setup(self, config):
        super().setup(config)

    def handle_event(self, event: SwitchEvent):
        from_rec: Recommender = Recommender.name2recommender(event.from_rec)
        from_rec.delete_user(event.consumer_id)      

# For Universal and Algorithm-specific, nothing specific happens at switching item.   

TriggerFactory.register('save_switch', SwitchSaveInfoTrigger)
TriggerFactory.register('cold_start', ProfileColdStartTrigger)
TriggerFactory.register('user_ownership', ProfileUserOwnershipTrigger)