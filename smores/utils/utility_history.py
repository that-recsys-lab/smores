
class UtilityHistoryEntry():
    LIST_ENTRY = -1

    def __init__(self, item_id, time, recommender, utility):
        self.time: int = time
        self.item_id: int = item_id
        self.recommender = recommender
        self.value: float = utility

    def is_list_entry(self):
        return self.item_id == UtilityHistoryEntry.LIST_ENTRY


class UtilityHistory():
    def __init__(self):
        self.collection: list[UtilityHistoryEntry] = []

    def get_history(self):
        return self.collection

    def get_history_for_recommender(self, recommender):
        return [entry for entry in self.collection if entry.recommender == recommender]

    def add_entry(self, item_id, time, recommender, utility):
        entry = UtilityHistoryEntry(item_id, time, recommender, utility)
        self.collection.append(entry)

    def add_list_entry(self, time, recommender, utility):
        entry = UtilityHistoryEntry(UtilityHistoryEntry.LIST_ENTRY, time, recommender, utility)
        self.collection.append(entry)

    def contains_item(self, item_id):
        return any([entry.item_id == item_id for entry in self.collection])
    