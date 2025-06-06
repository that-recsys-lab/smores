
class UtilityHistoryEntry():
    def __init__(self, time, recommender, utility):
        self.time: int = time
        self.recommender = recommender
        self.value: float = utility


class UtilityHistory():
    def __init__(self):
        self.collection: list[UtilityHistoryEntry] = []

    def get_history(self):
        return self.collection

    def get_history_for_recommender(self, recommender):
        return [entry for entry in self.collection if entry.recommender ==  recommender]

    def add_entry(self, time, recommender, utility):
        entry = UtilityHistoryEntry(time, recommender, utility)
        self.collection.append(entry)

