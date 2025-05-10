from abc import ABC, abstractmethod

class ItemChoiceModel(ABC):
    @abstractmethod
    def select_item(self, items, threshold, category_preferences, prohibited_genres):
        pass