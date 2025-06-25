from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
# Cursed circular imports
from typing import TYPE_CHECKING


from lenskit.data.items import ItemList

from smores.utils import itemList2rankedTuples
if TYPE_CHECKING:
    from .consumer import Consumer

import smores

class ItemSelectionModel(ABC):
    """
    Abstract base class for item selection models.
    
    Item selection models are responsible for selecting an item from a slate
    based on consumer preferences and other factors.
    """
    EMPTY_OUTPUT = (-1, -1.0)
    
    @abstractmethod
    def setup(self, config):
        """
        Set up the model with configuration parameters.
        
        Args:
            config: Configuration object with parameters for the model.
        """
        pass
    
    @abstractmethod
    def select_item(self, consumer: "Consumer", item_list: ItemList):
        """
        Select an item from a slate based on consumer preferences.
        
        Args:
            consumer: The consumer object
            item_list: List of items to evaluate.
            
        Returns:
            int: item_id, utility
        """
        return ItemSelectionModel.EMPTY_OUTPUT
    
    @classmethod
    def is_empty_selection(cls, tuple):
        return tuple[0] == ItemSelectionModel.EMPTY_OUTPUT[0]


class CategorySimilarityLogitModel(ItemSelectionModel):
    """
    Item choice model that uses a multinomial logit approach based on category similarity.
    """
    def __init__(self):
        self.threshold = 0.3  # Default threshold
        self.selection_utility = None

    def setup(self, config):
        """
        Set up the model with configuration parameters.

        Args:
            config: Configuration object with parameters for the model.
        """
        self.threshold = config.params['threshold']
        self.selection_utility = config.params['selection_utility']

    def select_item(self, consumer: "Consumer", item_list: ItemList):
        """
        Select an item from a slate based on category similarity.

        Args:
            consumer: The consumer object
            item_list: List of items to evaluate.
            
        Returns:
            int: item_id, score
        """

        category_similarities = []

        # Calculate utility for each item based on category similarity
        # item_id, score
        item_tuples = itemList2rankedTuples(item_list)
        # remove items that the user has selected previously
        # TODO: Maybe this should be a configurable aspect?

        item_tuples_filtered = [item_tuple for item_tuple in item_tuples \
                                 if not consumer.history.contains_item(item_tuple[0])]
        
        if len(item_tuples_filtered) == 0:
            return ItemSelectionModel.EMPTY_OUTPUT
        
        for id, score in item_tuples_filtered:
            item = smores.Smores.state.items.get_item(id)
            if consumer.preference_vector is not None:
                similarity = np.dot(item.features, consumer.preference_vector)
                if similarity < self.threshold:
                    similarity = 0
                category_similarities.append(similarity)

        if np.sum(category_similarities) == 0:
            return ItemSelectionModel.EMPTY_OUTPUT

        probabilities = np.exp(category_similarities - np.max(category_similarities)) / \
            np.sum(np.exp(category_similarities - np.max(category_similarities)))

        # Select an item index based on probabilities
        selected_tuple = smores.Smores.state.rand.choice(item_tuples, p=probabilities)
        return (selected_tuple[0], self.selection_utility)
    

class ItemSelectionModelFactory:
    """
    The ItemSelectionModelLookup associates names with class objects to return item selection models
    based on configuration information. 
    """
    
    _class_name_map = {}
    
    @classmethod
    def register(cls, model_name, model_class):
        """
        Register a model class with the factory.
        
        Args:
            model_name (str): Name of the model to register.
            model_class (class): Class object to register.
        """
        if not issubclass(model_class, ItemSelectionModel):
            raise InvalidItemSelectionModelError(model_name)
        cls._class_name_map[model_name] = model_class
    
    @classmethod
    def register_all(cls, model_specs):
        """
        Register multiple model classes with the factory.
        
        Args:
            model_specs (list): List of (model_name, model_class) tuples.
        """
        for model_name, model_class in model_specs:
            cls.register(model_name, model_class)
    
    @classmethod
    def create(cls, model_name):
        """
        Get a model class by name.
        
        Args:
            model_name (str): Name of the model to get.
            
        Returns:
            object: Instance of the model.
        """
        model_class = cls._class_name_map.get(model_name)
        if model_class is None:
            raise UnregisteredItemSelectionModelError(model_name)
        return model_class()

# Register classes
# Register the model with the factory
ItemSelectionModelFactory.register('category_similarity_logit', CategorySimilarityLogitModel)



# Exceptions
class InvalidItemSelectionModelError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create item selection model: Class {name} is not a subclass of ItemSelectionModel.'
        super().__init__(self.message)


class UnregisteredItemSelectionModelError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create item selection model: Class {name} is not registered and may not exist.'
        super().__init__(self.message)