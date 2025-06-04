from abc import ABC, abstractmethod
import numpy as np


class ItemSelectionModel(ABC):
    """
    Abstract base class for item selection models.
    
    Item selection models are responsible for selecting an item from a slate
    based on consumer preferences and other factors.
    """
    
    @abstractmethod
    def setup(cls, config):
        """
        Set up the model with configuration parameters.
        
        Args:
            config: Configuration object with parameters for the model.
        """
        pass
    
    @abstractmethod
    def select_item(cls, items, threshold, category_preferences, prohibited_genres):
        """
        Select an item from a slate based on consumer preferences.
        
        Args:
            items (list): List of items to evaluate.
            threshold (float): Threshold value to evaluate scores.
            category_preferences (dict): User's preferences for genres.
            prohibited_genres (set): Genres to penalize in the scoring.
            
        Returns:
            int: Index of the selected item in the input list, or None if no selection is possible.
        """
        pass




class CategorySimilarityLogitModel(ItemSelectionModel):
    """
    Item choice model that uses a multinomial logit approach based on category similarity.
    """
    def __init__(self):
        self.threshold = 0.3  # Default threshold

    def setup(self, config):
        """
        Set up the model with configuration parameters.

        Args:
            config: Configuration object with parameters for the model.
        """
        if hasattr(config, 'threshold'):
            self.threshold = config.threshold

    def select_item(self, items, threshold=None, category_preferences=None, prohibited_genres=None):
        """
        Select an item from a slate based on category similarity.

        Args:
            items (list): List of items to evaluate.
            threshold (float, optional): Threshold value to evaluate scores.
                Defaults to the class's threshold value.
            category_preferences (dict): User's preferences for genres.
            prohibited_genres (set): Genres to penalize in the scoring.

        Returns:
            int: Index of the selected item in the input list, or None if no selection is possible.
        """
        if threshold is None:
            threshold = self.threshold

        # Placeholder for item scores
        scores = np.zeros(len(items))

        # Calculate utility for each item based on category similarity
        for i, item in enumerate(items):
            category_similarity = 0.0
            for category in item.genres:
                if category in prohibited_genres:
                    category_similarity -= 1
                else:
                    category_similarity += category_preferences.get(category, 0)
            scores[i] = category_similarity

        # Calculate probabilities using a multinomial logit choice model
        if np.all(scores <= threshold):
            probabilities = np.zeros(len(scores))
        else:
            probabilities = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))

        # Handle case where probabilities sum to zero
        if np.sum(probabilities) == 0:
            return None

        # Select an item index based on probabilities
        selected_index = np.random.choice(len(items), p=probabilities)
        return selected_index


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