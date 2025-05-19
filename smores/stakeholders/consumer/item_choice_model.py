from abc import ABC, abstractmethod

class ItemChoiceModel(ABC):
    """
    Abstract base class for item choice models.
    
    Item choice models are responsible for selecting an item from a slate
    based on consumer preferences and other factors.
    """
    
    @classmethod
    @abstractmethod
    def setup(cls, config):
        """
        Set up the model with configuration parameters.
        
        Args:
            config: Configuration object with parameters for the model.
        """
        pass
    
    @classmethod
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


class ItemChoiceModelFactory:
    """
    The ItemChoiceModelFactory associates names with class objects to create item choice models
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
        if not issubclass(model_class, ItemChoiceModel):
            raise InvalidItemChoiceModelError(model_name)
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
    def get_class(cls, model_name):
        """
        Get a model class by name.
        
        Args:
            model_name (str): Name of the model to get.
            
        Returns:
            class: Class object for the model.
        """
        model_class = cls._class_name_map.get(model_name)
        if model_class is None:
            raise UnregisteredItemChoiceModelError(model_name)
        return model_class


# Exceptions
class InvalidItemChoiceModelError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create item choice model: Class {name} is not a subclass of ItemChoiceModel.'
        super().__init__(self.message)


class UnregisteredItemChoiceModelError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create item choice model: Class {name} is not registered and may not exist.'
        super().__init__(self.message)