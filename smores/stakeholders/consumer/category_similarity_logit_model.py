import numpy as np
from smores.stakeholders.consumer.item_selection_model import ItemSelectionModel, ItemSelectionModelFactory

class CategorySimilarityLogitModel(ItemSelectionModel):
    """
    Item choice model that uses a multinomial logit approach based on category similarity.
    """
    
    threshold = 0.3  # Default threshold
    
    @classmethod
    def setup(cls, config):
        """
        Set up the model with configuration parameters.
        
        Args:
            config: Configuration object with parameters for the model.
        """
        if hasattr(config, 'threshold'):
            cls.threshold = config.threshold
    
    @classmethod
    def select_item(cls, items, threshold=None, category_preferences=None, prohibited_genres=None):
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
            threshold = cls.threshold
        
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


# Register the model with the factory
ItemSelectionModelFactory.register('category_similarity_logit', CategorySimilarityLogitModel)