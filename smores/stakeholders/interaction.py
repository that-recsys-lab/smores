# Represents a interaction between a user and an item through a recommender.
class Interaction:
    def __init__(self, user_id, item_id, recommender_id, rating=None):
        self.user_id = user_id # Consumer/user
        self.item_id = item_id # Item the user interacted with
        self.recommender_id = recommender_id # Recommender system that recorded this interaction
        self.rating = rating 

    def __repr__(self):
        return f"Interaction(user_id={self.user_id}, item_id={self.item_id}, recommender_id={self.recommender_id}, rating={self.rating})"