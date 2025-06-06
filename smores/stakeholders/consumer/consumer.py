from icecream import ic
from pathlib import Path
from csv import DictReader
from json import loads
from pydantic import BaseModel, PositiveInt

from .consumer_utility_model import ConsumerUtilityModelFactory
from .item_selection_model import ItemSelectionModelFactory
from .recommender_choice_model import RecommenderChoiceModelFactory
from smores.utils import UtilityHistory, ConsumerTypeConfig
import smores

class ConsumerInfo (BaseModel):
    consumer_id: PositiveInt
    consumer_type: str
    preferences: list[float]

class Consumer:
    def __init__(self):
        self.id = -1
        self.preference_vector = None
        self.recommender = None
        self.utility_model = None
        self.item_selection_model = None
        self.recommender_choice_model = None
        self.history = None

    def __str__(self):
        return f'<Consumer {self.id}>'

    def setup(self, config_type: ConsumerTypeConfig, config_instance: ConsumerInfo):
        # Instance-specific
        self.id = config_instance.consumer_id
        self.preference_vector = config_instance.preferences

        # Consumer-type specific
        self.history = UtilityHistory()
        consumer_models = smores.Smores.state.consumer_models

        # Get utility model
        utility_model_name = config_type.utility_model
        self.utility_model = consumer_models.get_utility_model(utility_model_name)

        # Get selection model
        selection_model_name = config_type.item_selection_model
        self.item_selection_model = consumer_models.get_item_selection_model(selection_model_name)

        # Setup recommender choice model
        choice_model_config = config_type.recommender_choice_model
        self.recommender_choice_model = RecommenderChoiceModelFactory.create(choice_model_config.class_name)
        self.recommender_choice_model.setup(choice_model_config)


class ConsumerCollection():
    def __init__(self):
        self.collection: dict[int, Consumer] = {}
        self.types: dict[str, ConsumerTypeConfig] = {}

    def setup(self, config: list[ConsumerTypeConfig]):
        for type_config in config:
            self.types[type_config.name] = type_config

    def add_consumer(self, consumer: Consumer):
        self.collection[consumer.id] = consumer

    def get_consumer(self, consumer_id):
        return self.collection[consumer_id]

    def __iter__(self):
        return iter(self.collection.values())
    
    def load_consumers(self, consumer_data_path: Path):
        with open(consumer_data_path, 'r') as consumer_file:
            reader = DictReader(consumer_file)
            for row in reader:
                feature_list_str = row['preferences']
                feature_list = loads(feature_list_str)
                row['preferences'] = feature_list
                consumer_config: ConsumerInfo = ConsumerInfo.model_validate(row)
                consumer_type_config = self.types[consumer_config.consumer_type]

                consumer = Consumer()
                consumer.setup(consumer_type_config, consumer_config)

                self.add_consumer(consumer)


class ConsumerFactory():
    """
    The ConsumerFactory associates consumer types with class names and allows appropriate instances
    to be created. A consumer class must registered in the factory before it can be
    created.
    """

    _class_name_map = {}

    @classmethod
    def register(cls, type_name, consumer_class):
        if not issubclass(consumer_class, Consumer):
            raise InvalidConsumerError(type_name)
        cls._class_name_map[type_name] = consumer_class

    @classmethod
    def register_all(cls, type_specs):
        for type_name, consumer_class in type_specs:
            cls.register(type_name, consumer_class)

    @classmethod
    def make_object(cls, type_name):
        consumer_class = cls._class_name_map.get(type_name)
        if consumer_class is None:
            raise UnregisteredConsumerError(type_name)
        return consumer_class()



# Exceptions
class InvalidConsumerError(Exception):
    def __init__(self, name):
        self.message = self.message = f'Cannot create Consumer object: Class {name} is not a subclass of Provider.'
        super().__init__(self.message)


class UnregisteredConsumerError(Exception):
    def __init__(self, name):
        self.message = f'Cannot create Consumer object: Class {name} is not registered and may not exist.'
        super().__init__(self.message)



'''
class Consumer:
    def __init__(
            self,
            consumer_id,
            sensitivity,
            category_preferences,
            prohibited_genres=set(),
            favorite_genres=set(),
            choice_model=None,
            historical_distribution={}
    ):
        """
        Initialize a Consumer object with user-specific parameters.

        Args:
            consumer_id (int): Unique identifier for the consumer.
            category_preferences (dict): Dictionary of cetegories and generated probabilites of liking
        """
        self.consumer_id = consumer_id
        self.sensitivity = sensitivity
        self.net_quality_exposure = {}
        self.category_preferences = category_preferences
        self.connected_recommenders = {}
        self.satisfaction_scores = (
            {}
        )  # Dictionary to store satisfaction scores per recommender system
        self.recommender_counts = (
            {}
        )  # Dictionary to store how many times the recommender system has been chosen
        self.recommender_category_success = (
            {}
        )  # Dictionary to store how many times the recommender system got the favorite category
        self.ucb_scores = {}  # Store UCB scores for each recommender system
        self.prohibited_genres = (
            prohibited_genres  # Set to store genres penalized by the consumer
        )
        self.favorite_genres = (
            favorite_genres  # Set to store genres the consumer likes
        )
        self.alpha = 3  # exploration decay / exploitation weight
        self.beta = 2  # recency bias

        self.choice_model = choice_model if choice_model is not None else CategorySimilarityLogitModel()
        self.genre_recommendation_counts = {}
        self.historical_distribution = historical_distribution
        self.kl_divergence = defaultdict(float)

    def subscribe_to_recommender_system(self, recommender_system_id, state=1):
        """
        Subscribe to a new recommender system and initialize the satisfaction score and count.

        Args:
            recommender_system_id (str): Identifier for the new recommender system.
        """
        self.connected_recommenders[recommender_system_id] = state
        self.satisfaction_scores[recommender_system_id] = self.satisfaction_scores.get(
            recommender_system_id, 0
        )
        self.kl_divergence[recommender_system_id] = self.kl_divergence.get(
            recommender_system_id, 0
        )
        self.recommender_counts[recommender_system_id] = self.recommender_counts.get(
            recommender_system_id, 0
        )  # get the value that is in self.recommender_counts[recommender_system_id] otherwise set as zero
        self.recommender_category_success[recommender_system_id] = 0
        self.genre_recommendation_counts[recommender_system_id] = self.genre_recommendation_counts.get(
            recommender_system_id, {}
        )

    def unsubscribe_from_recommender_system(self, recommender_system_id):
        """
        Unsubscribe from a recommender system.

        Args:
            recommender_system_id (str): Identifier of the recommender system to unsubscribe from.
        """
        self.connected_recommenders[recommender_system_id] = 0

    def choose_recommender(self, preselected_recommender_id=None):
        """
        Choose a recommender system based on the Upper Confidence Bound (UCB) algorithm.

        Returns:
            str: ID of the selected recommender system.
        """
        # Filter connected recommender systems with value 1
        self.available_recommenders = [
            recommender_id
            for recommender_id, value in self.connected_recommenders.items()
            if value == 1
        ]

        if preselected_recommender_id:
            self.recommender_counts[preselected_recommender_id] += 1
            return preselected_recommender_id

        if not self.available_recommenders:
            print("no available recommenders")
            return None

        if len(self.available_recommenders) == 1:
            recommender_id = self.available_recommenders[0]
            self.recommender_counts[recommender_id] += 1
            return recommender_id

        # Check if all recommenders have zero counts
        if all(count == 0 for count in self.recommender_counts.values()):
            selected_recommender_id = random.choice(self.available_recommenders)
            self.recommender_counts[selected_recommender_id] += 1
            return selected_recommender_id

        # Calculate UCB scores for available recommender systems
        # Define round UCB scores
        round_ucb_scores = {
            recommender_id: score for recommender_id, score in self.ucb_scores.items()
        }
        for recommender_id in self.available_recommenders:
            if self.recommender_counts[recommender_id] == 0:
                round_ucb_scores[recommender_id] = float(
                    "inf"
                )  # Prioritize unselected recommenders
            else:
                exploitation_term = self.satisfaction_scores[recommender_id] + self.alpha
                # Adjusted UCB; added decay
                exploration_term = np.sqrt(
                    np.log(sum(self.recommender_counts.values()))
                    / self.recommender_counts[recommender_id]
                ) / (1 + sum(self.recommender_counts.values()))
                # update the ucb only for the selected recommender
                round_ucb_scores[recommender_id] = exploitation_term + exploration_term

        # Select the recommender with the highest UCB score
        available_ucb_scores = {
            recommender_id: score
            for recommender_id, score in round_ucb_scores.items()
            if recommender_id in self.available_recommenders
        }
        selected_recommender_id = max(
            available_ucb_scores, key=available_ucb_scores.get
        )

        # Update the count and ucb_score for the selected recommender
        self.ucb_scores[recommender_id] = round_ucb_scores[recommender_id]
        self.recommender_counts[selected_recommender_id] += 1

        return selected_recommender_id

    def simulate_response(self, slate_items, recommender_system_id):
        """
        Simulate response to a slate of items and return the responses.

        Args:
            slate_items (list): List of items to evaluate.
            recommender_system (Recommender): Recommender system instance.

        Returns:
            list: List of response dictionaries corresponding to the items.
        """
        # update genre recommendation counts
        self.update_genre_recommendation(recommender_system_id, slate_items)

        responses = []

        # Determine whether the user will click on anything
        click_prob = np.random.random()
        if click_prob <= 1:  # 100% chance of clicking
            # Choice model to select an item from the slate
            selected_index = self.choice_model.select_item(
                items=slate_items,
                threshold=0.3,
                category_preferences=self.category_preferences,
                prohibited_genres=self.prohibited_genres,
            )
            # print(selected_index)
            if selected_index is not None:
                for i, item in enumerate(slate_items):
                    if i == selected_index:
                        responses.append(
                            {"click": 1}
                        )  # Mark the selected item as clicked
                    else:
                        responses.append(
                            {"click": 0}
                        )  # Mark other items as not clicked
            else:
                # If choice model returned None, return all zeros
                responses = [{"click": 0} for _ in range(len(slate_items))]
                # print("Choice Model returned none")
        else:
            # User didn't click on anything
            responses = [{"click": 0} for _ in range(len(slate_items))]
            # print("User didn't click anything.")

        # update state
        self.update_state(slate_items, responses, recommender_system_id)

        return responses

    def update_genre_recommendation(self, recommender_system_id, items):
        """
        Update the genre recommendation counts for a list of items.

        Args:
            items (list): List of Item objects.
            recommender_system_id (str): Identifier of the recommender system.
        """
        if recommender_system_id not in self.genre_recommendation_counts:
            self.genre_recommendation_counts[recommender_system_id] = {}

        for item in items:
            for genre in item.genres:
                if genre not in self.genre_recommendation_counts[recommender_system_id]:
                    self.genre_recommendation_counts[recommender_system_id][genre] = 0
                self.genre_recommendation_counts[recommender_system_id][genre] += 1

    def update_state(self, slate_items, responses, recommender_system_id):
        """
        Update the net quality exposure (nqe) for the user after choosing a item.

        Args:
            slate_items (list): List of items presented to the user.
            responses (list): List of response dictionaries corresponding to the presented items.
            recommender_system (Recommender): Recommender system instance.
        """
        sim_score = 0
        for item in slate_items:
            # dot product of the intersection between the user interest and the item features for each item in the slate
            intersecting_keys = set(self.category_preferences.keys()).intersection(
                set(item.normalized_genres_vector.keys())
            )
            # Compute the dot product only for the intersecting keys
            sim_score += sum(
                self.category_preferences[key] * item.normalized_genres_vector[key]
                for key in intersecting_keys
            )
        # normalize sim_score
        norm_sim_score = sim_score / len(slate_items)

        self.satisfaction_scores[recommender_system_id] = (
                                                                  self.satisfaction_scores.get(recommender_system_id,
                                                                                               0) * self.beta
                                                                  + norm_sim_score
                                                          ) / (1 + self.beta)

    def get_satisfaction_score(self, recommender_system_id="default"):
        """
        Get the satisfaction score of the consumer for a specific recommender system.

        Args:
            recommender_system_id (str): Identifier of the recommender system (default is "default").

        Returns:
            float: Satisfaction score for the specified recommender system.
        """
        return self.satisfaction_scores.get(recommender_system_id, 0.0)

    def reset(self):
        self.net_quality_exposure = {}
        self.connected_recommenders = {}
        self.satisfaction_scores = (
            {}
        )  # Dictionary to store satisfaction scores per recommender system
        self.recommender_counts = (
            {}
        )  # Dictionary to store how many times the recommender system has been chosen
        self.ucb_scores = {}  # Store UCB scores for each recommender system
        self.kl_divergence = defaultdict(float)

    def compute_kl_divergence(self, recommender_system_id):
        """
        Compute the KL divergence between the consumer's category preferences and the historical distribution.

        """
        # Normalize historical_distribution
        profile_historical = sum(self.historical_distribution.values())
        historical_distribution_normalized = {
            genre: count / profile_historical
            for genre, count in self.historical_distribution.items()
        }

        # Normalize consumer's genre preferences
        total_count = sum(
            self.genre_recommendation_counts[recommender_system_id].values()
        )
        genre_preferences_normalized = {
            genre: count / total_count
            for genre, count in self.genre_recommendation_counts[recommender_system_id].items()
        }

        # Compute KL divergence
        kl_divergence = 0
        for genre, p in genre_preferences_normalized.items():
            q = historical_distribution_normalized.get(genre,
                                                       0.0)  # Default to 0 if genre is not in historical_distribution
            if p > 0:
                if q > 0:
                    kl_divergence += p * math.log(p / q)
                else:
                    # Avoid division by zero; handle cases where q is 0
                    kl_divergence += p * math.log(p / 1e-10)

        self.kl_divergence[recommender_system_id] = kl_divergence

        return kl_divergence

        # print("genre_recommendation_counts",self.genre_recommendation_counts[recommender_system_id].keys())
        # print("historical_distribution",self.historical_distribution.keys())
        # print(self.consumer_id, kl_divergence)
'''