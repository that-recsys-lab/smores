import numpy as np
import random
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from smores.stakeholders.choice import category_similarity_logit


class Item:
    def __init__(self, item_id, quality, genres, provider_id, dataset_genres):
        self.item_id = item_id
        self.quality = quality
        self.genres = genres
        self.dataset_genres = dataset_genres
        self.provider_id = provider_id
        self.weight = 0.5
        self.normalized_genres_vector = self.normalize_genres()

    def normalize_genres(self):
        genres_vector = {}

        # Assign weights to the genres based on their position
        if self.genres:
            total_genres = len(self.genres)
            decreasing_weights = [
                (total_genres - i) for i in range(total_genres)
            ]
            total_weight = sum(decreasing_weights)
            normalized_weights = [weight / total_weight for weight in decreasing_weights]

            for cat, weight in zip(self.genres, normalized_weights):
                if cat in self.dataset_genres:
                    genres_vector[cat] = weight

        return genres_vector

    def __str__(self):
        return (
            f"Item ID: {self.item_id}, Quality: {self.quality}, "
            f"genres: {self.genres}, Provider ID: {self.provider_id}, "
            f"Normalized genres Vector: {self.normalized_genres_vector}"
        )


class Provider:
    def __init__(self, provider_id, items):
        """
        Initialize a Provider object with items, profit, and other attributes.

        Args:
            provider_id (int): Unique identifier for the provider.
            items (list): List of Item objects representing the items offered by the provider.
            profit (float, optional): Initial profit (default is 0).
        """
        self.provider_id = provider_id
        self.items = items
        self.profit = {}
        self.recommender_counts = {}
        self.connected_recommenders = {}
        self.utility_per_show = 0.1
        self.utility_per_click = 0.4
        self.pay_cycle_fee = {}
        self.pay_cycle_clicks = {}
        self.pay_cycle_shows = {}
        self.adv_budget_per_document = np.random.lognormal(
            mean=9, sigma=1
        )  # Log-normal distribution for budget

    def subscribe_to_recommender_system(self, recommender_system_id):
        """
        Subscribe to a new recommender system and initialize the profit and count.

        Args:
            recommender_system_id (str): Identifier for the new recommender system.
        """
        self.connected_recommenders[recommender_system_id] = 1
        self.profit[recommender_system_id] = []  # Initialize profit as a list
        self.recommender_counts[recommender_system_id] = 0  # Initialize count to 0

    def unsubscribe_from_recommender_system(self, recommender_system_id):
        """
        Unsubscribe from a recommender system.

        Args:
            recommender_system_id (str): Identifier of the recommender system to unsubscribe from.
        """
        self.connected_recommenders[recommender_system_id] = 0

    def update_recommender_subscription(self):
        """
        Choose recommender systems based on historical profit

        Returns:
            unsubscribed_list (list): List of the recommenders the provider unsubscribed from
        """
        unsubscribed_list = []
        for recommender in self.connected_recommenders.keys():
            if self.connected_recommenders[recommender] == 1:
                # If the provider starts losing money, then unsubscribe from the recommender
                if self.profit[recommender] and self.profit[recommender][-1] < 0:
                    self.unsubscribe_from_recommender_system(recommender)
                    unsubscribed_list.append(recommender)
        return unsubscribed_list

    def update_profit(
        self,
        recommender_system_id,
        base_fee=0,
        total_fee=0,
        click_count=0,
        fee_per_click=0,
        show_count=0,
        fee_per_show=0,
    ):
        """
        Update the orofit of the provider for a specific recommender system based on clicks and total revenue.

        Args:
            recommender_system_id (str): Identifier of the recommender system.
            total_fee (float): Total fee charged by the recommender system.
            click_count (int): Number of clicks recorded.
            fee_per_click (float): Fee charged per click.
            show_count (int): Number of shows recorded.
            fee_per_show (float): Fee charged per show.
        """
        if recommender_system_id not in self.profit:
            raise ValueError(
                f"Recommender system '{recommender_system_id}' is not connected to this provider."
            )

        # Calculate the recommender utility
        recommender_utility = (
            self.utility_per_click * click_count + self.utility_per_show * show_count
        ) - total_fee

        # Append the utility to the list of utilities for the recommender system
        self.profit[recommender_system_id].append(recommender_utility)

    def charge_subscription_fee(
        self,
        recommender_system_id,
        base_fee=0,
        total_fee=0,
        click_count=0,
        fee_per_click=0,
        show_count=0,
        fee_per_show=0,
    ):
        """
        Charge subscription fee to the provider based on the number of clicks received.

        Args:
            recommender_system_id (str): Identifier of the recommender system.
            total_fee (float): Total fee charged by the recommender system.
            click_count (int): Number of clicks recorded.
            fee_per_click (float): Fee charged per click.
            show_count (int): Number of shows recorded.
            fee_per_show (float): Fee charged per show.
        """
        if recommender_system_id not in self.profit:
            raise ValueError(
                f"Recommender system '{recommender_system_id}' is not connected to this provider."
            )

        self.pay_cycle_fee.setdefault(recommender_system_id, []).append(total_fee)
        self.pay_cycle_clicks.setdefault(recommender_system_id, []).append(click_count)
        self.pay_cycle_shows.setdefault(recommender_system_id, []).append(show_count)

        # Update profit based on the total fee received
        self.update_profit(
            recommender_system_id,
            base_fee,
            total_fee,
            click_count,
            fee_per_click,
            show_count,
            fee_per_show,
        )

    def get_profit(self, recommender_system_id):
        """
        Get the profit of the provider for a specific recommender system.

        Args:
            recommender_system_id (str): Identifier of the recommender system.

        Returns:
            float: profit for the specified recommender system.
        """
        if recommender_system_id not in self.profit:
            raise ValueError(
                f"Recommender system '{recommender_system_id}' is not connected to this provider."
            )

        # Calculate the average utility
        average_utility = np.mean(self.profit[recommender_system_id])
        return average_utility

    def reset(self):
        self.profit = {}
        self.recommender_counts = {}
        self.connected_recommenders = {}
        self.pay_cycle_fee = {}
        self.pay_cycle_clicks = {}
        self.pay_cycle_shows = {}

    def __str__(self):
        """
        Return a string representation of the Provider object.
        """
        return f"Provider {self.provider_id}: items={self.items}, genres={self.genres}"


class Consumer:
    def __init__(
        self,
        consumer_id,
        sensitivity,
        category_preferences,
        prohibited_genres=set(),
        favorite_genres=set(),
        choice_model=category_similarity_logit,
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
        self.choice_model = choice_model
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
            selected_index = self.choice_model(
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
        
    def remove_user(self, retain_profile=True):
        for recommender_id in list(self.connected_recommenders.keys()):
            self.unsubscribe_from_recommender_system(recommender_id)
        if not retain_profile:
            self.net_quality_exposure = {}
            self.satisfaction_scores = {}
            self.recommender_counts = {}
            self.recommender_category_success = {}
            self.ucb_scores = {}
            self.genre_recommendation_counts = {}

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
            self.satisfaction_scores.get(recommender_system_id, 0) * self.beta
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
            q = historical_distribution_normalized.get(genre, 0.0)  # Default to 0 if genre is not in historical_distribution
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


class Recommender(ABC):

    def __init__(
        self,
        recommender_id,
        consumers=None,
        providers=None,
        most_popular_item_ids=None,
    ):
        """
        Initialize the Recommender.
        """
        self.recommender_id = recommender_id
        self.consumers = consumers if consumers is not None else []
        self.providers = providers if providers is not None else []
        self.connected_providers = {}
        self.connected_consumers = {}
        self.clicks = defaultdict(int)
        self.shows = defaultdict(int)
        self.provider_docs_clicked = defaultdict(list)
        self.consumer_docs_clicked = defaultdict(set)
        self.consumer_docs_ids_clicked = defaultdict(set)
        self.consumer_category_preferences = defaultdict(lambda: defaultdict(int))
        self.items = []
        self.items_click_counts = defaultdict(int)
        self.sorted_items = []
        self.weighted = False
        self.weighted_category = {}
        self.items_weights = []
        self.interactions = []
        self.initialize_recommender()
        self.most_popular_item_ids = most_popular_item_ids

    def initialize_recommender(self):
        """
        Subscribe consumers and providers to the recommender.
        """
        print("Initializing recommender", self.recommender_id)
        for consumer in self.consumers:
            self.connect_consumer(consumer)
            consumer.subscribe_to_recommender_system(self.recommender_id)

        for provider in self.providers:
            self.connect_provider(provider)
            provider.subscribe_to_recommender_system(self.recommender_id)

    def add_interaction(self, consumer_id, item_id, rating):
        self.interactions.append((consumer_id, item_id, rating))

    def get_user_interactions(self, user_id):
        return [(cid, iid, rating) for cid, iid, rating in self.interactions if cid == user_id]

    def remove_user_interactions(self, user_id):
        self.interactions = [(cid, iid, rating) for cid, iid, rating in self.interactions if cid != user_id]
        
    @abstractmethod
    def recommend_items(self, consumers, slate_size=1):
        """
        Abstract method to recommend items to consumers.

        Args:
            consumers (list): List of Consumer instances to recommend items to.
            slate_size (int): Number of items to recommend per consumer (default is 1).

        Returns:
            dict: Dictionary mapping consumer IDs to lists of recommended items.
        """
        pass

    @abstractmethod
    def charge_subscription_fees(self, fee_per_click):
        """
        Abstract method to charge subscription fees based on clicks.

        Args:
            fee_per_click (float): Fee amount charged per click.
        """
        pass

    def update_items_list(self, force_update=False):
        if not self.sorted_items or force_update:
            if self.specialized_genres:  # Check if there are specialized genres
                self.items = [
                    item
                    for provider in self.connected_providers.values()
                    for item in provider.items
                    if item.genres.intersection(
                        self.specialized_genres
                    )  # Check if item has any specialized genres
                    and not self.prohibited_genres.intersection(
                        item.genres
                    )  # Check if item has any prohibited genres
                ]
            else:
                self.items = [
                    item
                    for provider in self.connected_providers.values()
                    for item in provider.items
                    if not self.prohibited_genres.intersection(
                        item.genres
                    )  # Check if item has any prohibited genres
                ]

            # Update weights if weighted category available
            if self.weighted_category:
                self.set_items_weights()

            # Sort the items by click counts in descending order
            self.sorted_items = sorted(
                self.items,
                key=lambda item: self.items_click_counts.get(item, 0),
                reverse=True,
            )

    def set_items_weights(self):
        self.items_weights = []
        weighted_key, weighted_value = list(self.weighted_category.items())[0]
        for item in self.items:
            if weighted_key in item.genres:
                item.weight = weighted_value
            self.items_weights.append(item.weight)

    def connect_provider(self, provider):
        """
        Add a new provider to the recommender.

        Args:
            provider (Provider): Provider instance to add.
        """
        if provider.provider_id not in self.connected_providers:
            self.connected_providers[provider.provider_id] = provider
            self.shows[provider.provider_id] = 0
            self.clicks[provider.provider_id] = 0
            self.provider_docs_clicked[provider.provider_id] = (
                []
            )  # Initialize clicked items list for the provider
        self.sorted_items = []

    def disconnect_provider(self, provider_id):
        """
        Remove a provider from the recommender.

        Args:
            provider_id (int): ID of the provider to remove.
        """
        print("Recommender", self.recommender_id, "diconnected provider", provider_id)
        if provider_id in self.connected_providers:
            for item_id in self.provider_docs_clicked[provider_id]:
                del self.items_click_counts[item_id]
            del self.connected_providers[provider_id]
            del self.clicks[provider_id]
            del self.provider_docs_clicked[provider_id]
        self.sorted_items = []

        # Unsubscribe all consumers if provider list is empty
        if len(self.connected_providers) <= 0:
            print("Deleting all consumers from recommender", self.recommender_id)
            consumers_to_disconnect = list(self.connected_consumers.values())
            for consumer in consumers_to_disconnect:
                self.disconnect_consumer(consumer)

    def connect_consumer(self, consumer):
        """
        Connect a consumer to the recommender system.

        Args:
            consumer (Consumer): Consumer instance to connect.
        """
        if consumer.consumer_id not in self.connected_consumers:
            self.connected_consumers[consumer.consumer_id] = consumer
            self.consumer_docs_clicked[consumer.consumer_id] = (
                set()
            )  # Initialize clicked items set for the consumer
            self.consumer_docs_ids_clicked[consumer.consumer_id] = (
                set()
            )  # Initialize clicked item IDs set

    def disconnect_consumer(self, consumer):
        """
        Disconnect a consumer from the recommender system.

        Args:
            consumer_id (int): ID of the consumer to disconnect.
        """
        if consumer in self.connected_consumers.values():
            del self.connected_consumers[consumer.consumer_id]
            consumer.unsubscribe_from_recommender_system(self.recommender_id)

    def record_click(self, consumer_id, clicked_document):
        clicked_document_id = clicked_document.item_id
        clicked_provider_id = clicked_document.provider_id

        if clicked_provider_id in self.connected_providers:
            self.clicks[clicked_provider_id] += 1
            if clicked_document not in self.provider_docs_clicked[clicked_provider_id]:
                self.provider_docs_clicked[clicked_provider_id].append(clicked_document)
            
            # Add the clicked item to both tracking dictionaries
            self.consumer_docs_clicked[consumer_id].add(clicked_document)
            self.consumer_docs_ids_clicked[consumer_id].add(clicked_document_id)

            if clicked_document not in self.items_click_counts:
                self.items_click_counts[clicked_document] = 0
            self.items_click_counts[clicked_document] += 1

            self.update_user_category_preferences(consumer_id, clicked_document)
            self.sorted_items = []

        else:
            print(
                f"Record Click - Provider with ID {clicked_provider_id} is not connected to the recommender."
            )

    def record_show(self, provider_id):
        """
        Record that a item from a specific provider has been recommended.

        Args:
            provider_id (int): ID of the provider whose item is recommended.
        """
        if provider_id in self.connected_providers:
            self.shows[provider_id] += 1
        else:
            print(
                f"Record Show - Provider with ID {provider_id} is not connected to the recommender."
            )

    def update_user_category_preferences(self, consumer_id, clicked_document):
        """
        Update the category preferences for a user based on the clicked item.

        Args:
            consumer_id (int): ID of the user.
            clicked_document (Item): The item that the user clicked on.
        """
        # Get the genres of the clicked item
        clicked_document_genres = clicked_document.genres

        # Update the user's category preferences
        if consumer_id not in self.consumer_category_preferences:
            self.consumer_category_preferences[consumer_id] = defaultdict(int)

        for category in clicked_document_genres:
            if category in self.consumer_category_preferences[consumer_id]:
                self.consumer_category_preferences[consumer_id][category] += 1
            else:
                self.consumer_category_preferences[consumer_id][category] = 1

    def get_user_category_preferences(self, consumer_id):
        """
        Get the category preferences for a user.

        Args:
            consumer_id (int): ID of the user.

        Returns:
            dict: Dictionary of genres and associated probabilities for the user.
        """
        return self.consumer_category_preferences.get(consumer_id, {})

    def print_providers_clicked_items(self):
        """
        Print information about clicked items for each provider.
        """
        print("Clicked items by Providers:")
        for provider_id, clicked_items in self.provider_docs_clicked.items():
            clicked_document_titles = [
                item.title for item in clicked_items
            ]  # Extract item titles
            clicked_document_titles_str = ", ".join(clicked_document_titles)
            print(
                f"Provider ID: {provider_id} - Clicked items: {clicked_document_titles_str}"
            )

    def print_consumers_clicked_items(self):
        """
        Print information about clicked items for each consumer.
        """
        print("Clicked items by Consumers:")
        for consumer_id, clicked_items in self.consumer_docs_clicked.items():
            clicked_document_titles = [
                item.title for item in clicked_items
            ]  # Extract item titles
            clicked_document_titles_str = ", ".join(clicked_document_titles)
            print(
                f"Consumer {consumer_id}: Clicked items: {clicked_document_titles_str}"
            )

    def print_providers(self):
        """
        Print information about the current providers in the recommender.
        """
        print("Providers Information:")
        for provider in self.connected_providers:
            print(provider)

    def print_consumers(self):
        """
        Print information about the current consumers in the recommender.
        """
        print("Consumers Information:")
        for consumer_id, consumer in self.connected_consumers.items():
            print(
                f"Consumer {consumer_id}:"
                f"Sensitivity={consumer.sensitivity}, Net Quality Exposure={consumer.net_quality_exposure:.2f}"
            )

    def print_clicks(self):
        """
        Print information about the clicks recorded by providers.
        """
        print("Clicks Information:")
        for provider_id, click_count in self.clicks.items():
            print(f"Provider (ID: {provider_id}) - Clicks: {click_count}")

    def print_connected_consumers(self):
        """
        Print information about the connected consumers.
        """
        print("Connected Consumers:")
        for consumer_id in self.connected_consumers:
            print(f"Consumer ID: {consumer_id}")

    def print_connected_providers(self):
        """
        Print information about the connected providers.
        """
        print("Connected Providers:")
        for provider_id, provider in self.connected_providers.items():
            print(f"Provider ID: {provider_id}")

    def print_recommendations(self, recommendations):
        """
        Print recommendations made by the recommender to consumers.

        Args:
            recommendations (dict): Dictionary mapping consumer IDs to lists of recommended items.
        """
        for consumer_id, recommended_items in recommendations.items():
            consumer = self.connected_consumers[consumer_id]
            print(f"Consumer {consumer_id} - Recommended items:")
            for item in recommended_items:
                print(f"- {item}")

    def recommender_profit(self):
        """
        Returns the profit earned by the recommender.
        """
        return self.profit

    def charge_subscription_fees(self):
        """
        Charge subscription fees to providers based on the number of clicks received.
        """
        cycle_profit = 0
        for provider_id, provider in self.connected_providers.items():
            show_count = self.shows.get(provider_id, 0)
            if show_count >= 0:
                click_count = self.clicks.get(provider_id, 0)
                total_fee = self.base_fee + (
                    self.fee_per_click * click_count + self.fee_per_show * show_count
                )
                cycle_profit += total_fee  # Update profit attribute
                provider.charge_subscription_fee(
                    self.recommender_id,
                    self.base_fee,
                    total_fee,
                    click_count,
                    self.fee_per_click,
                    show_count,
                    self.fee_per_show,
                )
                # Reset show count and click count
                self.shows[provider_id] = 0
                self.clicks[provider_id] = 0
        self.profit.append(cycle_profit)

        # Train the model after a full cycle
        print("Train model", self.trainable_model)
        if self.trainable_model:
            self.train_model_if_ready()

    def add_interaction(self, consumer_id, item_id, rating):
        """
        Add a new interaction and retrain the model if enough data is collected.

        Args:
            consumer_id (): The consumer interacted with the item.
            item_id (): the item the user is interacting with.
            rating (boolean): 1 if the consumer clicked on the item, 0 otherwise.

        """
        self.interactions.append((consumer_id, item_id, rating))