import random
from smores.stakeholders.stakeholders import Recommender
from abc import ABC, abstractmethod
from collections import defaultdict


class GenreCalibratedPopularity(Recommender):
    def __init__(
        self,
        recommender_id,
        fee_per_click=0,
        fee_per_show=0,
        base_fee=0,
        exploration_prob=0.2,
        specialized_categories=set(),
        prohibited_categories=set(),
        weighted_category={},
        weighted_value=0.5,
    ):
        """
        The `PopularRecommender` class is a type of recommender system that suggests documents based on their overall popularity,
        with an option to incorporate user-specific preferences and categories. The class balances between exploration
        (recommending less popular or random items) and exploitation (recommending the most popular items)
        to provide a diverse slate of recommendations.

        Key Features:
        - Maintains historical recommendations and tracks user interactions to inform future recommendations.
        - Incorporates fees for clicks, shows, and subscriptions, allowing for profit calculations based on provider interactions.
        - Supports specialized and prohibited categories, as well as weighted categories for fine-tuned recommendations.
        - Utilizes user category preferences and document popularity to filter and prioritize recommendations.
        - Records and logs interactions to monitor recommendation performance and identify gaps (e.g., undersized recommendation slates).

        Methods:
        - `__init__`: Initializes the recommender with configurable parameters, including exploration probability, fees, and category filters.
        - `recommend_documents`: Generates recommendations for a list of consumers, using a mix of exploration and exploitation strategies.
        - `recommender_profit`: Calculates and returns the total profit earned by the recommender.
        - `charge_subscription_fees`: Charges subscription fees to providers based on document interactions (shows and clicks).

        """
        super().__init__()
        self.name = "GenreCalibratedPopularity"
        self.trainable_model = False
        self.recommender_id = recommender_id
        self.exploration_prob = exploration_prob
        self.recommendations = {}
        self.historical_recommendations = []
        self.fee_per_click = fee_per_click
        self.fee_per_show = fee_per_show
        self.base_fee = base_fee
        self.profit = []
        self.specialized_categories = specialized_categories
        self.prohibited_categories = prohibited_categories
        # weighted category variables
        self.weighted_category = (
            weighted_category  # dictionary of category as key and weight as value
        )
        self.weighted_value = weighted_value

    def recommend_documents(self, consumers, slate_size=5):
        """
        Recommend documents to consumers based on overall popularity (highest click counts),
        using a combination of exploration and exploitation.

        Args:
            consumers (list): List of Consumer instances to recommend documents to.
            slate_size (int): Number of documents to recommend per consumer (default is 5).

        Returns:
            dict: Dictionary mapping consumer IDs to lists of recommended documents.
        """
        # Update the documents list
        self.update_documents_list()

        recommendations = {}  # Initialize recommendations dictionary

        for consumer in consumers:
            recommended_documents = []
            consumer_id = consumer.consumer_id
            if consumer_id in self.connected_consumers.keys():
                # if there's a historical category preference
                if self.consumer_category_preferences[consumer_id]:
                    consumer_preferences = self.consumer_category_preferences[
                        consumer_id
                    ]
                    sorted_preferences = sorted(
                        consumer_preferences.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                    top_genres = set([genre for genre, _ in sorted_preferences])

                    if (
                        random.random() < self.exploration_prob
                        and len(self.sorted_documents) > slate_size
                    ):
                        # Explore by randomly selecting documents with consumer's top genres
                        genre_filtered_documents = [
                            doc
                            for doc in self.documents
                            if top_genres.intersection(doc.categories)
                        ]

                        if len(genre_filtered_documents) > slate_size:
                            # if weighted category available
                            if self.weighted_category:
                                weights = [
                                    doc.weight for doc in genre_filtered_documents
                                ]
                                recommended_documents = random.choices(
                                    genre_filtered_documents,
                                    weights=weights,
                                    k=slate_size,
                                )
                            else:
                                recommended_documents = random.sample(
                                    genre_filtered_documents, slate_size
                                )
                        else:
                            # If not enough items in the genre filtered, recommend from the genres the ones the user already liked
                            category_preferences_set = set(
                                self.consumer_category_preferences[consumer_id].keys()
                            )
                            if (
                                category_preferences_set
                            ):  # if the consumer alraedy liked some objects
                                genre_filtered_documents = [
                                    doc
                                    for doc in self.sorted_documents
                                    if category_preferences_set.intersection(
                                        doc.categories
                                    )
                                ]
                                recommended_documents = genre_filtered_documents[
                                    :slate_size
                                ]
                    else:
                        # Exploit by recommending popular documents with consumer's top genres
                        category_preferences_set = set(
                            self.consumer_category_preferences[consumer_id].keys()
                        )
                        if (
                            category_preferences_set
                        ):  # if the consumer alraedy liked some objects
                            genre_filtered_documents = [
                                doc
                                for doc in self.sorted_documents
                                if category_preferences_set.intersection(doc.categories)
                            ]
                            recommended_documents = genre_filtered_documents[
                                :slate_size
                            ]

                # if the list is smaller than slate size extend from popular items
                if len(recommended_documents) < slate_size:
                    diff = slate_size - len(recommended_documents)
                    # if weighted category available
                    if self.weighted_category:
                        random_documents = random.choices(
                            self.documents, weights=self.documents_weights, k=10
                        )
                    else:
                        random_documents = random.sample(self.documents, 10)
                    recommended_documents.extend(
                        [
                            doc
                            for doc in random_documents
                            if doc not in recommended_documents
                        ][:diff]
                    )

                # Record shows for recommended documents
                for document in recommended_documents:
                    self.record_show(document.provider_id)

                recommendations[consumer_id] = recommended_documents

                # LOGGING
                if len(recommendations[consumer_id]) == 0:
                    print("Recommendations dictionary is empty")
                if len(recommendations[consumer_id]) < 5:
                    print("Slate is too small ")

        self.historical_recommendations.extend(list(recommendations.values()))

        return recommendations
