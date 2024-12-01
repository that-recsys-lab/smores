import random
from abc import ABC, abstractmethod
from collections import defaultdict
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

from smores.stakeholders.stakeholders import Recommender


class SurpriseSVD(Recommender):

    def __init__(
        self,
        recommender_id,
        fee_per_click=0,
        fee_per_show=0,
        base_fee=0,
        exploration_prob=0.2,
        specialized_categories=None,
        prohibited_categories=None,
        weighted_category=None,
        weighted_value=0.5,
        consumers=None,
        providers=None,
    ):
        """
        Initialize the PopularRecommender.

        Args:
            exploration_prob (float): Probability of exploring (randomly selecting items).
        """
        super().__init__(recommender_id, consumers, providers)
        self.recommender_id = recommender_id
        self.fee_per_click = fee_per_click
        self.fee_per_show = fee_per_show
        self.base_fee = base_fee
        self.exploration_prob = exploration_prob
        self.specialized_categories = specialized_categories or set()
        self.prohibited_categories = prohibited_categories or set()
        self.weighted_category = weighted_category or {}
        self.weighted_value = weighted_value
        self.trainable_model = True
        self.has_trained_model = False
        self.precomputed_recommendations = {}
        self.historical_recommendations = []
        self.profit = []
        self.algo = SVD()

    def train_model_if_ready(self):
        """Train the SVD model once there are enough interactions and precompute recommendations."""
        if len(self.interactions) > 10:  # Arbitrary threshold, adjust as needed
            print("Training model")
            reader = Reader(rating_scale=(0, 1))
            data = Dataset.load_from_df(
                pd.DataFrame(
                    self.interactions, columns=["consumer_id", "doc_id", "rating"]
                ),
                reader,
            )
            trainset = data.build_full_trainset()
            self.algo.fit(trainset)
            self.has_trained_model = True
            print("Model training completed")

            # Precompute recommendations for all users
            self.precompute_recommendations()

    def precompute_recommendations(self):
        """Precompute recommendations for all users and store them in a dictionary."""
        print("Precomputing recommendations for all users")
        # Get all unique users and items
        user_ids = {interaction[0] for interaction in self.interactions}
        item_ids = {doc.doc_id for doc in self.items}

        # Precompute predictions for each user
        for user_id in user_ids:
            predictions = [
                (item_id, self.algo.predict(user_id, item_id).est)
                for item_id in item_ids
            ]
            # Sort items by predicted rating and store the top N items
            sorted_items = sorted(predictions, key=lambda x: x[1], reverse=True)
            self.precomputed_recommendations[user_id] = [
                item_id for item_id, _ in sorted_items
            ]
        print("Precomputation completed")

    def recommend_items(self, consumers, slate_size=5):
        recommendations = {}

        if not self.has_trained_model:
            self.update_items_list()  # Collect items from providers

        for consumer in consumers:
            consumer_id = consumer.consumer_id
            recommended_items = []

            # Get the list of items already clicked by the consumer
            clicked_items = self.consumer_docs_ids_clicked.get(consumer_id, set())

            # Check if we should explore (cold start) or use precomputed recommendations
            if not self.has_trained_model:
                # Cold start: Recommend random items excluding already clicked items
                available_items = [
                    doc for doc in self.items if doc.doc_id not in clicked_items
                ]
                recommended_items = random.sample(
                    available_items, min(slate_size, len(available_items))
                )
            else:
                # Retrieve precomputed recommendations, excluding already clicked items
                top_item_ids = [
                    item_id
                    for item_id in self.precomputed_recommendations.get(consumer_id, [])
                    if item_id not in clicked_items
                ][:slate_size]
                recommended_items = [
                    doc for doc in self.items if doc.doc_id in top_item_ids
                ]

            # Store recommendations and record shows for tracking purposes
            recommendations[consumer_id] = recommended_items
            for document in recommended_items:
                self.record_show(document.provider_id)

        self.historical_recommendations.extend(list(recommendations.values()))

        return recommendations
