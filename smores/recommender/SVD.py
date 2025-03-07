import random
import os
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import heapq
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csr_matrix
from smores.stakeholders.stakeholders import Recommender


class SurpriseSVD(Recommender):
    def __init__(
        self,
        recommender_id,
        most_popular_item_ids,
        fee_per_click=0,
        fee_per_show=0,
        base_fee=0,
        exploration_prob=0.2,
        specialized_genres=None,
        prohibited_genres=None,
        weighted_category=None,
        weighted_value=0.5,
        consumers=None,
        providers=None,
    ):
        super().__init__(recommender_id, consumers, providers, most_popular_item_ids)
        self.recommender_id = recommender_id
        self.fee_per_click = fee_per_click
        self.fee_per_show = fee_per_show
        self.base_fee = base_fee
        self.exploration_prob = exploration_prob
        self.specialized_genres = specialized_genres or set()
        self.prohibited_genres = prohibited_genres or set()
        self.weighted_category = weighted_category or {}
        self.weighted_value = weighted_value
        self.trainable_model = True
        self.has_trained_model = False
        self.precomputed_recommendations_dir = "tmp"
        self.precomputed_recommendations = {}
        self.historical_recommendations = []
        self.profit = []
        self.algo = SVD()
        self.user_clusters = {}

        # Create the tmp directory if it doesn't exist
        os.makedirs(self.precomputed_recommendations_dir, exist_ok=True)

    def train_model_if_ready(self):
        """Train the SVD model once there are enough interactions and generate user clusters."""
        print("Training model if ready")
        if len(self.interactions) > 10:  # Arbitrary threshold, adjust as needed
            print("Interactions summary")
            print(
                pd.DataFrame(
                    self.interactions, columns=["consumer_id", "item_id", "rating"]
                )["rating"].describe()
            )
            print("Training model")
            reader = Reader()
            df = pd.DataFrame(
                self.interactions, columns=["consumer_id", "item_id", "rating"]
            )

            num_missing_interactions = len(
                set(consumer.consumer_id for consumer in self.consumers)
                - set(df["consumer_id"].unique())
            )
            print(
                f"Number of consumers missing in interactions dataset: {num_missing_interactions}"
            )

            data = Dataset.load_from_df(df, reader)
            trainset = data.build_full_trainset()
            self.algo.fit(trainset)

            # Create user-item matrix for clustering using a sparse matrix
            consumer_map = {
                id: idx for idx, id in enumerate(df["consumer_id"].unique())
            }
            item_map = {id: idx for idx, id in enumerate(df["item_id"].unique())}

            df["consumer_idx"] = df["consumer_id"].map(consumer_map)
            df["item_idx"] = df["item_id"].map(item_map)

            user_item_matrix = coo_matrix(
                (df["rating"], (df["consumer_idx"], df["item_idx"])),
                shape=(len(consumer_map), len(item_map)),
            ).tocsr()

            self._create_user_clusters(user_item_matrix, consumer_map)

            self.has_trained_model = True
            print("Model training completed")
            self.compute_and_store_recommendations()
            self.load_recommendations()
        else:
            print("Not enough interactions to train the model yet")

    def _create_user_clusters(self, user_item_matrix, consumer_map):
        """Cluster users based on their clicked items using KNN."""
        print("Clustering users based on their clicked items")

        # Calculate the number of neighbors
        num_users = user_item_matrix.shape[0]
        n_neighbors = 5

        # Convert user-item matrix to sparse format
        if not isinstance(user_item_matrix, csr_matrix):
            user_item_matrix = csr_matrix(user_item_matrix)

        # Fit KNN model
        knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
        knn.fit(user_item_matrix)

        # Generate sparse neighbors graph
        neighbors_graph = knn.kneighbors_graph(user_item_matrix, mode="connectivity")

        # Create reverse consumer map with the correct size
        max_index = max(consumer_map.values())
        reverse_consumer_map = np.zeros(max_index + 1, dtype=int)
        for consumer_id, idx in consumer_map.items():
            reverse_consumer_map[idx] = consumer_id

        # Build user clusters robustly
        self.user_clusters = {}
        for user_idx, neighbors in enumerate(
            np.split(neighbors_graph.indices, neighbors_graph.indptr[1:-1])
        ):
            if user_idx >= len(reverse_consumer_map):
                print(f"Skipping user_idx {user_idx} as it's out of bounds.")
                continue
            self.user_clusters[reverse_consumer_map[user_idx]] = reverse_consumer_map[
                neighbors
            ].tolist()

        print("User clustering completed.")


    def compute_and_store_recommendations(self, batch_size=100):
        """Precompute recommendations for all users based on their clusters."""
        print("Precomputing recommendations for all users")
        user_ids = list(self.user_clusters.keys())
        item_ids = [item.item_id for item in self.items]

        for file in os.listdir(self.precomputed_recommendations_dir):
            os.remove(os.path.join(self.precomputed_recommendations_dir, file))

        for start_idx in tqdm(
            range(0, len(user_ids), batch_size), desc="Processing batches", unit="batch"
        ):
            batch_user_ids = user_ids[start_idx : start_idx + batch_size]
            batch_predictions = {}

            for user_id in batch_user_ids:
                # Aggregate liked items from users in the same cluster
                cluster_items = set()
                for cluster_user in self.user_clusters[user_id]:
                    cluster_items.update(
                        self.consumer_docs_ids_clicked.get(cluster_user, set())
                    )

                # Ensure at least 100 unseen items for the user
                unseen_items = cluster_items - self.consumer_docs_ids_clicked.get(
                    user_id, set()
                )
                if len(unseen_items) < 100:
                    unseen_items.update(
                        random.sample(item_ids, 100 - len(unseen_items))
                    )

                predictions = {
                    item_id: predicted_rating
                    for item_id, predicted_rating in (
                        (item_id, self.algo.predict(user_id, item_id).est)
                        for item_id in unseen_items
                    )
                }
                # slate_size, number of days, buffer
                top_n = 5 * 30 * 3  # Precompute enough for 30 days of recommendations
                top_predictions = heapq.nlargest(
                    top_n, predictions.items(), key=lambda x: x[1]
                )
                batch_predictions[int(user_id)] = top_predictions

            batch_file = os.path.join(
                self.precomputed_recommendations_dir, f"batch_{start_idx}.json"
            )
            with open(batch_file, "w") as f:
                json.dump(batch_predictions, f)

        print("Precomputation completed and stored on disk")

    def load_recommendations(self):
        """Load precomputed recommendations from disk into memory."""
        print("Loading precomputed recommendations from disk")
        self.precomputed_recommendations = {}

        for file in os.listdir(self.precomputed_recommendations_dir):
            file_path = os.path.join(self.precomputed_recommendations_dir, file)
            with open(file_path, "r") as f:
                batch_recommendations = json.load(f)
                self.precomputed_recommendations.update(batch_recommendations)
        print(
            "Recommendations loaded into memory, length:",
            len(self.precomputed_recommendations),
        )

    def recommend_items(self, consumers, slate_size=5):
        """Generate recommendations for consumers based on user clusters."""
        recommendations = {}

        if not self.has_trained_model and not self.items:
            print("Collect items from providers")
            self.update_items_list()
            print(f"{self.recommender_id}Number of items: {len(self.items)}")

        if not self.precomputed_recommendations and self.has_trained_model:
            print("No precomputed recommendations found. Loading from disk...")
            self.load_recommendations()

        item_map = {item.item_id: item for item in self.items}
        all_item_ids = set(item_map.keys())
        popular_item_ids = set(self.most_popular_item_ids)
        popular_item_ids = popular_item_ids.intersection(all_item_ids)
        

        # for consumer in tqdm(consumers, desc=f"Recommending items from {self.recommender_id}", unit="consumer"):
        for consumer in consumers:
            consumer_id = consumer.consumer_id
            clicked_items = self.consumer_docs_ids_clicked.get(consumer_id, set())

            if not self.has_trained_model:

                available_item_ids = popular_item_ids.difference(clicked_items)

                sampled_item_ids = (
                    random.sample(sorted(available_item_ids), slate_size)
                    if len(available_item_ids) > slate_size
                    else list(available_item_ids)
                )
                    
                recommended_items = [item_map[item_id] for item_id in sampled_item_ids]

            else:
                precomputed = self.precomputed_recommendations.get(str(consumer_id), [])

                filtered_item_ids = (
                    item_id
                    for item_id, _ in precomputed
                    if item_id not in clicked_items
                )
                sampled_item_ids = list(filtered_item_ids)[:slate_size]

                recommended_items = [item_map[item_id] for item_id in sampled_item_ids]

                # if not enough recommendations, sample randomly
                if not recommended_items or len(recommended_items) < slate_size:
                    available_item_ids = popular_item_ids.difference(clicked_items)

                    sampled_item_ids = (
                        random.sample(list(available_item_ids), slate_size)
                        if len(available_item_ids) > slate_size
                        else list(available_item_ids)
                    )
                    recommended_items = [
                        item_map[item_id] for item_id in sampled_item_ids
                    ]
                
                if len(recommended_items) < slate_size:
                    print(f"Consumer: {consumer_id}. Recommender: {self.recommender_id}. items: {len(self.items)}. Recommended items: {len(recommended_items)}. Precomputed items: {len(precomputed)}. Available items: {len(available_item_ids)}. sampled items: {len(sampled_item_ids)}")
            
            recommendations[consumer_id] = recommended_items
            for document in recommended_items:
                self.record_show(document.provider_id)

        self.historical_recommendations.extend(recommendations.values())
        return recommendations
