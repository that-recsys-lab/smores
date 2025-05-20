from abc import ABC, abstractmethod
from pydantic import BaseModel

from lenskit.data import Dataset
from lenskit.training import Trainable, TrainingOptions
from lenskit.pipeline import Pipeline, PipelineBuilder, Component
from lenskit.basic.candidates import UnratedTrainingItemsCandidateSelector
from lenskit.basic import UserTrainingHistoryLookup, TopNRanker
from lenskit.knn import ItemKNNConfig, ItemKNNScorer
from lenskit.data import ID, ItemList


class Recommender(ABC):
    def __init__(self):
        self.pipeline: Pipeline
        # Use this if there isn't enough data overall
        self.data_coldstart_component = None
        # Use this if there isn't enough data for a particular user
        self.fallback_component = None

# Does not handle the cold start cases yet.
class LKRecommender(Recommender):
    def __init__(self):
        super().__init__()
        self.pipeline = self.build_pipeline()
        self.lk_config: BaseModel
        self.scorer = None

    def get_scorer(self):
        return self.scorer

    def train(self, dataset: Dataset):
        self.scorer.train(dataset)
    
    
    def build_pipeline(self):
        scorer = self.get_scorer()

        pipe = PipelineBuilder()
        # define an input parameter for the user ID (the 'query')
        query = pipe.create_input('query', ID)
        # allow candidate items to be optionally specified
        items = pipe.create_input('items', ItemList, None)
        # look up a user's history in the training data
        history = pipe.add_component('history-lookup', UserTrainingHistoryLookup, query=query)
        # find candidates from the training data
        default_candidates = pipe.add_component('candidate-selector',
            UnratedTrainingItemsCandidateSelector, query=history)
        # if the client provided items as a pipeline input, use those; otherwise
        # use the candidate selector we just configured.
        candidates = pipe.use_first_of('candidates', items, default_candidates)
        # score the candidate items using the specified scorer
        score = pipe.add_component('scorer', scorer, query=query, items=candidates)
        # rank the items by score
        recommend = pipe.add_component('ranker', TopNRanker, {'n': 50}, items=score)
        pipe.alias('recommender', recommend)
        pipe.default_component('recommender')
        return pipe.build()
    
    @abstractmethod
    def setup(self, config):
        pass

        

class ItemKnn(LKRecommender):
    def __init__(self):
        super().__init__()

    def setup(self, config):
        # get data from config
        max_nbrs = int(config.max_neighbors)
        min_nbrs = int(config.min_neighbors)
        min_sim = float(config.min_similarity)
        # create ItemKNNConfig object
        self.lk_config = ItemKNNConfig(max_nbrs=max_nbrs, min_nbrs=min_nbrs, 
                                       min_sim=min_sim, feedback='implicit')
        # create ItemKNNScorer
        self.scorer = ItemKNNScorer(self.lk_config)






'''
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
'''