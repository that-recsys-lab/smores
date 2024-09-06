import numpy as np
import random
from abc import ABC, abstractmethod
from collections import defaultdict

class Document:
    def __init__(self, doc_id, quality, categories, provider_id):
        self.doc_id = doc_id
        self.quality = quality
        self.categories = categories
        self.provider_id = provider_id
        self.weight = 0.5
        self.normalized_categories_vector = self.normalize_categories()
        
    def normalize_categories(self):
        categories_vector = {}
        categories_list = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        for cat in categories_list:
            if cat in self.categories:
                categories_vector[cat] = 1.0
        return self.normalize(categories_vector)
    
    @staticmethod
    def normalize(vector):
        total = sum(vector.values())
        if total == 0:
            return {key: 0 for key in vector}  # Handle case where total is zero
        return {key: value / total for key, value in vector.items()}
        
    def __str__(self):
        return f"Document ID: {self.doc_id}, Quality: {self.quality}, Categories: {self.categories}, Provider ID: {self.provider_id}"


class Provider:
    def __init__(self, provider_id, documents):
        """
        Initialize a Provider object with documents, profit, and other attributes.

        Args:
            provider_id (int): Unique identifier for the provider.
            documents (list): List of Document objects representing the documents offered by the provider.
            profit (float, optional): Initial profit (default is 0).
        """
        self.provider_id = provider_id
        self.documents = documents
        self.profit = {}
        self.recommender_counts = {}
        self.connected_recommenders = {}
        self.utility_per_show = 0.1
        self.utility_per_click = 0.4
        self.pay_cycle_fee = {}
        self.pay_cycle_clicks = {}
        self.pay_cycle_shows = {}
        self.adv_budget_per_document = np.random.lognormal(mean=9, sigma=1)  # Log-normal distribution for budget

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
    
    def update_profit(self, recommender_system_id, base_fee=0, total_fee=0, click_count=0, fee_per_click=0, show_count=0, fee_per_show=0):
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
            raise ValueError(f"Recommender system '{recommender_system_id}' is not connected to this provider.")

        # Calculate the recommender utility
        recommender_utility = (self.utility_per_click * click_count + self.utility_per_show * show_count) - total_fee

        # Append the utility to the list of utilities for the recommender system
        self.profit[recommender_system_id].append(recommender_utility)

    def charge_subscription_fee(self, recommender_system_id, base_fee=0, total_fee=0, click_count=0, fee_per_click=0, show_count=0, fee_per_show=0):
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
            raise ValueError(f"Recommender system '{recommender_system_id}' is not connected to this provider.")
            
        self.pay_cycle_fee.setdefault(recommender_system_id, []).append(total_fee)
        self.pay_cycle_clicks.setdefault(recommender_system_id, []).append(click_count)
        self.pay_cycle_shows.setdefault(recommender_system_id, []).append(show_count)

        # Update profit based on the total fee received
        self.update_profit(recommender_system_id, base_fee, total_fee, click_count, fee_per_click, show_count, fee_per_show)
        
    def get_profit(self, recommender_system_id):
        """
        Get the profit of the provider for a specific recommender system.

        Args:
            recommender_system_id (str): Identifier of the recommender system.

        Returns:
            float: profit for the specified recommender system.
        """
        if recommender_system_id not in self.profit:
            raise ValueError(f"Recommender system '{recommender_system_id}' is not connected to this provider.")

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
        return f"Provider {self.provider_id}: Documents={self.documents}, Categories={self.categories}"
    

class Consumer:
    def __init__(self, consumer_id, sensitivity, category_preferences, prohibited_categories=set(), favorite_categories=set()):
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
        self.satisfaction_scores = {}  # Dictionary to store satisfaction scores per recommender system
        self.recommender_counts = {}  # Dictionary to store how many times the recommender system has been chosen
        self.recommender_category_success = {} # Dictionary to store how many times the recommender system got the favorite category
        self.ucb_scores = {}  # Store UCB scores for each recommender system
        self.prohibited_categories = prohibited_categories # Set to store categories penalized by the consumer
        self.favorite_categories = favorite_categories # Set to store categories the consumer likes
        self.apha = 0.5
        self.beta = 2 # recency bias
        
    def subscribe_to_recommender_system(self, recommender_system_id):
        """
        Subscribe to a new recommender system and initialize the satisfaction score and count.

        Args:
            recommender_system_id (str): Identifier for the new recommender system.
        """
        self.connected_recommenders[recommender_system_id] = 1
        self.satisfaction_scores[recommender_system_id] = self.satisfaction_scores.get(recommender_system_id, 0) 
        self.recommender_counts[recommender_system_id] = self.recommender_counts.get(recommender_system_id, 0)  # get the value that is in self.recommender_counts[recommender_system_id] otherwise set as zero
        self.recommender_category_success[recommender_system_id] = 0
        
    def unsubscribe_from_recommender_system(self, recommender_system_id):
        """
        Unsubscribe from a recommender system.

        Args:
            recommender_system_id (str): Identifier of the recommender system to unsubscribe from.
        """
        self.connected_recommenders[recommender_system_id] = 0
    
    def choose_recommender(self):
        """
        Choose a recommender system based on the Upper Confidence Bound (UCB) algorithm.

        Returns:
            str: ID of the selected recommender system.
        """
        # Filter connected recommender systems with value 1
        self.available_recommenders = [recommender_id for recommender_id, value in self.connected_recommenders.items() if value == 1]

        if not self.available_recommenders:
            print('no available recommenders')
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
        round_ucb_scores = { recommender_id: score for recommender_id, score in self.ucb_scores.items()}
        for recommender_id in self.available_recommenders:
            if self.recommender_counts[recommender_id] == 0:
                round_ucb_scores[recommender_id] = float('inf')  # Prioritize unselected recommenders
            else:
                exploitation_term = self.satisfaction_scores[recommender_id] + self.apha
                # Adjusted UCB; added decay 
                exploration_term = np.sqrt(np.log(sum(self.recommender_counts.values())) / self.recommender_counts[recommender_id]) / (1 + sum(self.recommender_counts.values()))
                # update the ucb only for the selected recommender
                round_ucb_scores[recommender_id] = exploitation_term + exploration_term

        # Select the recommender with the highest UCB score
        available_ucb_scores = {recommender_id: score for recommender_id, score in round_ucb_scores.items() if recommender_id in self.available_recommenders}
        selected_recommender_id = max(available_ucb_scores, key=available_ucb_scores.get)

        # Update the count and ucb_score for the selected recommender
        self.ucb_scores[recommender_id] = round_ucb_scores[recommender_id]
        self.recommender_counts[selected_recommender_id] += 1

        return selected_recommender_id

          
    def choice_model(self, documents):
        """
        Args:
            documents (list): List of documents to evaluate.

        Returns:
            int: Index of the selected document in the input list.
        """
        probabilities = self._score_documents(documents)
    
        if np.all(probabilities == 0):
            return None
        else:
            # Select an index based on the computed probabilities
            selected_index = np.random.choice(len(documents), p=probabilities)

            return selected_index
    
    def _score_documents(self, documents):
        """
        Score the given list of documents using a multinomial logit choice model.

        Args:
            documents (list): List of documents to score.

        Returns:
            np.array: Array of scores corresponding to each document.
        """
        # Placeholder for document scores
        scores = np.zeros(len(documents))
        
        # Calculate utility for each document based on category similarity and document position
        for i, doc in enumerate(documents):
            category_similarity = 0.0
            for category in doc.categories:
                if category in self.prohibited_categories:
                    category_similarity -= 1
                else:
                    category_similarity += self.category_preferences.get(category, 0)
            category_similarity /= len(doc.categories) if len(doc.categories) > 0 else 1  # Handle zero division
            scores[i] = category_similarity
            
        probabilities = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))
        return probabilities
    
    def simulate_response(self, slate_documents, recommender_system_id):
        """
        Simulate response to a slate of documents and return the responses.

        Args:
            slate_documents (list): List of documents to evaluate.
            recommender_system (Recommender): Recommender system instance.

        Returns:
            list: List of response dictionaries corresponding to the documents.
        """
        responses = []

        # Determine whether the user will click on anything
        click_prob = np.random.random()
        if click_prob <= 1:  # 100% chance of clicking
            selected_index = self.choice_model(slate_documents)
            if selected_index is not None:
                for i, doc in enumerate(slate_documents):
                    if i == selected_index:
                        responses.append({'click': 1})  # Mark the selected document as clicked
                    else:
                        responses.append({'click': 0})  # Mark other documents as not clicked
            else:
                # If choice model returned None, return all zeros
                responses = [{'click': 0} for _ in range(len(slate_documents))]
        else:
            # User didn't click on anything
            responses = [{'click': 0} for _ in range(len(slate_documents))]

        # update state
        self.update_state(slate_documents, responses, recommender_system_id)

        return responses

    def update_state(self, slate_documents, responses, recommender_system_id):
        """
        Update the net quality exposure (nqe) for the user after choosing a document.

        Args:
            slate_documents (list): List of documents presented to the user.
            responses (list): List of response dictionaries corresponding to the presented documents.
            recommender_system (Recommender): Recommender system instance.
        """        
        sim_score = 0
        for doc in slate_documents:
            # dot product of the intersection between the user interest and the item features for each document in the slate
            intersecting_keys = set(self.category_preferences.keys()).intersection(set(doc.normalized_categories_vector.keys()))
            # Compute the dot product only for the intersecting keys 
            sim_score += sum(self.category_preferences[key] * doc.normalized_categories_vector[key] for key in intersecting_keys)
        # normalize sim_score
        norm_sim_score = sim_score / len(slate_documents)
        
        self.satisfaction_scores[recommender_system_id] = (self.satisfaction_scores.get(recommender_system_id, 0) * self.beta + norm_sim_score) / (1 + self.beta)
                    
                
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
        self.satisfaction_scores = {}  # Dictionary to store satisfaction scores per recommender system
        self.recommender_counts = {}  # Dictionary to store how many times the recommender system has been chosen
        self.ucb_scores = {}  # Store UCB scores for each recommender system
        

class Recommender(ABC):
    def __init__(self):
        """
        Initialize the Recommender.
        """
        self.connected_providers = {}
        self.clicks = {}
        self.shows = {}
        self.connected_consumers = {}
        self.provider_docs_clicked = {}  # Initialize dictionary for clicked documents by providers
        self.consumer_docs_clicked = {}  # Initialize dictionary for clicked documents by consumers
        self.consumer_category_preferences = defaultdict(dict)  # Store category preferences for each consumer
        self.documents = []
        self.documents_click_counts = {}  # Store documents click counts
        self.sorted_documents = []
        self.weighted = False # Weighted distribution of items
        self.weighted_category = {}
        self.documents_weights = []

    @abstractmethod
    def recommend_documents(self, consumers, slate_size=1):
        """
        Abstract method to recommend documents to consumers.

        Args:
            consumers (list): List of Consumer instances to recommend documents to.
            slate_size (int): Number of documents to recommend per consumer (default is 1).

        Returns:
            dict: Dictionary mapping consumer IDs to lists of recommended documents.
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
    
    
    def update_documents_list(self, force_update=False):
        if not self.sorted_documents or force_update:
            if self.specialized_categories: # Check if there are specialized categories
                self.documents = [
                    document
                    for provider in self.connected_providers.values()
                    for document in provider.documents
                    if document.categories.intersection(self.specialized_categories) and  # Check if document has any specialized categories
                    not self.prohibited_categories.intersection(document.categories)  # Check if document has any prohibited categories
                ]
            else:
                self.documents = [
                    document
                    for provider in self.connected_providers.values()
                    for document in provider.documents
                    if not self.prohibited_categories.intersection(document.categories)  # Check if document has any prohibited categories
                ]

            # Update weights if weighted category available
            if self.weighted_category:
                self.set_documents_weights()

            # Sort the documents by click counts in descending order
            self.sorted_documents = sorted(
                self.documents,
                key=lambda doc: self.documents_click_counts.get(doc, 0),
                reverse=True
            )
            
            
    def set_documents_weights(self):
        self.documents_weights = []
        weighted_key, weighted_value = list(self.weighted_category.items())[0]
        for document in self.documents:
            if weighted_key in document.categories:
                document.weight = weighted_value
            self.documents_weights.append(document.weight)
        
        
            
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
            self.provider_docs_clicked[provider.provider_id] = []  # Initialize clicked documents list for the provider
        self.sorted_documents = []

        
    def disconnect_provider(self, provider_id):
        """
        Remove a provider from the recommender.

        Args:
            provider_id (int): ID of the provider to remove.
        """
        print('Recommender', self.recommender_id, 'diconnected provider', provider_id)
        if provider_id in self.connected_providers:
            for doc_id in self.provider_docs_clicked[provider_id]:
                del self.documents_click_counts[doc_id]
            del self.connected_providers[provider_id]
            del self.clicks[provider_id]
            del self.provider_docs_clicked[provider_id]
        self.sorted_documents = []

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
            self.consumer_docs_clicked[consumer.consumer_id] = []  # Initialize clicked documents list for the consumer

    def disconnect_consumer(self, consumer):
        """
        Disconnect a consumer from the recommender system.

        Args:
            consumer_id (int): ID of the consumer to disconnect.
        """
        if consumer in self.connected_consumers.values():
            del self.connected_consumers[consumer.consumer_id]
            del self.consumer_docs_clicked[consumer.consumer_id]
            consumer.unsubscribe_from_recommender_system(self.recommender_id)
            

    def record_click(self, consumer_id, clicked_document):
        clicked_document_id = clicked_document.doc_id
        clicked_provider_id = clicked_document.provider_id

        if clicked_provider_id in self.connected_providers:
            self.clicks[clicked_provider_id] += 1
            if clicked_document not in self.provider_docs_clicked[clicked_provider_id]:
                self.provider_docs_clicked[clicked_provider_id].append(clicked_document)

            if consumer_id not in self.consumer_docs_clicked:
                self.consumer_docs_clicked[consumer_id] = []
            self.consumer_docs_clicked[consumer_id].append(clicked_document)

            if clicked_document not in self.documents_click_counts:
                self.documents_click_counts[clicked_document] = 0
            self.documents_click_counts[clicked_document] += 1
            
            self.update_user_category_preferences(consumer_id, clicked_document)
            self.sorted_documents = []

        else:
            print(f"Record Click - Provider with ID {clicked_provider_id} is not connected to the recommender.")

    def record_show(self, provider_id):
        """
        Record that a document from a specific provider has been recommended.

        Args:
            provider_id (int): ID of the provider whose document is recommended.
        """
        if provider_id in self.connected_providers:
            self.shows[provider_id] += 1
        else:
            print(f"Record Show - Provider with ID {provider_id} is not connected to the recommender.")
    
    def update_user_category_preferences(self, consumer_id, clicked_document):
        """
        Update the category preferences for a user based on the clicked document.

        Args:
            consumer_id (int): ID of the user.
            clicked_document (Document): The document that the user clicked on.
        """
        # Get the categories of the clicked document
        clicked_document_categories = clicked_document.categories

        # Update the user's category preferences
        if consumer_id not in self.consumer_category_preferences:
            self.consumer_category_preferences[consumer_id] = defaultdict(int)
        
        for category in clicked_document_categories:
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
            dict: Dictionary of categories and associated probabilities for the user.
        """
        return self.consumer_category_preferences.get(consumer_id, {})
    
    
    def print_providers_clicked_documents(self):
        """
        Print information about clicked documents for each provider.
        """
        print("Clicked Documents by Providers:")
        for provider_id, clicked_documents in self.provider_docs_clicked.items():
            clicked_document_titles = [document.title for document in clicked_documents]  # Extract document titles
            clicked_document_titles_str = ', '.join(clicked_document_titles)
            print(f"Provider ID: {provider_id} - Clicked Documents: {clicked_document_titles_str}")

            
    def print_consumers_clicked_documents(self):
        """
        Print information about clicked documents for each consumer.
        """
        print("Clicked Documents by Consumers:")
        for consumer_id, clicked_documents in self.consumer_docs_clicked.items():
            clicked_document_titles = [document.title for document in clicked_documents]  # Extract document titles
            clicked_document_titles_str = ', '.join(clicked_document_titles)
            print(f"Consumer {consumer_id}: Clicked Documents: {clicked_document_titles_str}")

            
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
            print(f"Consumer {consumer_id}:"
                  f"Sensitivity={consumer.sensitivity}, Net Quality Exposure={consumer.net_quality_exposure:.2f}")

            
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
            recommendations (dict): Dictionary mapping consumer IDs to lists of recommended documents.
        """
        for consumer_id, recommended_documents in recommendations.items():
            consumer = self.connected_consumers[consumer_id]
            print(f"Consumer {consumer_id} - Recommended Documents:")
            for document in recommended_documents:
                print(f"- {document}")

