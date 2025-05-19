class Provider:
    def __init__(self):
        self.id = None
        self.items = None
        self.history = None
        self.utility_model = None


'''
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
'''