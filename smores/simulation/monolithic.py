import pandas as pd
import numpy as np
import random


def monolithic(
    consumers,
    providers,
    recommenders,
    num_days=5,
    slate_size=3,
    num_cycles=1,
    model=None,
    forget_interactions=True,
    transfer_interactions=True
):
    """
    Run a multi-recommender experiment.

    Args:
        consumers (list): List of Consumer instances.
        providers (list): List of Provider instances.
        recommenders (dict): Dictionary of Recommender instances where keys are recommender IDs.
        num_days (int, optional): Number of days to run the experiment (default is 5).
        slate_size (int, optional): Number of items to recommend to each consumer (default is 3).

    Returns:
        pd.DataFrame: DataFrame containing provider data.
        pd.DataFrame: DataFrame containing consumer data.
        pd.DataFrame: DataFrame containing recommender system data.
    """

    print("Running monolithic recsys experiment")

    provider_data = []
    consumer_data = []
    recommender_data = []
    consumers_recommender_choice = []

    # Run the experiment for the specified number of days
    for cycle in range(1, num_cycles + 1):
        # Restore interactions from the previous cycle if transfer_interactions is True
        if transfer_interactions and 'previous_interactions' in locals():
            for recommender_id, recommender in recommenders.items():
                if recommender_id in previous_interactions:
                    for interaction in previous_interactions[recommender_id]:
                        consumer_id, item_id, rating = interaction
                        recommender.add_interaction(consumer_id, item_id, rating)
    
        for day in range(1, num_days + 1):
            print("===============> Day:", day, "<===============")
            # Organize consumers into lists based on the recommender they chose
            consumers_by_recommender = {
                recommender_id: [] for recommender_id in recommenders.keys()
            }
            for consumer in consumers:
                chosen_recommender_id = list(recommenders.keys())[0]  # Use the first recommender
                consumers_recommender_choice.append([consumer.consumer_id, chosen_recommender_id])
                consumers_by_recommender[chosen_recommender_id].append(consumer)

            # Make recommendations for each recommender's associated consumers
            recommendations = {}
            for (
                recommender_id,
                recommender_consumers,
            ) in consumers_by_recommender.items():
                recommender = recommenders[recommender_id]
                slate_items = recommender.recommend_items(
                    recommender_consumers, slate_size=slate_size
                )
                for consumer in recommender_consumers:
                    recommendations[consumer.consumer_id] = (
                        recommender_id,
                        slate_items[consumer.consumer_id],
                    )

            # Simulate user response, update satisfaction scores, and record clicks
            clicked_items = {
                consumer.consumer_id: [] for consumer in consumers
            }  # Initialize with empty lists
            for consumer in consumers:
                # continue if the consumer is not connected to any recommenders
                if len(consumer.available_recommenders) == 0:
                    print("Consumer is not connected to any recommenders")
                    continue

                consumer_id = consumer.consumer_id
                recommender_id, recommended_items = recommendations[consumer_id]
                slate_items = recommended_items
                chosen_recommender_id = (
                    recommender_id  # Choose recommender for the current user
                )
                responses = consumer.simulate_response(
                    slate_items, recommender_system_id=chosen_recommender_id
                )

                # Collect clicked items
                for i, response in enumerate(responses):
                    # record interaction (consumer_id, item_id, rating)
                    if response.get("click", 0) == 1:
                        clicked_items[consumer_id].append(slate_items[i])
                        recommenders[chosen_recommender_id].record_click(
                            consumer_id, slate_items[i]
                        )
                        recommenders[chosen_recommender_id].add_interaction(
                            consumer_id,
                            slate_items[i].item_id,
                            model.predict(consumer_id, slate_items[i].item_id).est,
                        )

        # Charge subscription fees to providers
        previous_interactions = {}
        if transfer_interactions:
            for recommender_id, recommender in recommenders.items():
                previous_interactions[recommender_id] = recommender.interactions
        if forget_interactions:
            for recommender_id, recommender in recommenders.items():
                recommender.interactions = []
                for consumer in consumers:
                    consumer.remove_user(retain_profile=False)
        for recommender in recommenders.values():
            recommender.charge_subscription_fees()

        # Get profit for each provider
        for provider in providers:
            for recommender_id, recommender in recommenders.items():
                if recommender_id in provider.connected_recommenders.keys():
                    provider_data.append(
                        [
                            provider.provider_id,
                            recommender_id,
                            (
                                provider.profit.get(recommender_id, [0])[cycle - 1]
                                if cycle
                                <= len(provider.profit.get(recommender_id, [0]))
                                else 0
                            ),
                            cycle,
                            (
                                provider.pay_cycle_fee.get(recommender_id, [0])[
                                    cycle - 1
                                ]
                                if cycle
                                <= len(provider.pay_cycle_fee.get(recommender_id, [0]))
                                else 0
                            ),
                            (
                                provider.pay_cycle_clicks.get(recommender_id, [0])[
                                    cycle - 1
                                ]
                                if cycle
                                <= len(
                                    provider.pay_cycle_clicks.get(recommender_id, [0])
                                )
                                else 0
                            ),
                            (
                                provider.pay_cycle_shows.get(recommender_id, [0])[
                                    cycle - 1
                                ]
                                if cycle
                                <= len(
                                    provider.pay_cycle_shows.get(recommender_id, [0])
                                )
                                else 0
                            ),
                            len(
                                [
                                    rec
                                    for rec in provider.connected_recommenders.keys()
                                    if provider.connected_recommenders[rec] == 1
                                ]
                            ),
                        ]
                    )

        for consumer in consumers:
            for recommender_id, recommender in recommenders.items():
                if recommender_id in consumer.connected_recommenders.keys():
                    consumer_id = consumer.consumer_id
                    consumer_satisfaction_score = consumer.get_satisfaction_score(
                        recommender_system_id=recommender_id
                    )
                    rec_state = (
                        "connected"
                        if consumer.connected_recommenders[recommender_id] == 1
                        else ""
                    )
                    kl_divergence = consumer.compute_kl_divergence(recommender_id)
                    consumer_data.append(
                        [
                            consumer_id,
                            recommender_id,
                            cycle,
                            consumer_satisfaction_score,
                            rec_state,
                            kl_divergence,
                        ]
                    )

        # Update the subscription for each provider
        for provider in providers:
            unsubscribed_list = provider.update_recommender_subscription()
            for recommender_id, recommender in recommenders.items():
                if recommender_id in unsubscribed_list:
                    recommender.disconnect_provider(provider.provider_id)

        for recommender_id, recommender in recommenders.items():
            num_connected_providers = len(recommender.connected_providers)
            num_connected_consumers = len(recommender.connected_consumers)
            total_profit = recommender.profit[-1]
            recommender_data.append(
                [
                    recommender_id,
                    num_connected_providers,
                    num_connected_consumers,
                    total_profit,
                    cycle,
                ]
            )

        print("\n")
        print("Consumer KL Divergence")
        print("\n")
        for consumer in consumers[:5]:
            consumer.compute_kl_divergence(recommender_id)

        print("\n====================================================")
        print("===============> Finished Cycle:", cycle, "<================")
        print("==================================================== \n")

    # Convert lists to DataFrames
    provider_df = pd.DataFrame(
        provider_data,
        columns=[
            "provider_id",
            "recommender_id",
            "profit",
            "cycle",
            "fee",
            "clicks",
            "shows",
            "connected_recommenders",
        ],
    )
    consumer_df = pd.DataFrame(
        consumer_data,
        columns=[
            "consumer_id",
            "recommender_id",
            "cycle",
            "satisfaction_score",
            "rec_state",
            "kl_divergence",
        ],
    )
    recommender_df = pd.DataFrame(
        recommender_data,
        columns=[
            "recommender_id",
            "num_connected_providers",
            "num_connected_consumers",
            "total_profit",
            "cycle",
        ],
    )
    consumer_recommender_df = pd.DataFrame(
        consumers_recommender_choice, columns=["consumer_id", "recommender_id"]
    )

    return (
        provider_df,
        consumer_df,
        recommender_df,
        consumer_recommender_df,
    )