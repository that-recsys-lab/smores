import pandas as pd
import numpy as np
import random


def MonolithicEcosystem(
    consumers,
    niche_consumers_set,
    providers,
    niche_providers_set,
    recommenders,
    num_days=5,
    slate_size=3,
    num_cycles=1,
):
    """
    Run a multi-recommender experiment.

    Args:
        consumers (list): List of Consumer instances.
        niche_consumers_set (set): Set of niche consumer IDs.
        providers (list): List of Provider instances.
        niche_providers_set (set): Set of niche provider IDs.
        recommenders (dict): Dictionary of Recommender instances where keys are recommender IDs.
        num_days (int, optional): Number of days to run the experiment (default is 5).
        slate_size (int, optional): Number of documents to recommend to each consumer (default is 3).

    Returns:
        pd.DataFrame: DataFrame containing provider data.
        pd.DataFrame: DataFrame containing consumer data.
        pd.DataFrame: DataFrame containing recommender system data.
    """
    provider_data = []
    consumer_data = []
    recommender_data = []
    customers_recommender_choice = []

    recommender_values = list(recommenders.values())
    mainstream_recommender = recommender_values[0]

    # Subscribe consumers to the recommender
    for consumer in consumers:
        mainstream_recommender.connect_consumer(consumer)
        consumer.subscribe_to_recommender_system(mainstream_recommender.recommender_id)

    print("mainstream Recommender", mainstream_recommender.connected_consumers.keys())

    # Subscribe providers to both recommenders
    for provider in providers:
        mainstream_recommender.connect_provider(provider)
        provider.subscribe_to_recommender_system(mainstream_recommender.recommender_id)

    ### LOGGING ###
    print(
        "Connected to mainstream recommender:",
        len(mainstream_recommender.connected_consumers),
    )

    # Run the experiment for the specified number of days
    for cycle in range(1, num_cycles + 1):
        for day in range(1, num_days + 1):
            print("===============> Day:", day, "<===============")
            # Organize consumers into lists based on the recommender they chose
            consumers_by_recommender = {
                recommender_id: [] for recommender_id in recommenders.keys()
            }
            for (
                consumer
            ) in consumers:  # Iterate over the consumers for the current recommender ID
                chosen_recommender_id = consumer.choose_recommender()
                if chosen_recommender_id is None:
                    continue
                # Append recommender to consumer to evaluate UCB
                customers_recommender_choice.append(
                    [consumer.consumer_id, chosen_recommender_id]
                )  # used for analysis
                consumers_by_recommender[chosen_recommender_id].append(consumer)

            # Make recommendations for each recommender's associated consumers
            recommendations = {}
            for (
                recommender_id,
                recommender_consumers,
            ) in consumers_by_recommender.items():
                recommender = recommenders[recommender_id]
                slate_documents = recommender.recommend_documents(
                    recommender_consumers, slate_size=slate_size
                )
                for consumer in recommender_consumers:
                    recommendations[consumer.consumer_id] = (
                        recommender_id,
                        slate_documents[consumer.consumer_id],
                    )

            # Simulate user response, update satisfaction scores, and record clicks
            clicked_documents = {
                consumer.consumer_id: [] for consumer in consumers
            }  # Initialize with empty lists
            for consumer in consumers:
                # continue if the consumer is not connected to any recommenders
                if len(consumer.available_recommenders) == 0:
                    print("Consumer is not connected to any recommenders")
                    continue

                consumer_id = consumer.consumer_id
                recommender_id, recommended_documents = recommendations[consumer_id]
                slate_documents = recommended_documents
                chosen_recommender_id = (
                    recommender_id  # Choose recommender for the current user
                )
                responses = consumer.simulate_response(
                    slate_documents, recommender_system_id=chosen_recommender_id
                )

                # Collect clicked items
                for i, response in enumerate(responses):
                    if response.get("click", 0) == 1:
                        clicked_documents[consumer_id].append(slate_documents[i])
                        recommenders[chosen_recommender_id].record_click(
                            consumer_id, slate_documents[i]
                        )

        # Charge subscription fees to providers
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
                            (
                                "niche"
                                if provider.provider_id in niche_providers_set
                                else "mainstream"
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
                    controlled = (
                        "niche"
                        if consumer.consumer_id in niche_consumers_set
                        else "mainstream"
                    )
                    consumer_data.append(
                        [
                            consumer_id,
                            recommender_id,
                            cycle,
                            consumer_satisfaction_score,
                            rec_state,
                            controlled,
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
            "controlled",
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
            "controlled",
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
    customer_recommender_df = pd.DataFrame(
        customers_recommender_choice, columns=["customer_id", "recommender_id"]
    )

    return (
        provider_df,
        consumer_df,
        recommender_df,
        customer_recommender_df,
    )
