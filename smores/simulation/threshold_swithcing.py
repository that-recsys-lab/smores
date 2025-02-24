import pandas as pd
import numpy as np
import random
import logging

def threshold_switching(
    consumers,
    providers,
    recommenders,
    num_days=5,
    slate_size=3,
    num_cycles=1,
    model=None,
    forget_flag=True, # If True, interactions are deleted; if False, they are transferred
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
    print("Running threshold-switching recsys experiment")

    provider_data = []
    consumer_data = []
    recommender_data = []
    consumers_recommender_choice = []

    recommender_values = list(recommenders.values())
    mainstream_recommender = recommender_values[0]
    niche_recommender = recommender_values[1]

    # Subscribe consumers to mainstream recommender
    for consumer in consumers:
        mainstream_recommender.connect_consumer(consumer)
        consumer.subscribe_to_recommender_system(mainstream_recommender.recommender_id, 1)
        consumer.subscribe_to_recommender_system(niche_recommender.recommender_id, 0)

    print("mainstream consumers count:", len(mainstream_recommender.connected_consumers.keys()))
    print("niche consumers count:", len(niche_recommender.connected_consumers.keys()))

    # Subscribe providers to both recommenders
    for provider in providers:

        mainstream_recommender.connect_provider(provider)
        provider.subscribe_to_recommender_system(mainstream_recommender.recommender_id)

        niche_recommender.connect_provider(provider)
        provider.subscribe_to_recommender_system(niche_recommender.recommender_id)

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
                consumers_recommender_choice.append(
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

        # Switching at the end of each cycle
        for consumer in consumers:
            # Get the connected recommender
            for key, value in consumer.connected_recommenders.copy().items():
                if value == 0:  # if not connected; break
                    continue
                recommender_id = key
                if consumer.satisfaction_scores[recommender_id] < 0.1:
                    if (max(consumer.satisfaction_scores, key=consumer.satisfaction_scores.get) == recommender_id and 
                        all(score > 0 for score in consumer.satisfaction_scores.values())):
                        break
                    elif (consumer.consumer_id in mainstream_recommender.connected_consumers.keys()):
                        mainstream_recommender.disconnect_consumer(consumer)
                        consumer.unsubscribe_from_recommender_system(mainstream_recommender.recommender_id)
                        niche_recommender.connect_consumer(consumer)
                        consumer.subscribe_to_recommender_system(niche_recommender.recommender_id)
                    else:
                        niche_recommender.disconnect_consumer(consumer)
                        consumer.unsubscribe_from_recommender_system(niche_recommender.recommender_id)
                        mainstream_recommender.connect_consumer(consumer)
                        consumer.subscribe_to_recommender_system(mainstream_recommender.recommender_id)
        for consumer in consumers:
            for rec_id, status in consumer.connected_recommenders.copy().items():
                if status == 1 and consumer.get_satisfaction_score(rec_id) < 0.05:
                    # print(f"Removing consumer {consumer.consumer_id} due to extremely low satisfaction on recommender {rec_id}.")
                    consumer.remove_user(retain_profile=False)
                    for rec in recommenders.values():
                        rec.disconnect_consumer(consumer)
                        rec.remove_user_interactions(consumer.consumer_id)
                    break
        # --- Verification: Confirm unsubscribed consumers from old recommender ---
        low_sat_total = 0
        still_connected = 0
        for consumer in consumers:
            if consumer.get_satisfaction_score(mainstream_recommender.recommender_id) < 0.05:
                low_sat_total += 1
                if consumer.consumer_id in mainstream_recommender.connected_consumers:
                    still_connected += 1
        
        if low_sat_total == 0:
            print("No consumers had low satisfaction for the old recommender.")
        elif still_connected == 0:
            print(f"Success: All {low_sat_total} low-satisfaction consumers have been unsubscribed from the old recommender.")
        else:
            print(f"Error: {still_connected} out of {low_sat_total} low-satisfaction consumers are still connected to the old recommender.")



        # --- New Verification Code ---
        # Print interactions in the old recommender 
        old_rec = mainstream_recommender  
        print("\n=== Old Recommender Interactions Summary ===")
        sample_consumers = consumers[:10]  
        for consumer in sample_consumers:
            old_ints = old_rec.get_user_interactions(consumer.consumer_id)
            if old_ints:
                snippet = old_ints[:2]  
                print(f"User {consumer.consumer_id} has {len(old_ints)} interactions. First few: {snippet}")
            else:
                print(f"User {consumer.consumer_id} has 0 interactions.")
        
        # Print overall counts without details for remaining consumers
        total_consumers = len(consumers)
        remaining = total_consumers - len(sample_consumers)
        if remaining > 0:
            zero_count = sum(1 for consumer in consumers[10:] if not old_rec.get_user_interactions(consumer.consumer_id))
            nonzero_count = remaining - zero_count
            print(f"... and {remaining} more consumers ({nonzero_count} with interactions, {zero_count} with 0 interactions).")
        
        # Subscribe to the new recommender (e.g., niche)
        new_rec = niche_recommender
        print("\n=== New Recommender Interactions Summary ===")
        for consumer in sample_consumers:
            new_rec.connect_consumer(consumer)
            consumer.subscribe_to_recommender_system(new_rec.recommender_id)
            if not forget_flag:
                old_ints = old_rec.get_user_interactions(consumer.consumer_id)
                for interaction in old_ints:
                    new_rec.add_interaction(interaction.user_id, interaction.item_id, interaction.rating)
            new_ints = new_rec.get_user_interactions(consumer.consumer_id)
            if new_ints:
                snippet = new_ints[:2]
                print(f"User {consumer.consumer_id} now has {len(new_ints)} interactions. First few: {snippet}")
            else:
                print(f"User {consumer.consumer_id} has 0 interactions in new recommender.")
        
        #  Print overall counts for the rest
        if remaining > 0:
            total_new = sum(len(new_rec.get_user_interactions(c.consumer_id)) for c in consumers[10:])
            print(f"... and interactions for the remaining {remaining} consumers were processed (total new interactions: {total_new}).")
        
        # Step 3: Force training on new recommender to verify interactions are recognized
        print("\nForcing training on new recommender to verify recognized interactions:")
        new_rec.train_model_if_ready()



        # Continue with existing logging and fee charging...
        removed_count = 0
        for consumer in consumers:
            if all(status == 0 for status in consumer.connected_recommenders.values()):
                removed_count += 1
        print(f"Cycle {cycle}: Removed {removed_count} consumers due to low satisfaction.")

        print(
            "Connected to mainstream recommender:",
            len(mainstream_recommender.connected_consumers),
        )
        print(
            "Connected to Niche recommender:",
            len(niche_recommender.connected_consumers),
        )

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

        print("\n====================================================")
        print("===============> Finished Cycle:", cycle, "<================")
        print("====================================================\n")

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

    return provider_df, consumer_df, recommender_df, consumer_recommender_df