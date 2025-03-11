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
        forget_interactions (bool): Whether to delete user interactions when leaving a recommender.
        transfer_interactions (bool): Whether to transfer past interactions to the new recommender.

    Returns:
        pd.DataFrame: DataFrame containing provider data.
        pd.DataFrame: DataFrame containing consumer data.
        pd.DataFrame: DataFrame containing recommender system data.
        pd.DataFrame: DataFrame containing consumers' recommender choice history.
    """
    print("Running threshold-switching recsys experiment")

    provider_data = []
    consumer_data = []
    recommender_data = []
    consumers_recommender_choice = []

    # Get the two recommenders we are working with
    recommender_values = list(recommenders.values())
    mainstream_recommender = recommender_values[0]
    niche_recommender = recommender_values[1]

    # Subscribe consumers to the mainstream recommender by default,
    # and subscribe to niche with a zero flag.
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

    # Run the simulation for the specified number of cycles and days
    for cycle in range(1, num_cycles + 1):
        # Reset switch flag for the cycle
        switch_occurred = False
        
        for day in range(1, num_days + 1):
            print("===============> Day:", day, "<===============")
            # Organize consumers by the recommender they choose via UCB
            consumers_by_recommender = {rec_id: [] for rec_id in recommenders.keys()}
            for consumer in consumers:
                # For the first 5 cycles, force the mainstream recommender
                if cycle <= 5:
                    chosen_recommender_id = consumer.choose_recommender(
                        preselected_recommender_id=mainstream_recommender.recommender_id
                    )
                else:
                    chosen_recommender_id = consumer.choose_recommender()
                if chosen_recommender_id is None:
                    continue
                consumers_recommender_choice.append([consumer.consumer_id, chosen_recommender_id])
                consumers_by_recommender[chosen_recommender_id].append(consumer)

            # Generate recommendations for each recommender's consumers
            recommendations = {}
            for recommender_id, recommender_consumers in consumers_by_recommender.items():
                recommender = recommenders[recommender_id]
                slate_items = recommender.recommend_items(recommender_consumers, slate_size=slate_size)
                for consumer in recommender_consumers:
                    recommendations[consumer.consumer_id] = (recommender_id, slate_items[consumer.consumer_id])

            # Simulate user responses and record clicks/interactions
            clicked_items = {consumer.consumer_id: [] for consumer in consumers}
            for consumer in consumers:
                if len(consumer.available_recommenders) == 0:
                    print("Consumer is not connected to any recommenders")
                    continue

                consumer_id = consumer.consumer_id
                recommender_id, recommended_items = recommendations[consumer_id]
                slate_items = recommended_items
                # Simulate the consumer's response on the given slate
                responses = consumer.simulate_response(slate_items, recommender_system_id=recommender_id)

                # Record clicks and add interactions based on responses
                for i, response in enumerate(responses):
                    if response.get("click", 0) == 1:
                        clicked_items[consumer_id].append(slate_items[i])
                        recommenders[recommender_id].record_click(consumer_id, slate_items[i])
                        # Use the provided model to predict the rating
                        recommenders[recommender_id].add_interaction(
                            consumer_id,
                            slate_items[i].item_id,
                            model.predict(consumer_id, slate_items[i].item_id).est,
                        )
        
        # After all days in the cycle
        if cycle > 5:
            # Switch low-satisfaction consumers.
            for consumer in consumers:
                for rec_id, status in list(consumer.connected_recommenders.items()):
                    if status == 1 and consumer.get_satisfaction_score(rec_id) < 0.1:
                        # Determine the old and new recommender based on the current rec_id.
                        if rec_id == mainstream_recommender.recommender_id:
                            old_rec = mainstream_recommender
                            new_rec = niche_recommender
                        else:
                            old_rec = niche_recommender
                            new_rec = mainstream_recommender
        
                        # Retrieve interactions from the old recommender if transferring.
                        old_interactions = old_rec.get_user_interactions(consumer.consumer_id) if transfer_interactions else []
        
                        # Disconnect consumer from the old recommender.
                        old_rec.disconnect_consumer(consumer)
                        consumer.unsubscribe_from_recommender_system(old_rec.recommender_id)
        
                        # If forget_interactions is True, delete interactions from the old recommender.
                        if forget_interactions:
                            old_rec.remove_user_interactions(consumer.consumer_id)
        
                        # Clear consumer's internal state if forget_interactions is True.
                        consumer.remove_user(retain_profile=not forget_interactions)
                        
                        # If transfer_interactions is True, transfer the old interactions to the new recommender.
                        if transfer_interactions:
                            for interaction in old_interactions:
                                new_rec.add_interaction(
                                    interaction.user_id,
                                    interaction.item_id,
                                    interaction.rating
                                )
                        
                        # Connect consumer to the new recommender.
                        new_rec.connect_consumer(consumer)
                        consumer.subscribe_to_recommender_system(new_rec.recommender_id)
                        
                        switch_occurred = True
                        break
        else:
            print(f"Cycle {cycle}: Skipping switching due to startup phase (mainstream enforced).")

        # ---- Summary Logs Per Cycle ----
        mainstream_to_niche = 0
        niche_to_mainstream = 0
        switching_satisfaction_scores = []
        
        for consumer in consumers:
            for rec_id, status in consumer.connected_recommenders.items():
                if status == 1 and consumer.get_satisfaction_score(rec_id) < 0.1:
                    if rec_id == mainstream_recommender.recommender_id:
                        mainstream_to_niche += 1
                    else:
                        niche_to_mainstream += 1
                    switching_satisfaction_scores.append(consumer.get_satisfaction_score(rec_id))
        
        if switching_satisfaction_scores:
            avg_switching_satisfaction = sum(switching_satisfaction_scores) / len(switching_satisfaction_scores)
        else:
            avg_switching_satisfaction = "N/A"
        
        print(f"Cycle {cycle}: {mainstream_to_niche} switched to niche, {niche_to_mainstream} switched to mainstream.")
        print(f"Average satisfaction score before switching: {avg_switching_satisfaction}")

        # Provider Utility Distribution
        mainstream_profit = sum(mainstream_recommender.profit)
        niche_profit = sum(niche_recommender.profit)
        
        mainstream_providers = len(mainstream_recommender.connected_providers)
        niche_providers = len(niche_recommender.connected_providers)
        
        print(f"Cycle {cycle}: Mainstream Profit = {mainstream_profit}, Niche Profit = {niche_profit}")
        print(f"Cycle {cycle}: Active Providers - Mainstream: {mainstream_providers}, Niche: {niche_providers}")

        # User Distribution Summary
        mainstream_users = len(mainstream_recommender.connected_consumers)
        niche_users = len(niche_recommender.connected_consumers)
        
        total_users = mainstream_users + niche_users
        switching_users = mainstream_to_niche + niche_to_mainstream
        
        print(f"Cycle {cycle}: {mainstream_users} users in mainstream, {niche_users} in niche.")
        if total_users > 0:
            print(f"Cycle {cycle}: {switching_users}/{total_users} users switched recommenders ({(switching_users/total_users)*100:.2f}%).")

        # Verification: subscribe sample consumers to the new recommender and verify their interactions.
        if switch_occurred:
            representative_new_rec = niche_recommender if mainstream_to_niche > 0 else mainstream_recommender

            sample_consumers = consumers[:10]  
            remaining = len(consumers) - len(sample_consumers) 
            
            print("\n=== New Recommender Interactions Summary ===")
            for consumer in sample_consumers:
                representative_new_rec.connect_consumer(consumer)
                consumer.subscribe_to_recommender_system(representative_new_rec.recommender_id)
                if not forget_interactions:
                    old_ints = old_rec.get_user_interactions(consumer.consumer_id)
                    for interaction in old_ints:
                        representative_new_rec.add_interaction(interaction.user_id, interaction.item_id, interaction.rating)
                new_ints = representative_new_rec.get_user_interactions(consumer.consumer_id)
                if new_ints:
                    snippet = new_ints[:2]
                    print(f"User {consumer.consumer_id} now has {len(new_ints)} interactions. First few: {snippet}")
                else:
                    print(f"User {consumer.consumer_id} has 0 interactions in new recommender.")
            
            if remaining > 0:
                total_new = sum(len(representative_new_rec.get_user_interactions(c.consumer_id)) for c in consumers[10:])
                print(f"... and interactions for the remaining {remaining} consumers were processed (total new interactions: {total_new}).")
            
            # Force training on the representative new recommender to verify that transferred interactions are recognized.
            print("\nForcing training on new recommender to verify recognized interactions:")
            representative_new_rec.train_model_if_ready()
        else:
            print("No switching occurred this cycle; skipping verification block that requires a new recommender.")

        # Continue with fee charging and logging.
        removed_count = 0
        for consumer in consumers:
            if all(status == 0 for status in consumer.connected_recommenders.values()):
                removed_count += 1
        print(f"Cycle {cycle}: Removed {removed_count} consumers due to low satisfaction.")

        print("Connected to mainstream recommender:", len(mainstream_recommender.connected_consumers))
        print("Connected to Niche recommender:", len(niche_recommender.connected_consumers))

        for recommender in recommenders.values():
            recommender.charge_subscription_fees()

        # Collect provider data for logging
        for provider in providers:
            for recommender_id, recommender in recommenders.items():
                if recommender_id in provider.connected_recommenders.keys():
                    provider_data.append([
                        provider.provider_id,
                        recommender_id,
                        (provider.profit.get(recommender_id, [0])[cycle - 1]
                         if cycle <= len(provider.profit.get(recommender_id, [0]))
                         else 0),
                        cycle,
                        (provider.pay_cycle_fee.get(recommender_id, [0])[cycle - 1]
                         if cycle <= len(provider.pay_cycle_fee.get(recommender_id, [0]))
                         else 0),
                        (provider.pay_cycle_clicks.get(recommender_id, [0])[cycle - 1]
                         if cycle <= len(provider.pay_cycle_clicks.get(recommender_id, [0]))
                         else 0),
                        (provider.pay_cycle_shows.get(recommender_id, [0])[cycle - 1]
                         if cycle <= len(provider.pay_cycle_shows.get(recommender_id, [0]))
                         else 0),
                        len([rec for rec in provider.connected_recommenders.keys() if provider.connected_recommenders[rec] == 1]),
                    ])

        # Collect consumer data for logging
        for consumer in consumers:
            for recommender_id, recommender in recommenders.items():
                if recommender_id in consumer.connected_recommenders.keys():
                    consumer_id = consumer.consumer_id
                    consumer_satisfaction_score = consumer.get_satisfaction_score(recommender_system_id=recommender_id)
                    rec_state = "connected" if consumer.connected_recommenders[recommender_id] == 1 else ""
                    consumer_data.append([
                        consumer_id,
                        recommender_id,
                        cycle,
                        consumer_satisfaction_score,
                        rec_state
                    ])

        # Update provider subscriptions based on performance
        for provider in providers:
            unsubscribed_list = provider.update_recommender_subscription()
            for recommender_id, recommender in recommenders.items():
                if recommender_id in unsubscribed_list:
                    recommender.disconnect_provider(provider.provider_id)

        # Collect recommender data for logging
        for recommender_id, recommender in recommenders.items():
            num_connected_providers = len(recommender.connected_providers)
            num_connected_consumers = len(recommender.connected_consumers)
            total_profit = recommender.profit[-1]
            recommender_data.append([
                recommender_id,
                num_connected_providers,
                num_connected_consumers,
                total_profit,
                cycle,
            ])

        print("\n====================================================")
        print("===============> Finished Cycle:", cycle, "<================")
        print("====================================================\n")

    # Convert logs to DataFrames
    provider_df = pd.DataFrame(provider_data, columns=[
        "provider_id", "recommender_id", "profit", "cycle", "fee", "clicks", "shows", "connected_recommenders"
    ])
    consumer_df = pd.DataFrame(consumer_data, columns=[
        "consumer_id", "recommender_id", "cycle", "satisfaction_score", "rec_state"
    ])
    recommender_df = pd.DataFrame(recommender_data, columns=[
        "recommender_id", "num_connected_providers", "num_connected_consumers", "total_profit", "cycle"
    ])
    consumer_recommender_df = pd.DataFrame(consumers_recommender_choice, columns=["consumer_id", "recommender_id"])

    return provider_df, consumer_df, recommender_df, consumer_recommender_df
