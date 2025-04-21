import pandas as pd
import numpy as np
import random

def ucb_switching(consumers, providers, recommenders, num_days=5, slate_size=3, num_cycles=1, model=None, forget_interactions=True, transfer_interactions=True):
    """
    Run a multi-recommender experiment with threshold-like switching logic.
    Args:
        consumers (list): List of Consumer instances.
        providers (list): List of Provider instances.
        recommenders (dict): Dictionary of Recommender instances where keys are recommender IDs.
        num_days (int, optional): Number of days per cycle (default is 5).
        slate_size (int, optional): Number of items to recommend (default is 3).
        num_cycles (int, optional): Number of cycles to run (default is 1).
        model: Predictive model for ratings (default is None).
        forget_interactions (bool, optional): Whether to forget interactions when switching (default is True).
        transfer_interactions (bool, optional): Whether to transfer interactions when switching (default is True).
    Returns:
        pd.DataFrame: Provider data.
        pd.DataFrame: Consumer data.
        pd.DataFrame: Recommender system data.
        pd.DataFrame: Consumer recommender choices.
    """
    provider_data = []
    consumer_data = []
    recommender_data = []
    consumers_recommender_choice = []
    
    recommender_values = list(recommenders.values())
    mainstream_recommender = recommender_values[0]
    niche_recommender = recommender_values[1]
    
    # Subscribe consumers to both recommenders
    for consumer in consumers:
        mainstream_recommender.connect_consumer(consumer)
        consumer.subscribe_to_recommender_system(mainstream_recommender.recommender_id, state=1)
        niche_recommender.connect_consumer(consumer)
        consumer.subscribe_to_recommender_system(niche_recommender.recommender_id, state=0)
        
        
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
        # Organize consumers by chosen recommender
        consumers_by_recommender = {recommender_id: [] for recommender_id in recommenders.keys()}
        for consumer in consumers:
            # For first 5 cycles, force mainstream recommender
            if cycle <= 5:
                chosen_recommender_id = consumer.choose_recommender(preselected_recommender_id=mainstream_recommender.recommender_id)
            else:
                chosen_recommender_id = consumer.choose_recommender()
            if chosen_recommender_id is None:
                continue
            consumers_recommender_choice.append([consumer.consumer_id, chosen_recommender_id])
            consumers_by_recommender[chosen_recommender_id].append(consumer)
            
        [print(key, len(value)) for key, value in consumers_by_recommender.items()]

        for recommender_id in recommenders.keys():
            print(f"Number of consumers choosing {recommender_id}:", len(consumers_by_recommender[recommender_id]))
            
        for day in range(1, num_days + 1):
            print("===============> Day:", day, "<===============") 
            # Make recommendations for each recommender's associated consumers
            recommendations = {}
            for recommender_id, recommender_consumers in consumers_by_recommender.items():
                recommender = recommenders[recommender_id]
                slate_items = recommender.recommend_items(recommender_consumers, slate_size=slate_size)
                for consumer in recommender_consumers:
                    recommendations[consumer.consumer_id] = (recommender_id, slate_items[consumer.consumer_id])

            # Simulate user response, update satisfaction scores, and record clicks
            clicked_items = {consumer.consumer_id: [] for consumer in consumers}  # Initialize with empty lists
            for consumer in consumers:
                # continue if the consumer is not connected to any recommenders
                if len(consumer.available_recommenders) == 0:
                    print('Consumer is not connected to any recommenders')
                    continue
                    
                consumer_id = consumer.consumer_id
                recommender_id, recommended_items = recommendations[consumer_id]
                slate_items = recommended_items
                chosen_recommender_id = recommender_id  # Choose recommender for the current user
                responses = consumer.simulate_response(slate_items, recommender_system_id=chosen_recommender_id)
                
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
                        
            
        # Charge subscription fees to providers
        for recommender in recommenders.values():
            recommender.charge_subscription_fees()

        # Switching logic after 5 cycles
        if cycle > 5:
            for consumer in consumers:
                for key, value in consumer.connected_recommenders.copy().items():
                    if value == 0:
                        continue
                    recommender_id = key
                    if consumer.satisfaction_scores.get(recommender_id, 0) < 0.1:
                        # Prevent switching if this is the best option and all scores are positive
                        if (max(consumer.satisfaction_scores, key=consumer.satisfaction_scores.get) == recommender_id and 
                            all(score > 0 for score in consumer.satisfaction_scores.values())):
                            break
                        elif consumer.consumer_id in mainstream_recommender.connected_consumers.keys():
                            old_rec = mainstream_recommender
                            new_rec = niche_recommender
                        else:
                            old_rec = niche_recommender
                            new_rec = mainstream_recommender
                        # Retrieve interactions if transferring
                        old_interactions = old_rec.get_user_interactions(consumer.consumer_id) if transfer_interactions else []
                        # Disconnect from old recommender
                        old_rec.disconnect_consumer(consumer)
                        consumer.unsubscribe_from_recommender_system(old_rec.recommender_id)
                        # Remove interactions if forgetting
                        if forget_interactions:
                            old_rec.remove_user_interactions(consumer.consumer_id)
                        # Retain profile if not forgetting interactions (i.e., keep data for continuity)
                        retain_profile = not forget_interactions
                        consumer.remove_user(retain_profile=retain_profile)
                        # Transfer interactions if enabled
                        if transfer_interactions:
                            for interaction in old_interactions:
                                new_rec.add_interaction(interaction.user_id, interaction.item_id, interaction.rating)
                        # Connect to new recommender
                        new_rec.connect_consumer(consumer)
                        consumer.subscribe_to_recommender_system(new_rec.recommender_id, state=1)
                        break  # Ensure only one switch per cycle
                    
        # Get profit for each provider
        for provider in providers:
            for recommender_id, recommender in recommenders.items():
                if recommender_id in provider.connected_recommenders.keys():
                    provider_data.append([
                        provider.provider_id,
                        recommender_id,
                        provider.profit.get(recommender_id, [0])[cycle-1] if cycle <= len(provider.profit.get(recommender_id, [0])) else 0,
                        cycle,
                        provider.pay_cycle_fee.get(recommender_id, [0])[cycle-1] if cycle <= len(provider.pay_cycle_fee.get(recommender_id, [0])) else 0,
                        provider.pay_cycle_clicks.get(recommender_id, [0])[cycle-1] if cycle <= len(provider.pay_cycle_clicks.get(recommender_id, [0])) else 0,
                        provider.pay_cycle_shows.get(recommender_id, [0])[cycle-1] if cycle <= len(provider.pay_cycle_shows.get(recommender_id, [0])) else 0,
                        len([rec for rec in provider.connected_recommenders.keys() if provider.connected_recommenders[rec] == 1]),
                    ])
                    
        for consumer in consumers:
            for recommender_id, recommender in recommenders.items():
                if recommender_id in consumer.connected_recommenders.keys():
                    consumer_id = consumer.consumer_id
                    consumer_satisfaction_score = consumer.get_satisfaction_score(recommender_system_id=recommender_id)
                    rec_state = 'connected' if consumer.connected_recommenders[recommender_id] == 1 else ''
                    consumer_data.append([consumer_id, recommender_id, cycle, consumer_satisfaction_score, rec_state])
        
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
            recommender_data.append([recommender_id, num_connected_providers, num_connected_consumers, total_profit, cycle])
        
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