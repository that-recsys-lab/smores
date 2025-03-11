import pandas as pd
import numpy as np
import random

def ucb_switching(
    consumers, 
    providers, 
    recommenders, 
    num_days=5, 
    slate_size=3, 
    num_cycles=1, 
    model=None,
    forget_interactions=True,   
    transfer_interactions=False  
):
    """
    Run a multi-recommender experiment with UCB switching that also supports four profile handling cases.

    Args:
        consumers (list): List of Consumer instances.
        providers (list): List of Provider instances.
        recommenders (dict): Dictionary of Recommender instances keyed by recommender IDs.
        num_days (int, optional): Number of days to run the experiment (default is 5).
        slate_size (int, optional): Number of items to recommend (default is 3).
        num_cycles (int, optional): Number of cycles to run (default is 1).
        model: A prediction model.
        forget_interactions (bool): Whether to delete user interactions when switching.
        transfer_interactions (bool): Whether to transfer past interactions to the new recommender.
        
    Returns:
        pd.DataFrame: DataFrame containing provider data.
        pd.DataFrame: DataFrame containing consumer data.
        pd.DataFrame: DataFrame containing recommender system data.
        pd.DataFrame: DataFrame containing consumers' recommender choice history.
    """
    provider_data = []
    consumer_data = []
    recommender_data = []
    consumers_recommender_choice = []

    # Assume we have two recommenders: mainstream and niche.
    recommender_values = list(recommenders.values())
    mainstream_recommender = recommender_values[0]
    niche_recommender = recommender_values[1]

    # Initially subscribe each consumer to both recommenders.
    for consumer in consumers:
        mainstream_recommender.connect_consumer(consumer)
        consumer.subscribe_to_recommender_system(mainstream_recommender.recommender_id)
        niche_recommender.connect_consumer(consumer)
        consumer.subscribe_to_recommender_system(niche_recommender.recommender_id)
    
    print("mainstream consumers count:", len(mainstream_recommender.connected_consumers.keys()))
    print("niche consumers count:", len(niche_recommender.connected_consumers.keys()))

    # Subscribe providers to both recommenders.
    for provider in providers:
        mainstream_recommender.connect_provider(provider)
        provider.subscribe_to_recommender_system(mainstream_recommender.recommender_id)
        niche_recommender.connect_provider(provider)
        provider.subscribe_to_recommender_system(niche_recommender.recommender_id)
    
    # Run the experiment for a number of cycles.
    for cycle in range(1, num_cycles + 1):
        # Organize consumers by chosen recommender.
        consumers_by_recommender = {rec_id: [] for rec_id in recommenders.keys()}
        for consumer in consumers:
            # For the first cycle, you might preselect a recommender (e.g., mainstream).
            if cycle == 1:
                chosen_recommender_id = consumer.choose_recommender(
                    preselected_recommender_id=mainstream_recommender.recommender_id
                )
            else:
                chosen_recommender_id = consumer.choose_recommender()
            if chosen_recommender_id is None:
                continue
            consumers_recommender_choice.append([consumer.consumer_id, chosen_recommender_id])
            consumers_by_recommender[chosen_recommender_id].append(consumer)
        
        for rec_id, consumers_list in consumers_by_recommender.items():
            print(f"Number of consumers choosing {rec_id}: {len(consumers_list)}")
        
        for day in range(1, num_days + 1):
            print("===============> Day:", day, "<===============")
            # Generate recommendations for each recommender's consumers.
            recommendations = {}
            for recommender_id, recommender_consumers in consumers_by_recommender.items():
                recommender = recommenders[recommender_id]
                slate_items = recommender.recommend_items(recommender_consumers, slate_size=slate_size)
                for consumer in recommender_consumers:
                    recommendations[consumer.consumer_id] = (recommender_id, slate_items[consumer.consumer_id])
            
            # Simulate user responses and record interactions.
            clicked_items = {consumer.consumer_id: [] for consumer in consumers}
            for consumer in consumers:
                if len(consumer.available_recommenders) == 0:
                    print('Consumer is not connected to any recommenders')
                    continue

                consumer_id = consumer.consumer_id
                recommender_id, recommended_items = recommendations.get(consumer_id, (None, []))
                if recommender_id is None:
                    continue
                responses = consumer.simulate_response(
                    recommended_items, recommender_system_id=recommender_id
                )
                for i, response in enumerate(responses):
                    if response.get("click", 0) == 1:
                        clicked_items[consumer_id].append(recommended_items[i])
                        rec = recommenders[recommender_id]
                        rec.record_click(consumer_id, recommended_items[i])
                        rec.add_interaction(
                            consumer_id,
                            recommended_items[i].item_id,
                            model.predict(consumer_id, recommended_items[i].item_id).est
                        )
        
        # --- Switching Block ---
        # For each consumer, check satisfaction for the recommender they last used.
        for consumer in consumers:
            # (Assuming get_satisfaction_score takes a recommender ID.)
            # Here we assume a low satisfaction threshold of 0.1.
            for rec_id, status in consumer.connected_recommenders.items():
                if status == 1 and consumer.get_satisfaction_score(rec_id) < 0.1:
                    # Determine old and new recommender based on current rec_id.
                    if rec_id == mainstream_recommender.recommender_id:
                        old_rec = mainstream_recommender
                        new_rec = niche_recommender
                    else:
                        old_rec = niche_recommender
                        new_rec = mainstream_recommender
                    
                    # Retrieve interactions if transferring.
                    old_interactions = (
                        old_rec.get_user_interactions(consumer.consumer_id)
                        if transfer_interactions else []
                    )
                    
                    # Disconnect consumer from the old recommender.
                    old_rec.disconnect_consumer(consumer)
                    consumer.unsubscribe_from_recommender_system(old_rec.recommender_id)
                    
                    # Optionally forget interactions.
                    if forget_interactions:
                        old_rec.remove_user_interactions(consumer.consumer_id)
                        consumer.remove_user(retain_profile=False)
                    else:
                        consumer.remove_user(retain_profile=True)
                    
                    # Transfer interactions if needed.
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
                    
                    # For simplicity, we break after one switch per consumer.
                    break
        
        # End of cycle: charge fees, record provider/consumer/recommender data.
        for recommender in recommenders.values():
            recommender.charge_subscription_fees()
            
        for provider in providers:
            for rec_id, rec in recommenders.items():
                if rec_id in provider.connected_recommenders.keys():
                    provider_data.append([
                        provider.provider_id,
                        rec_id,
                        provider.profit.get(rec_id, [0])[cycle-1] if cycle <= len(provider.profit.get(rec_id, [0])) else 0,
                        cycle,
                        provider.pay_cycle_fee.get(rec_id, [0])[cycle-1] if cycle <= len(provider.pay_cycle_fee.get(rec_id, [0])) else 0,
                        provider.pay_cycle_clicks.get(rec_id, [0])[cycle-1] if cycle <= len(provider.pay_cycle_clicks.get(rec_id, [0])) else 0,
                        provider.pay_cycle_shows.get(rec_id, [0])[cycle-1] if cycle <= len(provider.pay_cycle_shows.get(rec_id, [0])) else 0,
                        len([r for r in provider.connected_recommenders.keys() if provider.connected_recommenders[r] == 1])
                    ])
        
        for consumer in consumers:
            for rec_id, rec in recommenders.items():
                if rec_id in consumer.connected_recommenders.keys():
                    consumer_id = consumer.consumer_id
                    satisfaction = consumer.get_satisfaction_score(rec_id)
                    rec_state = 'connected' if consumer.connected_recommenders[rec_id] == 1 else ''
                    kl_div = None
                    consumer_data.append([consumer_id, rec_id, cycle, satisfaction, rec_state, kl_div])
        
        for provider in providers:
            unsubscribed_list = provider.update_recommender_subscription()
            for rec_id, rec in recommenders.items():
                if rec_id in unsubscribed_list:
                    rec.disconnect_provider(provider.provider_id)
                    
        for rec_id, rec in recommenders.items():
            num_providers = len(rec.connected_providers)
            num_consumers = len(rec.connected_consumers)
            total_profit = rec.profit[-1]
            recommender_data.append([rec_id, num_providers, num_consumers, total_profit, cycle])
        
        print("\n====================================================")
        print("===============> Finished Cycle:", cycle, "<================")
        print("====================================================\n")
    
    # Convert logs to DataFrames.
    provider_df = pd.DataFrame(provider_data, columns=[
        "provider_id", "recommender_id", "profit", "cycle", "fee", "clicks", "shows", "connected_recommenders"
    ])
    consumer_df = pd.DataFrame(consumer_data, columns=[
        "consumer_id", "recommender_id", "cycle", "satisfaction_score", "rec_state", "kl_divergence"
    ])
    recommender_df = pd.DataFrame(recommender_data, columns=[
        "recommender_id", "num_connected_providers", "num_connected_consumers", "total_profit", "cycle"
    ])
    consumer_recommender_df = pd.DataFrame(consumers_recommender_choice, columns=["consumer_id", "recommender_id"])

    return provider_df, consumer_df, recommender_df, consumer_recommender_df
