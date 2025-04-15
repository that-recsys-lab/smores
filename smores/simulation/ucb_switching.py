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
    transfer_interactions=True
):
    """
    Run a multi-recommender UCB experiment with threshold-like switching logic.
    
    Args:
        consumers (list): List of Consumer instances.
        providers (list): List of Provider instances.
        recommenders (dict): Dictionary of Recommender instances keyed by recommender IDs.
        num_days (int): Number of days per cycle.
        slate_size (int): Number of items recommended per consumer.
        num_cycles (int): Number of cycles to run.
        model: Model used to predict ratings.
        forget_interactions (bool): If True, delete interactions when switching.
        transfer_interactions (bool): If True, transfer interactions during switching.
        
    Returns:
        pd.DataFrame: Provider data.
        pd.DataFrame: Consumer data.
        pd.DataFrame: Recommender data.
        pd.DataFrame: Consumers’ recommender choice history.
    """
    provider_data = []
    consumer_data = []
    recommender_data = []
    consumers_recommender_choice = []
    
    recommender_values = list(recommenders.values())
    mainstream_recommender = recommender_values[0]
    niche_recommender = recommender_values[1]
    
    # Subscribe consumers with initial weights as in the threshold experiment:
    for consumer in consumers:
        mainstream_recommender.connect_consumer(consumer)
        consumer.subscribe_to_recommender_system(mainstream_recommender.recommender_id, 1)
        
        niche_recommender.connect_consumer(consumer)
        consumer.subscribe_to_recommender_system(niche_recommender.recommender_id, 0)
    
    print("Mainstream consumers count:", len(mainstream_recommender.connected_consumers.keys()))
    print("Niche consumers count:", len(niche_recommender.connected_consumers.keys()))
    
    # Subscribe providers to both recommenders:
    for provider in providers:
        mainstream_recommender.connect_provider(provider)
        provider.subscribe_to_recommender_system(mainstream_recommender.recommender_id)
        
        niche_recommender.connect_provider(provider)
        provider.subscribe_to_recommender_system(niche_recommender.recommender_id)
    
    for cycle in range(1, num_cycles + 1):
        # Organize consumers by chosen recommender
        consumers_by_recommender = {rec_id: [] for rec_id in recommenders.keys()}
        for consumer in consumers:
            if cycle <= 5:
                chosen_recommender_id = consumer.choose_recommender(
                    preselected_recommender_id=mainstream_recommender.recommender_id
                )
            else:
                chosen_recommender_id = consumer.choose_recommender()
            if chosen_recommender_id is None:
                continue
            consumers_by_recommender[chosen_recommender_id].append(consumer)
            consumers_recommender_choice.append([consumer.consumer_id, chosen_recommender_id])
        
        for day in range(1, num_days + 1):
            print("===============> Day:", day, "<===============")
            # Generate recommendations for each group of consumers
            recommendations = {}
            for recommender_id, recommender_consumers in consumers_by_recommender.items():
                recommender = recommenders[recommender_id]
                slate_items = recommender.recommend_items(recommender_consumers, slate_size=slate_size)
                for consumer in recommender_consumers:
                    recommendations[consumer.consumer_id] = (recommender_id, slate_items[consumer.consumer_id])
            
            # Simulate responses and record clicks/interactions
            clicked_items = {consumer.consumer_id: [] for consumer in consumers}
            for consumer in consumers:
                if len(consumer.available_recommenders) == 0:
                    print("Consumer is not connected to any recommenders")
                    continue
                consumer_id = consumer.consumer_id
                recommender_id, recommended_items = recommendations[consumer_id]
                responses = consumer.simulate_response(recommended_items, recommender_system_id=recommender_id)
                for i, response in enumerate(responses):
                    if response.get("click", 0) == 1:
                        clicked_items[consumer_id].append(recommended_items[i])
                        recommenders[recommender_id].record_click(consumer_id, recommended_items[i])
                        recommenders[recommender_id].add_interaction(
                            consumer_id,
                            recommended_items[i].item_id,
                            model.predict(consumer_id, recommended_items[i].item_id).est,
                        )
        
        # ---- Switching Logic (after day iterations) ----
        switch_occurred = False
        if cycle > 5:
            for consumer in consumers:
                for rec_id, status in consumer.connected_recommenders.copy().items():
                    if status == 0:
                        continue
                    satisfaction = consumer.get_satisfaction_score(recommender_system_id=rec_id)
                    if satisfaction < 0.1:
                        if (max(consumer.satisfaction_scores, key=consumer.satisfaction_scores.get) == rec_id and 
                                all(score > 0 for score in consumer.satisfaction_scores.values())):
                            break
                        
                        if rec_id == mainstream_recommender.recommender_id:
                            old_rec = mainstream_recommender
                            new_rec = niche_recommender
                        else:
                            old_rec = niche_recommender
                            new_rec = mainstream_recommender
                        
                        # Retrieve interactions if transferring is enabled
                        old_interactions = old_rec.get_user_interactions(consumer.consumer_id) if transfer_interactions else []
                        old_rec.disconnect_consumer(consumer)
                        consumer.unsubscribe_from_recommender_system(old_rec.recommender_id)
                        if forget_interactions:
                            old_rec.remove_user_interactions(consumer.consumer_id)
                        consumer.remove_user(retain_profile=not forget_interactions)
                        if transfer_interactions:
                            for interaction in old_interactions:
                                new_rec.add_interaction(interaction.user_id, interaction.item_id, interaction.rating)
                        new_rec.connect_consumer(consumer)
                        consumer.subscribe_to_recommender_system(new_rec.recommender_id)
                        switch_occurred = True
                        break
        
        # ---- Log provider, consumer, and recommender data ----
        for recommender in recommenders.values():
            recommender.charge_subscription_fees()
        
        for provider in providers:
            for rec_id, recommender in recommenders.items():
                if rec_id in provider.connected_recommenders.keys():
                    provider_data.append([
                        provider.provider_id,
                        rec_id,
                        provider.profit.get(rec_id, [0])[cycle-1] if cycle <= len(provider.profit.get(rec_id, [0])) else 0,
                        cycle,
                        provider.pay_cycle_fee.get(rec_id, [0])[cycle-1] if cycle <= len(provider.pay_cycle_fee.get(rec_id, [0])) else 0,
                        provider.pay_cycle_clicks.get(rec_id, [0])[cycle-1] if cycle <= len(provider.pay_cycle_clicks.get(rec_id, [0])) else 0,
                        provider.pay_cycle_shows.get(rec_id, [0])[cycle-1] if cycle <= len(provider.pay_cycle_shows.get(rec_id, [0])) else 0,
                        len([r for r in provider.connected_recommenders.keys() if provider.connected_recommenders[r] == 1]),
                    ])
                    
        for consumer in consumers:
            for rec_id in recommenders.keys():
                if rec_id in consumer.connected_recommenders.keys():
                    consumer_id = consumer.consumer_id
                    satisfaction = consumer.get_satisfaction_score(recommender_system_id=rec_id)
                    rec_state = 'connected' if consumer.connected_recommenders[rec_id] == 1 else ''
                    consumer_data.append([consumer_id, rec_id, cycle, satisfaction, rec_state])
        
        for provider in providers:
            unsubscribed_list = provider.update_recommender_subscription()
            for rec_id, recommender in recommenders.items():
                if rec_id in unsubscribed_list:
                    recommender.disconnect_provider(provider.provider_id)
        
        for rec_id, recommender in recommenders.items():
            num_connected_providers = len(recommender.connected_providers)
            num_connected_consumers = len(recommender.connected_consumers)
            total_profit = recommender.profit[-1]
            recommender_data.append([rec_id, num_connected_providers, num_connected_consumers, total_profit, cycle])
        
        print("\n====================================================")
        print("===============> Finished Cycle:", cycle, "<================")
        print("====================================================\n")
    
    # Convert lists to DataFrames
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
