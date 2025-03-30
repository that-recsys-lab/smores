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
    Run a multi-recommender experiment with UCB switching, supporting profile handling.

    Args:
        consumers (list): List of Consumer instances.
        providers (list): List of Provider instances.
        recommenders (dict): Dictionary of Recommender instances keyed by IDs.
        num_days (int): Number of days per cycle (default: 5).
        slate_size (int): Number of items to recommend (default: 3).
        num_cycles (int): Number of cycles to run (default: 1).
        model: Prediction model for interaction ratings.
        forget_interactions (bool): If True, delete interactions on switch.
        transfer_interactions (bool): If True, transfer interactions to new recommender.

    Returns:
        pd.DataFrame: Provider data.
        pd.DataFrame: Consumer data.
        pd.DataFrame: Recommender system data.
        pd.DataFrame: Consumers' recommender choice history.
    """
    # Initialize data logs
    provider_data = []
    consumer_data = []
    recommender_data = []
    choice_history = []

    # Extract recommenders (assuming two: mainstream and niche)
    mainstream_rec, niche_rec = list(recommenders.values())[0], list(recommenders.values())[1]

    # Connect all consumers and providers to both recommenders initially
    for consumer in consumers:
        for rec in [mainstream_rec, niche_rec]:
            rec.connect_consumer(consumer)
            consumer.subscribe_to_recommender_system(rec.recommender_id)
    for provider in providers:
        for rec in [mainstream_rec, niche_rec]:
            rec.connect_provider(provider)
            provider.subscribe_to_recommender_system(rec.recommender_id)

    print(f"Initial mainstream consumers: {len(mainstream_rec.connected_consumers)}")
    print(f"Initial niche consumers: {len(niche_rec.connected_consumers)}")

    # Run experiment across cycles
    for cycle in range(1, num_cycles + 1):
        # Group consumers by chosen recommender
        consumers_by_rec = {rec_id: [] for rec_id in recommenders}
        for consumer in consumers:
            # Optional: Force mainstream for first 5 cycles (uncomment to enforce)
            # if cycle <= 5:
            #     chosen_rec_id = mainstream_rec.recommender_id
            # else:
            #     chosen_rec_id = consumer.choose_recommender()
            
            # Default behavior: Preselect mainstream in cycle 1, then UCB
            chosen_rec_id = (
                consumer.choose_recommender(preselected_recommender_id=mainstream_rec.recommender_id)
                if cycle == 1
                else consumer.choose_recommender()
            )
            if chosen_rec_id:
                choice_history.append([consumer.consumer_id, chosen_rec_id])
                consumers_by_rec[chosen_rec_id].append(consumer)

        for rec_id, count in consumers_by_rec.items():
            print(f"Cycle {cycle}: {rec_id} chosen by {len(count)} consumers")

        # Simulate daily interactions
        for day in range(1, num_days + 1):
            print(f"Cycle {cycle}, Day {day}")
            recommendations = {}
            for rec_id, rec_consumers in consumers_by_rec.items():
                rec = recommenders[rec_id]
                slate = rec.recommend_items(rec_consumers, slate_size=slate_size)
                for consumer in rec_consumers:
                    recommendations[consumer.consumer_id] = (rec_id, slate[consumer.consumer_id])

            clicked_items = {c.consumer_id: [] for c in consumers}
            for consumer in consumers:
                if not consumer.available_recommenders:
                    print(f"Consumer {consumer.consumer_id} has no recommenders")
                    continue
                rec_id, items = recommendations.get(consumer.consumer_id, (None, []))
                if not rec_id:
                    continue
                responses = consumer.simulate_response(items, recommender_system_id=rec_id)
                rec = recommenders[rec_id]
                for i, resp in enumerate(responses):
                    if resp.get("click", 0) == 1:
                        clicked_items[consumer.consumer_id].append(items[i])
                        rec.record_click(consumer.consumer_id, items[i])
                        rec.add_interaction(
                            consumer.consumer_id,
                            items[i].item_id,
                            model.predict(consumer.consumer_id, items[i].item_id).est
                        )

        # Switching logic (after cycle 5)
        if cycle > 5:
            for consumer in consumers:
                for rec_id, state in consumer.connected_recommenders.copy().items():
                    if state != 1 or consumer.satisfaction_scores[rec_id] >= 0.1:
                        continue
                    if (rec_id == max(consumer.satisfaction_scores, key=consumer.satisfaction_scores.get) and 
                        all(s > 0 for s in consumer.satisfaction_scores.values())):
                        continue
                    old_rec = mainstream_rec if rec_id == mainstream_rec.recommender_id else niche_rec
                    new_rec = niche_rec if old_rec == mainstream_rec else mainstream_rec

                    # Handle interactions
                    interactions = old_rec.get_user_interactions(consumer.consumer_id) if transfer_interactions else []
                    old_rec.disconnect_consumer(consumer)
                    consumer.unsubscribe_from_recommender_system(old_rec.recommender_id)
                    if forget_interactions:
                        old_rec.remove_user_interactions(consumer.consumer_id)
                        consumer.remove_user(retain_profile=False)
                    else:
                        consumer.remove_user(retain_profile=True)
                    if transfer_interactions:
                        for inter in interactions:
                            new_rec.add_interaction(inter.user_id, inter.item_id, inter.rating)
                    new_rec.connect_consumer(consumer)
                    consumer.subscribe_to_recommender_system(new_rec.recommender_id)
                    break  # One switch per consumer

        # Record data at cycle end
        for rec in recommenders.values():
            rec.charge_subscription_fees()
        for provider in providers:
            for rec_id in provider.connected_recommenders:
                rec = recommenders[rec_id]
                provider_data.append([
                    provider.provider_id, rec_id, 
                    provider.profit.get(rec_id, [0])[cycle-1] if cycle <= len(provider.profit.get(rec_id, [0])) else 0,
                    cycle,
                    provider.pay_cycle_fee.get(rec_id, [0])[cycle-1] if cycle <= len(provider.pay_cycle_fee.get(rec_id, [0])) else 0,
                    provider.pay_cycle_clicks.get(rec_id, [0])[cycle-1] if cycle <= len(provider.pay_cycle_clicks.get(rec_id, [0])) else 0,
                    provider.pay_cycle_shows.get(rec_id, [0])[cycle-1] if cycle <= len(provider.pay_cycle_shows.get(rec_id, [0])) else 0,
                    sum(1 for r in provider.connected_recommenders if provider.connected_recommenders[r] == 1)
                ])
        for consumer in consumers:
            for rec_id in consumer.connected_recommenders:
                consumer_data.append([
                    consumer.consumer_id, rec_id, cycle,
                    consumer.get_satisfaction_score(rec_id),
                    'connected' if consumer.connected_recommenders[rec_id] == 1 else '',
                    None  # kl_div placeholder
                ])
        for rec_id, rec in recommenders.items():
            recommender_data.append([rec_id, len(rec.connected_providers), len(rec.connected_consumers), rec.profit[-1], cycle])

        # Update provider subscriptions
        for provider in providers:
            unsubscribed = provider.update_recommender_subscription()
            for rec_id in unsubscribed:
                recommenders[rec_id].disconnect_provider(provider.provider_id)

        print(f"\nCycle {cycle} completed\n")

    # Convert to DataFrames
    return (
        pd.DataFrame(provider_data, columns=["provider_id", "recommender_id", "profit", "cycle", "fee", "clicks", "shows", "connected_recommenders"]),
        pd.DataFrame(consumer_data, columns=["consumer_id", "recommender_id", "cycle", "satisfaction_score", "rec_state", "kl_divergence"]),
        pd.DataFrame(recommender_data, columns=["recommender_id", "num_connected_providers", "num_connected_consumers", "total_profit", "cycle"]),
        pd.DataFrame(choice_history, columns=["consumer_id", "recommender_id"])
    )