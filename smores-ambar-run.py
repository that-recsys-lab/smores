import os
import pickle
import pandas as pd
from smores.stakeholders.stakeholders import Provider
from smores.preprocessing.genre_preferences import genre_preferences
from smores.recommender.SVD import SurpriseSVD
from smores.simulation.monolithic import monolithic
from smores.simulation.threshold_swithcing import threshold_switching
from smores.simulation.ucb_switching import ucb_switching
from smores.utils.run import run_experiment
from smores.utils.train_model import train_model
from smores.stakeholders.choice import category_similarity_logit
from smores.preprocessing.create_item_objects import create_item_objects_from_csv
from smores.preprocessing.historical_distribution import historical_distribution

def get_top_items(ratings_df, items_df, n=30, genre=None):
    pop = ratings_df.groupby("itemId").size().reset_index(name="num_ratings")
    if genre:
        # pick only items whose first listed genre matches
        items_df["first_genre"] = items_df["genres"].str.split("|").str[0]
        pop = pop[pop["itemId"].isin(
            items_df[items_df["first_genre"].str.lower()==genre.lower()]["itemId"]
        )]
    return pop.sort_values("num_ratings", ascending=False)["itemId"].tolist()[:n]

def run_for_ambar(dataset_dir, experiment_name, scenario):
    print(f"\n=== Scenario: {scenario['name']} – forget={scenario['forget_interactions']}, transfer={scenario['transfer_interactions']} ===\n")
    # 1) load raw CSVs
    ratings_df = pd.read_csv(os.path.join(dataset_dir, "ratings_info.csv"))
    items_df   = pd.read_csv(os.path.join(dataset_dir, "tracks_info.csv"))

    # 2) rename to common schema
    ratings_df.rename(columns={"user_id":"consumerId","track_id":"itemId","rating":"rating"}, inplace=True)
    items_df.rename(columns={"track_id":"itemId","artist_id":"providerId","category_styles":"genres"}, inplace=True)

    # 3) merge genre info, then aggregate per‐item
    merged = ratings_df.merge(items_df, on="itemId", how="left")
    items_agg = (
        merged[["itemId","rating","genres","providerId"]]
        .groupby(["itemId","genres","providerId"])
        .mean().reset_index()
    )

    # 4) train or load global SVD model
    model_path = os.path.join("experiments","results", experiment_name, "model.pkl")
    if os.path.exists(model_path):
        try:
            print("Loading existing model…")
            with open(model_path,"rb") as f: model = pickle.load(f)
        except:
            print("↪ failed; retraining")
            model = train_model(ratings_df, experiment_name)
    else:
        print("Training new model…")
        model = train_model(ratings_df, experiment_name)

    # 5) historical distributions + genre‐based consumer prefs
    hist_dist = historical_distribution(ratings_df, items_agg, dataset_dir)
    consumers_list, _ = genre_preferences(merged, hist_dist, dataset_dir)

    # 6) build providers
    unique_genres = items_agg["genres"].str.split("|").explode().unique()
    providers = []
    for pid, grp in items_agg.groupby("providerId"):
        objs = create_item_objects_from_csv(grp, pid, unique_genres)
        providers.append(Provider(pid, objs))

    # 7) get popular item lists
    popular   = get_top_items(ratings_df, items_agg, 30)
    niche_pop = get_top_items(ratings_df, items_agg, 30, genre="Soul/funk")

    # 8) clear big dfs
    del ratings_df, items_df, merged

    # 9) run both switching experiments
    for tag, fn in [("threshold_switching", threshold_switching),
                    ("ucb_switching",       ucb_switching)]:
        exp_name = f"{experiment_name}_{tag}_{scenario['name']}"
        print(f"---> running {tag} → {exp_name}")
        run_experiment(
            experiment=fn,
            model=model,
            consumer_choice_model=category_similarity_logit,
            experiment_name=exp_name,
            num_days=10,
            num_cycles=10,
            slate_size=5,
            consumers=consumers_list,
            providers=providers,
            recommenders=[
                {
                    "type": SurpriseSVD,
                    "name": "mainstream_recommender",
                    "params": {
                        "fee_per_click": 0.1,
                        "fee_per_show": 0.01,
                        "base_fee": 0.0,
                        "most_popular_item_ids": popular,
                        "prohibited_genres": set(),
                        "weighted_category": {},
                        "specialized_genres": set(),
                        "consumers": consumers_list,
                    },
                },
                {
                    "type": SurpriseSVD,
                    "name": "niche_recommender",
                    "params": {
                        "fee_per_click": 0.1,
                        "fee_per_show": 0.01,
                        "base_fee": 0.0,
                        "most_popular_item_ids": niche_pop,
                        "prohibited_genres": set(),
                        "weighted_category": {},
                        "specialized_genres": {"Soul/funk"},
                        "consumers": [],
                    },
                },
            ],
            base_dir="experiments/results",
            forget_interactions=scenario["forget_interactions"],
            transfer_interactions=scenario["transfer_interactions"]
        )
    print(f"=== done scenario {scenario['name']} ===\n")

if __name__ == "__main__":
    dataset_directory = "data/raw/ambar"
    experiment_name  = "ambar-threshold-switching"

    scenarios = [
        {"name": "cold_start",                "forget_interactions": True,  "transfer_interactions": False},
        {"name": "user_ownership",            "forget_interactions": True,  "transfer_interactions": True },
        {"name": "universal_profile",         "forget_interactions": False, "transfer_interactions": True },
        {"name": "algorithm_specific_profile","forget_interactions": False, "transfer_interactions": False},
    ]

    for scen in scenarios:
        run_for_ambar(dataset_directory, experiment_name, scen)

    print("All AMBAR scenarios complete.")
