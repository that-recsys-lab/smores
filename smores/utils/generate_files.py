"""
generate_files.py

Generates three output CSV files (consumer.csv, items.csv, providers.csv) and a genre_index.json file
from consumer-item interaction data and item metadata.

Usage:
    python generate_files.py \
        --data_dir ./data \
        --ratings_file consumer_ratings.csv \
        --items_file provider_items.csv \
        --niche_genres "Horror" "Soul/Funk" \
        --output_dir ./processed_output

Arguments:
    --data_dir        Path to the folder containing your input files
    --ratings_file    Filename for the consumer-item interaction file (e.g., consumer_ratings.csv)
    --items_file      Filename for the item metadata and provider mapping file (e.g., provider_items.csv)
    --niche_genres    List of genres that define niche consumers and providers
    --output_dir      Folder to save output files (default: ./output)

Expected Input File Formats:

1. consumer_ratings.csv:
    Columns: userId, item_id, rating, timestamp
    Example:
        userId,item_id,rating,timestamp
        1,101,4,978300760
        2,102,5,978302109

2. provider_items.csv:
    Columns: item_id, providerId, genres
    Example:
        item_id,providerId,genres
        101,201,Comedy|Drama
        102,202,Horror

Output Files:
    - consumer.csv        → consumer_id, consumer_type, preferences
    - items.csv           → item_id, provider_id, features
    - providers.csv       → provider_id, provider_type
    - genre_index.json    → index-to-genre mapping for interpreting genre vectors

Example Bash Command:
    python generate_files.py \
    --data_dir ./data \
    --ratings_file consumer_ratings.csv \
    --items_file provider_items.csv \
    --niche_genres "Horror" "Soul/Funk" \
    --output_dir ./processed_output

Example Python code to call the main function directly:
    from smores.utils.generate_files import main

    data_dir = "./data"
    ratings_file = "consumer_ratings.csv"
    items_file = "provider_items.csv"
    niche_genres = ["Horror", "Soul/Funk"]
    output_dir = "./processed_output"

    # Construct full paths
    ratings_path = f"{data_dir}/{ratings_file}"
    items_path = f"{data_dir}/{items_file}"

    # Call the main function
    main(ratings_path, items_path, niche_genres, output_dir)
"""

import pandas as pd
import numpy as np
import os
import argparse
import json
from collections import defaultdict
from typing import List, Dict

NO_GENRE = "(no genres listed)"


def normalize_vector(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    return (
        {g: c / total for g, c in counts.items()} if total else {g: 0.0 for g in counts}
    )


def genre_to_vector(genres: List[str], genre_index_map: Dict[str, int]) -> List[float]:
    vector = [0] * len(genre_index_map)
    for g in genres:
        if g == NO_GENRE:
            continue
        if g in genre_index_map:
            vector[genre_index_map[g]] += 1
    total = sum(vector)
    return [v / total if total else 0.0 for v in vector]


def safe_split_genres(cell):
    if isinstance(cell, list):
        return [g for g in cell if g != NO_GENRE]
    if pd.isna(cell):
        return []
    return [g for g in str(cell).split("|") if g != NO_GENRE]


def generate_consumer_preferences(
    ratings_df: pd.DataFrame, items_df: pd.DataFrame, niche_genres: List[str]
):
    items_df["genres"] = items_df["genres"].apply(safe_split_genres)
    merged = pd.merge(ratings_df, items_df, on="item_id")

    all_genres = sorted({g for lst in items_df["genres"] for g in lst})
    genre_index_map = {g: i for i, g in enumerate(all_genres)}

    records = []
    for user_id, grp in merged.groupby("userId"):
        genre_counts = defaultdict(int)
        for genres in grp["genres"]:
            for g in genres:
                genre_counts[g] += 1
        norm = normalize_vector(genre_counts)
        vector = [norm.get(g, 0.0) for g in all_genres]
        top_genre = max(norm, key=norm.get) if norm else None
        consumer_type = "Niche" if top_genre in niche_genres else "Generic"
        records.append(
            {
                "consumer_id": int(user_id),
                "consumer_type": consumer_type,
                "preferences": json.dumps(vector),
            }
        )
    consumer_df = pd.DataFrame(records)
    return consumer_df, genre_index_map


def generate_items_and_providers(
    items_df: pd.DataFrame, niche_genres: List[str], genre_index_map: Dict[str, int]
):
    items_df["genres"] = items_df["genres"].apply(safe_split_genres)

    item_records = []
    provider_bucket = defaultdict(list)

    for _, row in items_df.iterrows():
        item_id = int(row["item_id"])
        provider_id = int(row["providerId"])
        genres = row["genres"]
        vec = genre_to_vector(genres, genre_index_map)
        item_records.append(
            {
                "item_id": item_id,
                "provider_id": provider_id,
                "features": json.dumps(vec),
            }
        )
        provider_bucket[provider_id].extend(genres)

    provider_records = []
    for pid, g_list in provider_bucket.items():
        g_list = [g for g in g_list if g != NO_GENRE]
        if g_list:
            dominant = pd.Series(g_list).value_counts().idxmax()
            ptype = "Niche" if dominant in niche_genres else "Generic"
        else:
            ptype = "Generic"
        provider_records.append({"provider_id": pid, "provider_type": ptype})

    return pd.DataFrame(item_records), pd.DataFrame(provider_records)


def main(ratings_path: str, items_path: str, niche_genres: List[str], output_dir: str):
    ratings_df = pd.read_csv(
        ratings_path, skiprows=1, names=["userId", "item_id", "rating", "timestamp"]
    )
    items_df = pd.read_csv(
        items_path, skiprows=1, names=["item_id", "providerId", "genres"]
    )

    consumer_df, genre_index_map = generate_consumer_preferences(
        ratings_df, items_df, niche_genres
    )
    items_df_out, providers_df_out = generate_items_and_providers(
        items_df, niche_genres, genre_index_map
    )

    os.makedirs(output_dir, exist_ok=True)
    consumer_df.to_csv(os.path.join(output_dir, "consumers.csv"), index=False)
    items_df_out.to_csv(os.path.join(output_dir, "items.csv"), index=False)
    providers_df_out.to_csv(os.path.join(output_dir, "providers.csv"), index=False)

    with open(os.path.join(output_dir, "genre_index.json"), "w") as f:
        json.dump({i: g for g, i in genre_index_map.items()}, f, indent=2)

    print("Files written to:", output_dir)


if __name__ == "__main__":
    assert safe_split_genres("Action|Horror|(no genres listed)") == ["Action", "Horror"]
    assert safe_split_genres(["Drama", NO_GENRE, "Comedy"]) == ["Drama", "Comedy"]
    assert NO_GENRE not in genre_to_vector([NO_GENRE, "Action"], {"Action": 0})
    print("All basic tests passed.")
