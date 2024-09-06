import pandas as pd
import numpy as np

# Function to calculate Bayesian average per user
def bayesian_average_per_user(rating, user_avg, reg_factor):
    bayesian_avg = (rating + reg_factor * user_avg) / (1 + reg_factor)
    return bayesian_avg

def normalize(vector):
    total = sum(vector.values())
    if total == 0:
        return {key: 0 for key in vector}  # Handle case where total is zero
    return {key: value / total for key, value in vector.items()}

def generate_genre_preferences(movies_with_ratings_df):
    # User Genre preferences using bayesian average
    
    # Step 1: Explode genres
    df_exploded = movies_with_ratings_df.copy()
    df_exploded['genres'] = df_exploded['genres'].str.split('|')
    df_exploded = df_exploded.explode('genres')

    # Step 2: Calculate Bayesian average for each genre per user
    grouped = df_exploded.groupby(['userId', 'genres'])['rating']
    genre_avg = grouped.mean().reset_index()

    # Calculate overall average rating per user
    user_avg_ratings = df_exploded.groupby('userId')['rating'].mean()

    # Regularization factor per user (you can adjust this based on your dataset characteristics)
    reg_factor = 3

    # Apply Bayesian average calculation per user
    genre_avg['bayesian_avg_rating'] = genre_avg.apply(lambda row: bayesian_average_per_user(row['rating'], user_avg_ratings[row['userId']], reg_factor), axis=1)

    genre_avg['rating'] = genre_avg['bayesian_avg_rating'].apply(lambda x: (x - 1) / 4)

    # Step 3: Pivot the table to have genres as columns
    user_genre_preference = genre_avg.pivot(index='userId', columns='genres', values='rating').fillna(0)
    user_genre_preference = user_genre_preference.drop('(no genres listed)', axis=1, errors='ignore')  # Handle cases where column may not exist

    # Normalize genre preferences for each user
    normalized_user_genre_preference = user_genre_preference.apply(lambda row: normalize(row.to_dict()), axis=1)
    normalized_user_genre_preference = pd.DataFrame(normalized_user_genre_preference.tolist(), index=user_genre_preference.index)

    return normalized_user_genre_preference
