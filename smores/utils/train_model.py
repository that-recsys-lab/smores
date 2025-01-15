import os
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


def train_model(ratings_df, experiment_name):
    # Prepare the data for Surprise
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
    print("Data info:", len(data.raw_ratings))

    # Split the data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Train the SVD model
    model = SVD()
    model.fit(trainset)

    # Test the model
    predictions = model.test(testset)

    # Construct the directory and filename
    directory = f"experiments/results/{experiment_name}"
    model_filename = f"{directory}/model.pkl"

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Save the model to a pickle file
    with open(model_filename, "wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Model saved to {model_filename}")
    return model
