# Multi-stakeholder Recommendation Simulation Framework

This repository contains the code for the simulation framework. We recommend creating a separate repository for running experiments.

## Installation

### 1. Create a Virtual Environment
The package is set up to use the `uv` package manager. Run the following commands to set up the virtual environment:

```bash
# Create a virtual environment
uv sync

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies
Once the virtual environment is activated, install the required packages:

```bash
uv pip install -e .
```

install latest lenskit
```bash
uv pip install -U git+https://github.com/lenskit/lkpy
```

## Running the test file
```bash
python -m tests.test_smores
```


## Components
The following simulation components are implemented:

### Recommenders

- `FixedItemRecommender` (label: fixed_recommender): **Only useful for testing.** Recommends a fixed set of items from dataset. It does remove items already recommended to a given user to prevent duplicate interactions in the dataset.
- `PopularRecommender` (label: popular): Wrapper for the LensKit `PopScorer` class. Recommends popular itesm within the dataset.
- `ItemKnnRecommender` (label: item_knn): Wrapper for the LensKit `ItemKNNScorer`. 
- `ImplicitMFRecommender` (label: implicit_mf): Wrapper for the LensKit `lenskit.implicit.ALS` recommender.
- `FileRecommender` (label: file_based): Similar to FixedItemRecommender. Reads a file of popular items in a genre and recommends randomly from those.

Recommended structure for a recommender is to have PopularRecommender as the cold user fallback and PopularFromFileRecommender as the cold start fallback.

### Consumer Utility Models 

- `ConsumerFixedUtilityModel` (label: fixed_utility): **Only useful for testing.** Assigns a fixed utility to every recommendation list.
- `ConsumerPrefCosineAvgUtilityModel` (label: list_average): Computes the cosine of the user's genre preference vector with the item's genre vector and averages over the recommendation list.

### Item Selection Models

- `CategorySimilarityLogitModel` (label: category_similarity_logit): Constructs a probability distribution over the items in the recommendation list based on the dot product of the user's genre preference vector with the item's genre vector and probabilistically selects an item to represent the user's click.

### Recommender Choice Models

- `FixedRecommenderChoiceModel` (label: fixed): **Only useful for testing.** Always picks the same recommender that the consumer is already connected to. 
- `ThresholdRecommenderChoiceModel` (label: threshold): Maintains an aggregate utility score based on list utility. When the utility falls below the given threshold, moves to another recommender, if there is one with highest aggregate utility.
- `UCBRecommenderChoiceModel` (label: ucb): Maintains an aggregate utility score based on list utility. Calculates an upper confidence bound for the utility and moves to the recommender with the highest UCB.

### Provider Utility Models

- `ProviderClickFixedUtilityModel` (label: click_fixed): Provides a fixed utility when a consumer clicks on a provider's item.
- Still needed: a model that has utility for exposure. Not essential for the paper, though.

### Triggers

- `InitialBurnInTrigger (CycleTrigger)` (label: initial_burnin): Keeps recommender allocation fixed until a given number of cycles is past. 
- `UniversalProfileTrigger (InteractionBatchTrigger)` (label: universal_profile): Copies interactions from a given day to all available recommenders. 
- `ProfileColdStartTrigger` (SwitchTrigger) (label: cold_start): When a user switches away from their current recommender, the profile on that recommender (the "from" recommender) is deleted.
- `ProfileUserOwnershipTrigger` (SwitchTrigger) (label: user_ownership): When a user switches away from their current recommender, the profile on the "from" recommender is deleted and added to the new recommender, the "to" recommender.
- `SwitchSaveInfoTrigger` (SwitchTrigger) (label: save_switch): **Only useful for testing.** Creates a list of all the switching events.



## License
This project is licensed under the terms specified in the `LICENSE` file.
