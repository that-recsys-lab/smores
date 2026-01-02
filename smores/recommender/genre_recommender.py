from dataclasses import dataclass
from sys import maxsize

from lenskit.pipeline import PipelineBuilder
from lenskit.basic.candidates import UnratedTrainingItemsCandidateSelector
from lenskit.basic import UserTrainingHistoryLookup, TopNRanker
from lenskit.knn import ItemKNNConfig, ItemKNNScorer
from lenskit.data import ID
from lenskit.data import ItemList, QueryInput, RecQuery

from .recommender import RecommenderFactory
from .lk_recommenders import ImplicitMFRecommender, ItemKnnRecommender, BPRRecommender

import smores

@dataclass
class MyGenreConfig:
    genre_list: list[int] | None = None

class UnratedItemsGenreCandidateSelector (UnratedTrainingItemsCandidateSelector):
    """
    Candidate selector that selects all known items from the training data that
    do not appear in the request user's history (:attr:`RecQuery.user_items`),
    and then filters those items based on a genre list. 
    If no item history is available, then all training items are returned.

    In order to look up the user's history in the training data, this needs to
    be combined with a component like
    :class:`~.history.UserTrainingHistoryLookup`.

    Stability:
        Caller
    """

    config: MyGenreConfig

    def __call__(self, query: QueryInput) -> ItemList:
        query = RecQuery.create(query)
        items = ItemList.from_vocabulary(self.items_)

        if query.user_items is not None:
            items = items.remove(numbers=query.user_items.numbers(vocabulary=self.items_))

        genre_mask = [(int(id) in self.config.genre_list) for id in items.ids()]

        items = items[genre_mask]

        return items

class ImplicitMFGenreRecommender(ImplicitMFRecommender):
    def __init__(self):
        super().__init__()

    def setup(self, config):
        params = config.params or {}

        self.genre = int(params['genre_feature'])
        # Maybe should be a set
        self.genre_items = smores.Smores.state.items.get_genre_items(self.genre)

        super().setup(config)


    def build_pipeline(self):
        selector_config = MyGenreConfig()
        selector_config.genre_list = self.genre_items
        return self._build_history_pipeline(
            self.get_scorer(),
            candidate_selector=UnratedItemsGenreCandidateSelector,
            selector_config=selector_config,
        )

    def train(self):
        if self._should_train(self.min_user_count, self.min_interaction_count):
            super().train()

    def isDatasetViable(self):
        return self._dataset_viable(self.min_user_count, self.min_interaction_count)
        
    def isProfileViable(self, user_id: ID):
        return self._profile_viable(user_id, self.min_profile_size)
            
class GenreKnnRecommender(ItemKnnRecommender):
    def __init__(self):
        super().__init__()
        self.min_user_count = maxsize
        self.min_interaction_count = maxsize
        self.min_profile_size = maxsize

    def setup(self, config):
        params = config.params or {}

        self.genre = int(params['genre_feature'])
        # Maybe should be a set
        self.genre_items = smores.Smores.state.items.get_genre_items(self.genre)

        super().setup(config)

    def build_pipeline(self):
        selector_config = MyGenreConfig()
        selector_config.genre_list = self.genre_items
        return self._build_history_pipeline(
            self.get_scorer(),
            candidate_selector=UnratedItemsGenreCandidateSelector,
            selector_config=selector_config,
        )

    def train(self):
        if self._should_train(self.min_user_count, self.min_interaction_count):
            super().train()

    def isDatasetViable(self):
        return self._dataset_viable(self.min_user_count, self.min_interaction_count)

        
    def isProfileViable(self, user_id: ID):
        return self._profile_viable(user_id, self.min_profile_size)

class GenreBPRRecommender(BPRRecommender):
    def __init__(self):
        super().__init__()

    def setup(self, config):
        params = config.params or {}

        self.genre = int(params['genre_feature'])
        self.genre_items = smores.Smores.state.items.get_genre_items(self.genre)

        super().setup(config)

    def build_pipeline(self):
        selector_config = MyGenreConfig()
        selector_config.genre_list = self.genre_items
        return self._build_history_pipeline(
            self.get_scorer(),
            candidate_selector=UnratedItemsGenreCandidateSelector,
            selector_config=selector_config,
        )

    def train(self):
        if self._should_train(self.min_user_count, self.min_interaction_count):
            super().train()

    def isDatasetViable(self):
        return self._dataset_viable(self.min_user_count, self.min_interaction_count)

    def isProfileViable(self, user_id: ID):
        return self._profile_viable(user_id, self.min_profile_size)


RecommenderFactory.register('genre_implicit_mf', ImplicitMFGenreRecommender)
RecommenderFactory.register('genre_knn', GenreKnnRecommender)
RecommenderFactory.register('genre_bpr', GenreBPRRecommender)
