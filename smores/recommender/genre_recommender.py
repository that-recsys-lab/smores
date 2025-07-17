from dataclasses import dataclass
from sys import maxsize

from lenskit.pipeline import PipelineBuilder
from lenskit.basic.candidates import UnratedTrainingItemsCandidateSelector
from lenskit.basic import UserTrainingHistoryLookup, TopNRanker
from lenskit.knn import ItemKNNConfig, ItemKNNScorer
from lenskit.als import ImplicitMFConfig, ImplicitMFScorer
from lenskit.data import ID
from lenskit.data import ItemList, QueryInput, RecQuery

from .recommender import RecommenderFactory
from .lk_recommenders import ImplicitMFRecommender, ItemKnnRecommender

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

    # Note: not calling superclass setup because we need to build our own pipeline.
    def setup(self, config):
        params = config.params
        if params is not None:
            if 'cold_start_fallback' in params:
                self.cold_start_fallback = params['cold_start_fallback']
            if 'cold_user_fallback' in params: 
                self.cold_user_fallback = params['cold_user_fallback']
        self.name = config.name

        self.embedding_size = int(params['embedding_size'])
        self.epochs = int(params['epochs'])
        self.regularization = float(params['regularization'])
        self.positive_weight = float(params['positive_weight'])

        self.genre = int(params['genre_feature'])
        # Maybe should be a set
        self.genre_items = smores.Smores.state.items.get_genre_items(self.genre)

        self.min_user_count = int(params['min_user_count'])
        self.min_interaction_count = int(params['min_interaction_count'])
        self.min_profile_size = int(params['min_profile_size'])

        self.lk_config = ImplicitMFConfig(embedding_size=self.embedding_size,
                                          epochs=self.epochs,
                                          regularization=self.regularization,
                                          weight=self.positive_weight,
                                          user_embeddings=True,
                                          use_ratings=False)

        self.scorer = ImplicitMFScorer(self.lk_config)
        self.pipeline = self.build_pipeline()


    def build_pipeline(self):
        scorer = self.get_scorer()
        slate_size = smores.Smores.state.slate_size

        pipe = PipelineBuilder()
        # define an input parameter for the user ID (the 'query')
        query = pipe.create_input('query', ID)
        # look up a user's history in the training data
        history = pipe.add_component('history-lookup', UserTrainingHistoryLookup, query=query)
        # find candidates from the training data
        # Adding the candidate selector for the specific niche genre
        selector_config = MyGenreConfig()
        selector_config.genre_list = self.genre_items
        default_candidates = pipe.add_component('candidate-selector',
            UnratedItemsGenreCandidateSelector, selector_config,
            query=history)
        # score the candidate items using the specified scorer
        score = pipe.add_component('scorer', scorer, query=query, items=default_candidates)
        # rank the items by score
        recommend = pipe.add_component('ranker', TopNRanker, {'n': slate_size}, items=score)
        pipe.alias('recommender', recommend)
        pipe.default_component('recommender')
        return pipe.build()

    def train(self):
        if self.get_dataset().interaction_count >= self.min_interaction_count and \
                self.get_dataset().user_count >= self.min_user_count:
            super().train()

    def isDatasetViable(self):
        if not self.trained:
            return False
        else:
            user_count = self.dataset_active_users()
            interaction_count = self.get_dataset().interaction_count
            if user_count >= self.min_user_count and interaction_count >= self.min_interaction_count:
                return True
            else:
                return False
        
    def isProfileViable(self, user_id: ID):
        items = self.get_dataset().user_row(user_id)
        if items is None:
            return False
        else:
            profile_size = items.ids().size
            if profile_size < self.min_profile_size:
                return False
            else:
                return True
            
class GenreKnnRecommender(ItemKnnRecommender):
    def __init__(self):
        super().__init__()
        self.min_user_count = maxsize
        self.min_interaction_count = maxsize
        self.min_profile_size = maxsize

    # Note: not calling superclass setup because we need to build our own pipeline.
    def setup(self, config):
        params = config.params
        if params is not None:
            if 'cold_start_fallback' in params:
                self.cold_start_fallback = params['cold_start_fallback']
            if 'cold_user_fallback' in params: 
                self.cold_user_fallback = params['cold_user_fallback']
        self.name = config.name

        # get data from config
        params = config.params
        max_nbrs = int(params['max_neighbors'])
        min_nbrs = int(params['min_neighbors'])
        min_sim = float(params['min_similarity'])

        self.genre = int(params['genre_feature'])
        # Maybe should be a set
        self.genre_items = smores.Smores.state.items.get_genre_items(self.genre)

        self.min_user_count = int(params['min_user_count'])
        self.min_interaction_count = int(params['min_interaction_count'])
        self.min_profile_size = int(params['min_profile_size'])
        # create ItemKNNConfig object
        self.lk_config = ItemKNNConfig(max_nbrs=max_nbrs, min_nbrs=min_nbrs, 
                                       min_sim=min_sim, feedback='implicit')
        # create ItemKNNScorer
        self.scorer = ItemKNNScorer(self.lk_config)
        self.pipeline = self.build_pipeline()

    def build_pipeline(self):
        scorer = self.get_scorer()
        slate_size = smores.Smores.state.slate_size

        pipe = PipelineBuilder()
        # define an input parameter for the user ID (the 'query')
        query = pipe.create_input('query', ID)
        # look up a user's history in the training data
        history = pipe.add_component('history-lookup', UserTrainingHistoryLookup, query=query)
        # find candidates from the training data
        # Adding the candidate selector for the specific niche genre
        selector_config = MyGenreConfig()
        selector_config.genre_list = self.genre_items
        default_candidates = pipe.add_component('candidate-selector',
            UnratedItemsGenreCandidateSelector, selector_config,
            query=history)
        # score the candidate items using the specified scorer
        score = pipe.add_component('scorer', scorer, query=query, items=default_candidates)
        # rank the items by score
        recommend = pipe.add_component('ranker', TopNRanker, {'n': slate_size}, items=score)
        pipe.alias('recommender', recommend)
        pipe.default_component('recommender')
        return pipe.build()

    def train(self):
        if self.get_dataset().interaction_count >= self.min_interaction_count and \
                self.get_dataset().user_count >= self.min_user_count:
            super().train()

    def isDatasetViable(self):
        if not self.trained:
            return False
        else:
            user_count = self.dataset_active_users()
            interaction_count = self.get_dataset().interaction_count
            if user_count >= self.min_user_count and interaction_count >= self.min_interaction_count:
                # smores.Smores.state.logger.debug(f"Recommender: {self.name} is viable. Interaction count {interaction_count}. User count {user_count}")
                return True
            else:
                return False

        
    def isProfileViable(self, user_id: ID):
        items = self.get_dataset().user_row(user_id)
        if items is None:
            return False
        else:
            profile_size = items.ids().size
            if profile_size < self.min_profile_size:
                return False
            else:
                # smores.Smores.state.logger.debug(f"Recommender: {self.name} user {user_id} is viable.")
                return True


RecommenderFactory.register('genre_implicit_mf', ImplicitMFGenreRecommender)
RecommenderFactory.register('genre_knn', GenreKnnRecommender)