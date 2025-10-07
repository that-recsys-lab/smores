from __future__ import annotations

from lenskit.data.items import ItemList

import smores
from smores.recommender import Recommender, RecommenderFactory
from smores.samplers.rejection_sampler import RejectionSampler


class FileBasedRecommender(Recommender):
    """Recommender that loads items from a file and samples based on popularity."""

    def __init__(self) -> None:
        super().__init__()
        # Needs no training
        self.trained = True

    def setup(self, config):
        super().setup(config)

        if self.item_sampler is None:
            params = config.params or {}
            file_name = params.get('file_name')
            if file_name is None:
                raise ValueError("FileBasedRecommender requires 'file_name' in params")
            file_path = smores.Smores.state.data_directory / file_name
            sampler = RejectionSampler()
            sampler.load_from_file(file_path)
            self.item_sampler = sampler

    def train(self):
        pass

    def isDatasetViable(self):
        return self.item_sampler is not None and self.item_sampler.has_items()

    def isProfileViable(self, user_id):
        return True

    def get_recommendations(self, user_id) -> ItemList:
        prior_interactions = self.get_user(user_id)
        interacted_items = set()
        if prior_interactions is not None:
            interacted_items.update(prior_interactions.ids())

        slate_size = smores.Smores.state.slate_size
        sampled_items = self.item_sampler.sample(slate_size, exclude_items=interacted_items)

        scores = [1.0] * len(sampled_items)
        ranks = list(range(1, len(sampled_items) + 1))

        return ItemList(None, item_ids=sampled_items, scores=scores, rank=ranks)


RecommenderFactory.register('file_based', FileBasedRecommender)
