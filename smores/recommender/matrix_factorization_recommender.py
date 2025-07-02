from .recommender import LKRecommender, RecommenderFactory
from lenskit.algorithms.als import BiasedMF  

class MatrixFactorizationRecommender(LKRecommender):
    def setup(self, config):
        params = config.params
        features = int(params.get('features', 20))
        self.lk_config = BiasedMF(features=features)
        self.scorer = self.lk_config
        super().setup(config)

# Register with factory
RecommenderFactory.register('matrix_factorization', MatrixFactorizationRecommender)