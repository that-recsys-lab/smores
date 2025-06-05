from smores.recommender import Recommender, RecommenderFactory

class RecommenderMap:
    def __init__(self):
        self._rec_map = {}

    # Assumes input comes from config.recommender.defintions (a list)
    # A little bit odd because usually setup doesn't create the objects
    def setup(self, config):
        for rec_config in config:
            name = rec_config.name
            class_name = rec_config.class_name

            inst = RecommenderFactory.create(class_name)
            self.set_recommender(name, inst)

            inst.setup(rec_config)

    def is_recommender(self, name: str):
        return name in self._rec_map

    def get_recommender(self, name: str):
        return self._rec_map[name]
    
    def set_recommender(self, name: str, rec: Recommender):
        self._rec_map[name] = rec

    def clear(self):
        self._rec_map.clear()

    def del_recommender(self, name: str):
        del self._rec_map[name]

    def items(self):
        return self._rec_map.items()
    