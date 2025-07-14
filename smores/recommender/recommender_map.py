from smores.recommender import Recommender, RecommenderFactory

class RecommenderMap:
    def __init__(self):
        self._rec_map: dict[str, Recommender] = {}

    # Assumes input comes from config.recommender.defintions (a list)
    # A little bit odd because usually setup doesn't create the objects
    def setup(self, config):
        for rec_config in config:
            name = rec_config.name
            class_name = rec_config.class_name

            inst = RecommenderFactory.create(class_name)
            self.set_recommender(name, inst)

            inst.setup(rec_config)

    def setup_datasets(self):
        for _, rec in self._rec_map.items():
            rec.setup_dataset()

    def setup_fallbacks(self):
        for _, rec in self._rec_map.items():
            cold_start_rec = rec.get_cold_start_fallback()
            cold_user_rec = rec.get_cold_user_fallback()
            if cold_start_rec is not None:
                cold_start_rec.parent = rec
            if cold_user_rec is not None:
                cold_user_rec.parent = rec

    def is_recommender(self, name: str):
        return name in self._rec_map

    def get_recommender(self, name: str):
        return self._rec_map[name]
    
    def set_recommender(self, name: str, rec: Recommender):
        self._rec_map[name] = rec
        rec.name = name

    def clear(self):
        self._rec_map.clear()

    def del_recommender(self, name: str):
        del self._rec_map[name]

    # Sorts the recommenders by key so that the order is repeatable
    # Necessary for logging
    def get_names(self):
        keys = self._rec_map.keys()
        sorted_keys = sorted(keys)
        return sorted_keys

    def items(self):
        return [self._rec_map[key] for key in self.get_names()]


class UnknownRecommenderError(Exception):
    def __init__(self, name):
        self.message = f'Recommender {name} is not part of this collection.'
        super().__init__(self.message) 