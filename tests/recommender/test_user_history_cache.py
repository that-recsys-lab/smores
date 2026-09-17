import yaml

from smores import Smores
from tests.paths import FIXTURE_DATA_DIR, TEST_CONFIG_PATH
from smores.utils import SmoresConfig


def test_user_history_cache_tracks_updates_deletes_and_fallback_parent():
    config = SmoresConfig.model_validate(yaml.safe_load(TEST_CONFIG_PATH.read_text()))
    simulation = Smores(config)
    simulation.setup()

    recommender = simulation.state.recommenders_base.get_recommender('Generic')
    fallback = simulation.state.recommenders_fallback.get_recommender('Popular Fallback')
    interactions = [
        [100, 200, 1, 1],
        [100, 201, 1, 2],
        [101, 202, 1, 2],
    ]

    recommender.update_dataset(interactions)

    expected = frozenset(int(item_id) for item_id in recommender.get_user(100).ids())
    assert recommender.get_user_item_ids(100) == expected
    assert fallback.get_user_item_ids(100) == expected

    recommender.delete_user(100)

    assert recommender.get_user_item_ids(100) == frozenset()
    assert fallback.get_user_item_ids(100) == frozenset()


def test_empty_history_cache_matches_empty_dataset():
    config = SmoresConfig.model_validate(yaml.safe_load(TEST_CONFIG_PATH.read_text()))
    simulation = Smores(config)
    simulation.setup()

    recommender = simulation.state.recommenders_base.get_recommender('Generic')

    assert recommender.get_user_item_ids(100) == frozenset()
