from smores.recommender.capacity import inspect_item_capacity, required_item_capacity
from tests.paths import FIXTURE_DATA_DIR


def test_required_capacity_includes_click_and_recency_budgets():
    result = required_item_capacity(
        num_days=3,
        num_cycles=10,
        slate_size=10,
        recency_threshold=3,
        cooldown_cycles=1,
    )

    assert result == {
        "requests_per_user": 30,
        "prior_click_budget": 29,
        "recency_block_budget": 60,
        "required_items": 99,
    }


def test_required_capacity_boundary_and_no_recency():
    result = required_item_capacity(
        num_days=1,
        num_cycles=1,
        slate_size=5,
    )

    assert result["required_items"] == 5
    assert result["recency_block_budget"] == 0


def test_inventory_uses_genre_pool_and_sampler_runtime_filters(tmp_path):
    item_path = tmp_path / "items.csv"
    item_path.write_text(FIXTURE_DATA_DIR.joinpath("items.csv").read_text())
    sampler_path = tmp_path / "sample.csv"
    sampler_path.write_text(
        "item_id,popularity\n"
        "200,10\n"
        "200,3\n"
        "999,8\n"
        "201,0\n"
        "202,-2\n"
        "invalid,2\n"
        "204,1\n"
    )
    config = {
        "data": {"directory": str(tmp_path), "item_file": "items.csv"},
        "simulation": {"num_days": 1, "num_cycles": 1, "slate_size": 5},
        "recommender": {
            "base_recommenders": [
                {
                    "name": "Thriller",
                    "class_name": "genre_bpr",
                    "params": {"genre_feature": 3},
                },
                {
                    "name": "Generic",
                    "class_name": "bpr",
                    "params": {
                        "item_sampler": {
                            "class_name": "rejection_sampler",
                            "params": {"file_name": "sample.csv"},
                        }
                    },
                },
            ],
            "fallback_recommenders": [
                {
                    "name": "Popular File",
                    "class_name": "file_based",
                    "params": {"file_name": "sample.csv"},
                }
            ],
        },
    }

    rows = inspect_item_capacity(config)
    by_key = {(row["recommender"], row["source"]): row for row in rows}

    assert by_key[("Thriller", "genre_candidates:3")]["available_items"] == 2
    assert by_key[("Generic", "item_sampler")]["available_items"] == 2
    assert by_key[("Popular File", "file_recommender_pool")]["available_items"] == 2
    assert by_key[("Generic", "model_candidates")]["required_items"] == 5
    assert by_key[("Generic", "model_candidates")]["available_items"] == 5
    assert by_key[("Generic", "model_candidates")]["passed"] is True


def test_sampler_inventory_rejects_empty_or_missing_source(tmp_path):
    item_path = tmp_path / "items.csv"
    item_path.write_text(FIXTURE_DATA_DIR.joinpath("items.csv").read_text())
    config = {
        "data": {"directory": str(tmp_path), "item_file": "items.csv"},
        "simulation": {"num_days": 1, "num_cycles": 1, "slate_size": 6},
        "recommender": {
            "base_recommenders": [
                {
                    "name": "Generic",
                    "class_name": "bpr",
                    "params": {
                        "item_sampler": {
                            "class_name": "rejection_sampler",
                            "params": {"file_name": "missing.csv"},
                        }
                    },
                }
            ],
            "fallback_recommenders": [],
        },
    }

    rows = inspect_item_capacity(config)
    sampler_row = next(row for row in rows if row["source"] == "item_sampler")

    assert sampler_row["passed"] is False
    assert sampler_row["error"].startswith("FileNotFoundError:")
