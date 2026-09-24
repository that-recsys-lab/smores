"""Read-only worst-case recommendation item-capacity inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smores.item import ItemMap
from smores.samplers.rejection_sampler import RejectionSampler


_MODEL_RECOMMENDERS = {
    "bpr",
    "implicit_mf",
    "item_knn",
    "popular",
    "fixed_recommender",
}
_GENRE_RECOMMENDERS = {"genre_bpr", "genre_implicit_mf", "genre_knn"}


def required_item_capacity(
    *,
    num_days: int,
    num_cycles: int,
    slate_size: int,
    recency_threshold: int = 0,
    cooldown_cycles: int = 0,
) -> dict[str, int]:
    """Return a conservative distinct-item budget for one user and source."""

    requests = max(int(num_days), 0) * max(int(num_cycles), 0)
    prior_clicks = max(requests - 1, 0)
    blocked_items = 0
    if int(recency_threshold) > 0:
        active_cycles = max(int(cooldown_cycles), 0) + 1
        blocked_items = int(slate_size) * max(int(num_days), 0) * active_cycles
    required = int(slate_size) + prior_clicks + blocked_items
    return {
        "requests_per_user": requests,
        "prior_click_budget": prior_clicks,
        "recency_block_budget": blocked_items,
        "required_items": required,
    }


def inspect_item_capacity(config: Any) -> list[dict[str, Any]]:
    """Inspect every configured recommender item source without running a simulation.

    Accepts a ``SmoresConfig`` instance or its mapping representation. The model
    inventory follows SMORES' candidate selectors: all catalog items for normal
    recommenders and feature-matched catalog items for genre recommenders.
    """

    if hasattr(config, "model_dump"):
        config = config.model_dump(mode="python")
    if not isinstance(config, dict):
        raise TypeError("config must be a mapping or SmoresConfig")

    data_cfg = config.get("data") or {}
    simulation = config.get("simulation") or {}
    data_dir = Path(data_cfg.get("directory", ""))
    item_path = data_dir / data_cfg.get("item_file", "")
    if not item_path.is_file():
        raise FileNotFoundError(f"Item catalog not found: {item_path}")

    catalog = ItemMap()
    catalog.load_items(item_path)
    catalog_ids = set(catalog.all_items())
    if not catalog_ids:
        raise ValueError(f"Item catalog is empty: {item_path}")

    rec_cfg = config.get("recommender") or {}
    base_recommenders = rec_cfg.get("base_recommenders") or []
    fallback_recommenders = rec_cfg.get("fallback_recommenders") or []
    by_name = {
        rec.get("name"): rec
        for rec in [*base_recommenders, *fallback_recommenders]
        if rec.get("name")
    }

    rows: list[dict[str, Any]] = []
    for role, recommenders in (
        ("base", base_recommenders),
        ("fallback", fallback_recommenders),
    ):
        for recommender in recommenders:
            name = str(recommender.get("name") or "unnamed")
            class_name = str(recommender.get("class_name") or "")
            params = recommender.get("params") or {}
            budget = required_item_capacity(
                num_days=simulation.get("num_days", 0),
                num_cycles=simulation.get("num_cycles", 0),
                slate_size=simulation.get("slate_size", 0),
                recency_threshold=params.get("recency_threshold", 0),
                cooldown_cycles=params.get("cooldown_cycles", 0),
            )

            model_source = "model_candidates"
            model_path = str(item_path)
            model_items: set[int] = set()
            error = None
            try:
                if class_name in _MODEL_RECOMMENDERS:
                    model_items = catalog_ids
                elif class_name in _GENRE_RECOMMENDERS:
                    feature = int(params["genre_feature"])
                    model_items = set(catalog.get_genre_items(feature))
                    model_source = f"genre_candidates:{feature}"
                elif class_name == "file_based":
                    model_source = "file_recommender_pool"
                    model_path = str(
                        data_dir / (params.get("file_name") or "")
                    )
                    model_items = _sampler_inventory(model_path, catalog_ids)
                else:
                    raise ValueError(f"Unsupported recommender class: {class_name}")
            except (KeyError, TypeError, ValueError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"

            rows.append(
                _capacity_row(
                    role,
                    name,
                    model_source,
                    model_path,
                    model_items,
                    budget,
                    error,
                )
            )

            sampler_cfg = params.get("item_sampler") or {}
            if sampler_cfg:
                sampler_name = sampler_cfg.get("class_name", "rejection_sampler")
                sampler_params = sampler_cfg.get("params") or {}
                sampler_path = str(
                    data_dir / (sampler_params.get("file_name") or "")
                )
                try:
                    if sampler_name != "rejection_sampler":
                        raise ValueError(f"Unsupported sampler class: {sampler_name}")
                    sampler_items = _sampler_inventory(sampler_path, catalog_ids)
                    sampler_error = None
                except (TypeError, ValueError, OSError) as exc:
                    sampler_items = set()
                    sampler_error = f"{type(exc).__name__}: {exc}"
                rows.append(
                    _capacity_row(
                        role,
                        name,
                        "item_sampler",
                        sampler_path,
                        sampler_items,
                        budget,
                        sampler_error,
                    )
                )

            if role == "base":
                for fallback_field in ("cold_start_fallback", "cold_user_fallback"):
                    fallback_name = params.get(fallback_field)
                    if not fallback_name:
                        continue
                    target = by_name.get(fallback_name)
                    if target is None:
                        rows.append(
                            {
                                "recommender_role": role,
                                "recommender": name,
                                "source": fallback_field,
                                "source_recommender": fallback_name,
                                "source_path": "",
                                "available_items": 0,
                                **budget,
                                "deficit": budget["required_items"],
                                "passed": False,
                                "error": f"Unknown fallback recommender: {fallback_name}",
                            }
                        )

    return rows


def _sampler_inventory(path: str, catalog_ids: set[int]) -> set[int]:
    item_ids, _, _ = RejectionSampler.read_eligible_items(path, catalog_ids)
    return set(item_ids)


def _capacity_row(
    role: str,
    name: str,
    source: str,
    path: str,
    item_ids: set[int],
    budget: dict[str, int],
    error: str | None,
) -> dict[str, Any]:
    available = len(item_ids)
    required = budget["required_items"]
    return {
        "recommender_role": role,
        "recommender": name,
        "source": source,
        "source_recommender": name,
        "source_path": path,
        "available_items": available,
        **budget,
        "deficit": max(required - available, 0),
        "passed": error is None and available >= required,
        "error": error or (None if available >= required else "insufficient_item_capacity"),
    }
