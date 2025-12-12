"""Base support for item sampling extensions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import smores


class ItemSampler(ABC):
    """Abstract helper that generates additional items for recommendation slates."""

    def __init__(self) -> None:
        self.item_ids: list[int] = []
        self.base_probabilities: list[float] = []

    @abstractmethod
    def load_from_file(self, path):
        """Load sampling data from a file. Implemented by subclasses."""

    def has_items(self) -> bool:
        return bool(self.item_ids)

    def _normalize_probabilities(self) -> None:
        total = sum(self.base_probabilities)
        if total <= 0:
            raise ValueError("ItemSampler requires strictly positive probability mass")
        self.base_probabilities = [prob / total for prob in self.base_probabilities]

    def _filtered_items(self, exclude_items: set[int]) -> tuple[list[int], list[float]]:
        if not self.item_ids:
            return [], []
        filtered_ids: list[int] = []
        filtered_probs: list[float] = []
        for item_id, prob in zip(self.item_ids, self.base_probabilities):
            if item_id not in exclude_items:
                filtered_ids.append(item_id)
                filtered_probs.append(prob)
        return filtered_ids, filtered_probs

    def sample(self, count: int, *, exclude_items: Iterable[int] | None = None) -> list[int]:
        """Sample up to *count* items, avoiding the provided exclusions."""

        if count <= 0 or not self.has_items():
            return []

        exclude_set = set(exclude_items or [])
        candidate_ids, candidate_probs = self._filtered_items(exclude_set)
        if not candidate_ids:
            return []

        limit = min(count, len(candidate_ids))

        prob_sum = sum(candidate_probs)
        if prob_sum <= 0:
            candidate_probs = [1.0 / len(candidate_ids)] * len(candidate_ids)
        else:
            candidate_probs = [p / prob_sum for p in candidate_probs]

        rng = smores.Smores.state.rand
        chosen_indices = rng.choice(len(candidate_ids), size=limit, replace=False, p=candidate_probs)
        return [candidate_ids[idx] for idx in chosen_indices]
