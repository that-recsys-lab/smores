"""Base support for item sampling extensions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

import smores


class ItemSampler(ABC):
    """Abstract helper that generates additional items for recommendation slates."""

    def __init__(self) -> None:
        self.item_ids: list[int] = []
        self.base_probabilities: list[float] = []
        self._item_ids_array = np.empty(0, dtype=np.int64)
        self._base_probabilities_array = np.empty(0, dtype=np.float64)

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

        # Sampling is the hottest path in large simulations. Cache NumPy arrays
        # once when data is loaded instead of rebuilding and filtering Python
        # lists for every recommendation request.
        self._item_ids_array = np.asarray(self.item_ids, dtype=np.int64)
        self._base_probabilities_array = np.asarray(
            self.base_probabilities, dtype=np.float64
        )

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
        if exclude_set:
            excluded = np.fromiter(
                exclude_set,
                dtype=self._item_ids_array.dtype,
                count=len(exclude_set),
            )
            candidate_mask = ~np.isin(self._item_ids_array, excluded)
            candidate_ids = self._item_ids_array[candidate_mask]
            candidate_probs = self._base_probabilities_array[candidate_mask]
        else:
            candidate_ids = self._item_ids_array
            candidate_probs = self._base_probabilities_array

        if candidate_ids.size == 0:
            return []

        limit = min(count, candidate_ids.size)

        prob_sum = candidate_probs.sum()
        if prob_sum <= 0:
            candidate_probs = np.full(
                candidate_ids.size,
                1.0 / candidate_ids.size,
                dtype=np.float64,
            )
        else:
            candidate_probs = candidate_probs / prob_sum

        rng = smores.Smores.state.rand
        chosen_indices = rng.choice(
            candidate_ids.size,
            size=limit,
            replace=False,
            p=candidate_probs,
        )
        return candidate_ids[chosen_indices].tolist()
