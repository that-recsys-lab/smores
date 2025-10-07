"""Lightweight runtime summary reporting for experiments."""

from __future__ import annotations

import time
from collections import Counter, defaultdict


class SummaryLogger:
    """Aggregates basic experiment metrics for a final console summary."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._start_time: float | None = None
        self._recommendation_counts: Counter[str] = Counter()
        self._recommendation_slots: Counter[str] = Counter()
        self._unique_users = defaultdict(set)

    def start_timer(self) -> None:
        if not self.enabled:
            return
        self._start_time = time.time()

    def log_recommendation(self, recommender_name: str, user_id: int, delivered: int, slate_size: int) -> None:
        if not self.enabled:
            return
        self._recommendation_counts[recommender_name] += delivered
        self._recommendation_slots[recommender_name] += slate_size
        self._unique_users[recommender_name].add(user_id)

    def print_summary(self) -> None:
        if not self.enabled:
            return

        duration = None
        if self._start_time is not None:
            duration = time.time() - self._start_time

        if duration is not None:
            print(f"Simulation runtime: {duration:.2f}s")

        if not self._recommendation_counts:
            print("No recommendations logged.")
            return

        print("Recommender summary:")
        for name in sorted(self._recommendation_counts.keys()):
            total = self._recommendation_counts[name]
            slots = self._recommendation_slots[name]
            user_count = len(self._unique_users[name])
            avg_fill = total / slots if slots else 0.0
            print(
                f"  - {name}: delivered={total}, users={user_count}, avg_fill={avg_fill:.2f}"
            )

