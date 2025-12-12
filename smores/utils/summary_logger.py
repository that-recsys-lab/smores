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
        self._active_users = defaultdict(int)
        self._per_recommender = defaultdict(
            lambda: {
                'users': set(),
                'items': set(),
                'events': 0,
                'slots': 0,
                'delivered': 0,
                'sampled': 0,
                'fallback': 0,
                'click_events': 0,
                'clicks': 0,
            }
        )
        self._catalog_size: int | None = None

    def start_timer(self) -> None:
        if not self.enabled:
            return
        self._start_time = time.time()

    def set_catalog_size(self, catalog_size: int) -> None:
        if not self.enabled:
            return
        self._catalog_size = catalog_size

    def set_active_users(self, recommender_name: str, count: int) -> None:
        if not self.enabled:
            return
        self._active_users[recommender_name] = max(int(count), 0)

    def log_recommendation(
        self,
        recommender_name: str,
        user_id: int,
        slate_items: list[int],
        slate_size: int,
        *,
        sampled_count: int = 0,
        used_fallback: bool = False,
    ) -> None:
        if not self.enabled:
            return

        delivered = len(slate_items)
        bucket = self._per_recommender[recommender_name]

        bucket['users'].add(user_id)
        bucket['items'].update(int(item_id) for item_id in slate_items)
        bucket['events'] += 1
        bucket['slots'] += slate_size
        bucket['delivered'] += delivered
        bucket['sampled'] += max(sampled_count, 0)
        if used_fallback:
            bucket['fallback'] += 1

        self._recommendation_counts[recommender_name] += delivered
        self._recommendation_slots[recommender_name] += slate_size
        self._unique_users[recommender_name].add(user_id)

    def log_click(self, recommender_name: str, clicked: bool) -> None:
        if not self.enabled:
            return
        bucket = self._per_recommender[recommender_name]
        bucket['click_events'] += 1
        if clicked:
            bucket['clicks'] += 1

    def print_summary(self) -> None:
        if not self.enabled:
            return

        duration = None
        if self._start_time is not None:
            duration = time.time() - self._start_time

        if duration is not None:
            print(f"Simulation runtime: {duration:.2f}s")

        if not self._recommendation_counts:
            print('No recommendations logged.')
            return

        for name in sorted(self._per_recommender.keys()):
            bucket = self._per_recommender[name]
            if bucket['events'] == 0:
                continue

            delivered = bucket['delivered']
            slots = bucket['slots']
            unique_users = len(bucket['users'])
            avg_fill = delivered / slots if slots else 0.0

            active_users = self._active_users.get(name, unique_users)
            coverage_count = len(bucket['items'])
            coverage_pct = None
            if self._catalog_size and self._catalog_size > 0:
                coverage_pct = 100.0 * coverage_count / self._catalog_size

            avg_slate = delivered / bucket['events'] if bucket['events'] else 0.0
            if bucket['sampled']:
                sampled_ratio = delivered / bucket['sampled']
                sampled_ratio = min(sampled_ratio, 1.0)
            else:
                sampled_ratio = 0.0
            ctr = bucket['clicks'] / bucket['click_events'] if bucket['click_events'] else 0.0
            fallback_share = bucket['fallback'] / bucket['events'] if bucket['events'] else 0.0

            print(f"Recommender: {name}")
            print(f"  delivered={delivered}, unique_users={unique_users}, avg_fill={avg_fill:.2f}")
            if coverage_pct is not None:
                print(f"  active_users={active_users}, coverage={coverage_count} ({coverage_pct:.1f}%)")
            else:
                print(f"  active_users={active_users}, coverage={coverage_count}")
            print(
                f"  avg_slate={avg_slate:.2f}, sampled_ratio={sampled_ratio:.2f}, CTR={ctr:.2f}, fallback_share={fallback_share:.2f}"
            )
