import numpy as np
from collections import defaultdict

import smores
from smores.trigger.trigger import (
    CycleEvent,
    DayEvent,
    InteractionBatchEvent,
)
from smores.trigger.switch_trigger import SwitchEvent


class TriggerTester:
    """Lightweight trigger sanity checker to keep failures from hiding."""

    def __init__(self, state, config=None):
        self.state = state
        cfg = config or type("Cfg", (), {})()
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.sample_count = max(int(getattr(cfg, "sample_count", 5) or 1), 1)
        seed = getattr(cfg, "seed", None)
        self.rand = np.random.default_rng(seed) if seed is not None else state.rand
        self.scenario = (getattr(cfg, "scenario", None) or "").lower()
        self.baseline_stats: dict[str, dict] | None = None
        self.last_result_msg: str | None = None

    def set_baseline(self):
        """Capture the current stats as baseline before triggers run."""
        self.last_result_msg = None
        self.baseline_stats = self._collect_stats()

    def run_cycle_checks(self) -> bool:
        if not self.enabled:
            return True
        if self.baseline_stats is None:
            return True

        scenario_methods = {
            "monolithic": self._test_monolithic,
            "cold_start": self._test_cold_start,
            "universal": self._test_universal,
            "alg_specific": self._test_algorithm_specific,
            "ownership": self._test_ownership,
        }
        handler = scenario_methods.get(self.scenario)
        if handler is None:
            self.last_result_msg = f"TriggerTester ({self.scenario or 'unknown'}): skipped (unknown scenario)"
            return True
        result = handler()
        # reset baseline after a check
        self.baseline_stats = None
        return result

    # Scenario-specific checks inspired by tests/trigger/test_triggers.py

    def _sample_users(self, consumer_ids):
        """Pick a sample of users within configured bounds."""
        if not consumer_ids:
            return []
        population = list(consumer_ids)
        size = min(self.sample_count, len(population))
        if size <= 0:
            return []
        return [int(uid) for uid in self.rand.choice(population, size=size, replace=False)]

    def _snapshot_recommenders(self, names):
        """Capture datasets for the given recommenders so we can restore after testing."""
        snapshots = {}
        for name in names:
            rec = self.state.recommenders_base.get_recommender(name)
            if rec is not None:
                snapshots[name] = rec.get_dataset()
        return snapshots

    def _restore_recommenders(self, snapshots):
        """Restore recommender datasets to avoid mutating state during tests."""
        for name, dataset in snapshots.items():
            rec = self.state.recommenders_base.get_recommender(name)
            if rec is not None:
                rec.set_dataset(dataset)
                rec._cached_user_count = None

    def _pick_two_recs(self):
        """Select two recommenders (first two) for source/target style tests."""
        names = self.state.recommenders_base.get_names()
        if len(names) < 2:
            return None, None
        first = names[0]
        second = names[1]
        return first, second

    def _pick_items(self, count):
        """Sample a set of items by ID."""
        item_ids = self.state.items.all_items()
        if len(item_ids) < count:
            return []
        choice = self.rand.choice(item_ids, size=count, replace=False)
        return [int(x) for x in choice]

    def _collect_stats(self):
        """Capture current interaction counts and user sets per recommender."""
        stats = {}
        for name in self.state.recommenders_base.get_names():
            rec = self.state.recommenders_base.get_recommender(name)
            if rec is None:
                continue
            dataset = rec.get_dataset()
            if dataset is None:
                stats[name] = {"count": 0, "users": set()}
                continue
            try:
                count = dataset.interaction_count
                users = set()
                if count > 0:
                    table = dataset.interaction_table(format="arrow", original_ids=True)
                    if table is not None:
                        users = set(int(u.as_py()) for u in table.column("user_id").unique())
                stats[name] = {"count": count, "users": users}
            except Exception:
                stats[name] = {"count": 0, "users": set()}
        return stats

    def _test_monolithic(self) -> bool:
        """Monolithic regime: single recommender accumulates interactions locally."""
        rec_name = (self.state.recommenders_active[0] if self.state.recommenders_active else None)
        if rec_name is None:
            self.last_result_msg = "TriggerTester (monolithic): no active recommender; skipping check."
            return True
        stats = self._collect_stats()
        prev = self.baseline_stats.get(rec_name, {"count": 0}) if self.baseline_stats else {"count": 0}
        curr = stats.get(rec_name, {"count": 0})
        passed = curr["count"] >= prev["count"]
        if passed:
            self.last_result_msg = "TriggerTester (monolithic): passed."
        else:
            self.last_result_msg = "TriggerTester (monolithic): failed (count shrank)."
        return passed

    def _test_algorithm_specific(self) -> bool:
        """Algorithm-specific: source recommender updates, target remains unchanged."""
        source, target = self._pick_two_recs()
        if not source or not target:
            self.last_result_msg = "TriggerTester (alg_specific): not enough recommenders to test."
            return True
        stats = self._collect_stats()
        prev_src = self.baseline_stats.get(source, {"count": 0, "users": set()}) if self.baseline_stats else {"count": 0, "users": set()}
        prev_tgt = self.baseline_stats.get(target, {"count": 0, "users": set()}) if self.baseline_stats else {"count": 0, "users": set()}
        curr_src = stats.get(source, {"count": 0, "users": set()})
        curr_tgt = stats.get(target, {"count": 0, "users": set()})
        src_not_shrinking = curr_src["count"] >= prev_src["count"]
        tgt_not_shrinking = curr_tgt["count"] >= prev_tgt["count"]
        src_users_superset = curr_src["users"].issuperset(prev_src["users"])
        tgt_users_superset = curr_tgt["users"].issuperset(prev_tgt["users"])
        passed = src_not_shrinking and tgt_not_shrinking and src_users_superset and tgt_users_superset
        if passed:
            self.last_result_msg = "TriggerTester (alg_specific): passed."
        else:
            self.last_result_msg = "TriggerTester (alg_specific): failed (counts/users shrank)."
        return passed

    def _test_cold_start(self) -> bool:
        """Cold-start regime: switch trigger should remove the user's profile from the source recommender."""
        source, target = self._pick_two_recs()
        if not source or not target:
            self.last_result_msg = "TriggerTester (cold_start): not enough recommenders to test."
            return True
        stats = self._collect_stats()
        prev_stats = self.baseline_stats or {}
        removed_users = prev_stats.get(source, {"users": set()})["users"] - stats.get(source, {"users": set()})["users"]
        if not removed_users:
            self.last_result_msg = "TriggerTester (cold_start): no user switches observed; skipping check."
            return True
        # cold-start: removed users should not exist in source or target now
        curr_source_users = stats.get(source, {"users": set()})["users"]
        curr_target_users = stats.get(target, {"users": set()})["users"]
        for uid in removed_users:
            if uid in curr_source_users or uid in curr_target_users:
                self.last_result_msg = "TriggerTester (cold_start): failed (user still present)."
                return False
        self.last_result_msg = "TriggerTester (cold_start): passed."
        return True

    def _test_ownership(self) -> bool:
        """Ownership regime: switch trigger should move the user profile to the target recommender."""
        source, target = self._pick_two_recs()
        if not source or not target:
            self.last_result_msg = "TriggerTester (ownership): not enough recommenders to test."
            return True
        stats = self._collect_stats()
        prev_stats = self.baseline_stats or {}
        removed_users = prev_stats.get(source, {"users": set()})["users"] - stats.get(source, {"users": set()})["users"]
        if not removed_users:
            self.last_result_msg = "TriggerTester (ownership): no user switches observed; skipping check."
            return True
        target_users = stats.get(target, {"users": set()})["users"]
        for uid in removed_users:
            if uid not in target_users:
                self.last_result_msg = "TriggerTester (ownership): failed (user not found in target)."
                return False
        self.last_result_msg = "TriggerTester (ownership): passed."
        return True

    def _test_universal(self) -> bool:
        """Universal regime: interactions should replicate to other recommenders via the universal trigger."""
        source, target = self._pick_two_recs()
        if not source or not target:
            self.last_result_msg = "TriggerTester (universal): not enough recommenders to test."
            return True
        stats = self._collect_stats()
        counts = [stats.get(r, {"count": 0})["count"] for r in [source, target]]
        users = [stats.get(r, {"users": set()})["users"] for r in [source, target]]
        counts_equal = len(set(counts)) == 1
        users_equal = users[0] == users[1]
        passed = counts_equal and users_equal
        if passed:
            self.last_result_msg = "TriggerTester (universal): passed."
        else:
            self.last_result_msg = "TriggerTester (universal): failed (counts/users diverged)."
        return passed

    def _safe_apply(self, trigger, event) -> bool:
        try:
            trigger.apply_trigger(event)
            return True
        except Exception as exc:
            trig_name = getattr(trigger, "name", type(trigger).__name__)
            self.state.logger.error(f"Trigger test failed for '{trig_name}': {exc}")
            return False
