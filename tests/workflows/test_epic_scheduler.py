"""Regression checks for canonical epic discovery and deterministic wave planning."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLANNER = load("plan_wave", ROOT / "skills/codex-epic-scheduler/scripts/plan_wave.py")
DISCOVERY = load("find_epic_parent", ROOT / ".github/scripts/find_epic_parent.py")


def snapshot(states=None):
    states = states or {}
    return {
        "parent_state": "in-progress",
        "execution_mode": "epic-dag",
        "max_parallel_workers": 2,
        "children": [
            {"issue": 10, "depends_on": [], "mutex": ["core"], "state": states.get(10, "queued")},
            {"issue": 11, "depends_on": [], "mutex": ["core"], "state": states.get(11, "queued")},
            {"issue": 12, "depends_on": [10], "mutex": [], "state": states.get(12, "queued")},
        ],
    }


def dag_comment(children=(10, 11, 12)):
    rows = "\n".join(
        "  - issue: {}\n    depends_on: []\n    mutex: []".format(issue) for issue in children
    )
    return {
        "body": """<!-- codex-epic-dag:v1 -->\n## Canonical epic DAG\n\n```yaml\nexecution_mode: epic-dag\nintegration_branch: codex/epic-issue-3\nmax_parallel_workers: 2\nchildren:\n{}\n```\n""".format(rows)
    }


class PlannerTests(unittest.TestCase):
    def test_selects_deterministic_mutex_compatible_wave(self):
        result = PLANNER.plan_wave(snapshot())
        self.assertEqual(result["candidates"], [10, 11])
        self.assertEqual(result["selected"], [10])
        self.assertEqual(result["available_slots"], 2)

    def test_completed_dependency_unlocks_next_child(self):
        result = PLANNER.plan_wave(snapshot({10: "completed"}))
        self.assertEqual(result["candidates"], [11, 12])
        self.assertEqual(result["selected"], [11, 12])

    def test_active_child_consumes_slot_and_mutex(self):
        result = PLANNER.plan_wave(snapshot({10: "in-progress"}))
        self.assertEqual(result["available_slots"], 1)
        self.assertEqual(result["selected"], [])

    def test_cycle_and_undeclared_dependency_fail_closed(self):
        cyclic = snapshot()
        cyclic["children"][0]["depends_on"] = [12]
        with self.assertRaises(PLANNER.ContractError):
            PLANNER.plan_wave(cyclic)
        missing = snapshot()
        missing["children"][2]["depends_on"] = [99]
        with self.assertRaises(PLANNER.ContractError):
            PLANNER.plan_wave(missing)

    def test_repeat_application_is_idempotent_for_selected_child(self):
        initial = snapshot()
        first = PLANNER.plan_wave(initial)
        updated = PLANNER.apply_wave(initial, first["selected"])
        second = PLANNER.plan_wave(updated)
        self.assertNotIn(10, second["selected"])


class ParentDiscoveryTests(unittest.TestCase):
    def issue(self, number, labels=("in-progress",)):
        return {"number": number, "title": "Epic {}".format(number), "state": "open", "labels": [{"name": x} for x in labels]}

    def test_finds_unique_active_parent_from_canonical_comment(self):
        issues = [self.issue(3), self.issue(4)]
        comments = {3: [dag_comment()], 4: [dag_comment((20, 21))]}
        self.assertEqual(DISCOVERY.find_parent(11, issues, comments), {"number": 3, "title": "Epic 3"})

    def test_uninitialized_parent_is_not_discovered(self):
        self.assertIsNone(DISCOVERY.find_parent(11, [self.issue(3)], {3: []}))

    def test_duplicate_canonical_comments_fail_closed(self):
        with self.assertRaises(DISCOVERY.DiscoveryError):
            DISCOVERY.find_parent(11, [self.issue(3)], {3: [dag_comment(), dag_comment()]})

    def test_multiple_active_parents_fail_closed(self):
        issues = [self.issue(3), self.issue(4)]
        comments = {3: [dag_comment()], 4: [dag_comment()]}
        with self.assertRaises(DISCOVERY.DiscoveryError):
            DISCOVERY.find_parent(11, issues, comments)

    def test_non_active_parent_is_ignored(self):
        issues = [self.issue(3, ("execution-ready",))]
        self.assertIsNone(DISCOVERY.find_parent(11, issues, {3: [dag_comment()]}))


if __name__ == "__main__":
    unittest.main()
