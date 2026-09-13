"""Static regression checks for the single label-routing workflow."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
DISPATCH = (ROOT / ".github/workflows/codex-issue-state.yml").read_text(encoding="utf-8")
EXECUTE = (ROOT / ".github/workflows/codex-execute-ready.yml").read_text(encoding="utf-8")
REVIEW = (ROOT / ".github/workflows/codex-review-ready.yml").read_text(encoding="utf-8")


class DispatcherTests(unittest.TestCase):
    def test_only_dispatcher_has_issue_label_trigger(self):
        self.assertIn("issues:\n    types: [labeled]", DISPATCH)
        self.assertNotIn("types: [labeled]", EXECUTE)
        self.assertNotIn("types: [labeled]", REVIEW)
        self.assertIn("workflow_call:", EXECUTE)
        self.assertIn("workflow_call:", REVIEW)

    def test_routes_all_automatic_workflow_states(self):
        for label in ("execution-ready", "review-ready", "completed|queued"):
            self.assertIn(label, DISPATCH)
        self.assertIn("find_epic_parent.py", DISPATCH)
        self.assertIn("wait_for_existing_turn=true", DISPATCH)

    def test_domain_specific_source_names_are_absent(self):
        for root in (ROOT / ".github", ROOT / "skills"):
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8").lower()
                except UnicodeDecodeError:
                    continue
                self.assertNotIn("bot bowl", text, str(path))
                self.assertNotIn("botbowl", text, str(path))


if __name__ == "__main__":
    unittest.main()
