"""Offline contract tests for profile enforcement in both real launcher bodies.

Run: python3 -m unittest discover -s .github/scripts -p 'test_codex_profile.py'
No App Server connection, model turn, GitHub mutation or product suite is used.
"""

import ast
import json
import os
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from codex_profile import CAP_KEY, ProfilePolicy
from prepare_pr_audit import build_audit_client


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/codex-execute-ready.yml").read_text()
BEGIN = '          cat > "$client" <<\'PY\'\n'
EXECUTOR = textwrap.dedent(WORKFLOW.split(BEGIN, 1)[1].split('\n          PY\n', 1)[0])


class Server:
    def __init__(self, home):
        self.home = home
        self.calls = []
        self.saved = {"model": "gpt-6-astra", "reasoningEffort": "ultra"}
        self.status = "idle"
        self.resumed_status = "idle"
        self.start_override = {}
        self.pages = [{"data": [{"model": "gpt-6-astra", "supportedReasoningEfforts": [
            {"reasoningEffort": effort} for effort in ("xhigh", "max", "ultra")
        ]}], "nextCursor": None}]

    def send_notification(self, *args):
        pass

    def request(self, method, params):
        self.calls.append((method, params))
        if method == "initialize":
            return {"codexHome": str(self.home), "platformOs": "linux"}
        if method == "model/list":
            return self.pages[0 if params["cursor"] is None else int(params["cursor"])]
        if method == "thread/read":
            return {"thread": {"status": self.status}}
        if method == "thread/resume":
            return {"thread": {"id": "saved", "status": self.resumed_status}, **self.saved}
        if method == "thread/start":
            return {"thread": {"id": "new", "status": "idle"}, "model": params["model"],
                    "reasoningEffort": params["config"]["model_reasoning_effort"], **self.start_override}
        if method == "thread/name/set":
            return {}
        if method == "turn/start":
            return {"turn": {"id": "turn"}}
        raise AssertionError(f"Unexpected request: {method}")

    def drain_until_turn_complete(self):
        return {"id": "turn", "status": "completed"}


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.server = Server(self.home)
        self.logs = []
        self.profile("codex-critical", "ultra")
        self.profile("codex-general", "xhigh")
        self.profile("codex-review", "ultra")

    def profile(self, name, effort, model="gpt-6-astra", extra="", cap=3):
        path = self.home / f"{name}.config.toml"
        path.write_text(f'model = "{model}"\nmodel_reasoning_effort = "{effort}"\n{extra}'
                        f'[agents]\nmax_concurrent_threads_per_session = {cap}\n')
        return path

    def launch(self, *, role="execution", profile="codex-critical", existing=False):
        thread_file = self.home / "thread-id"
        if existing:
            thread_file.write_text("saved")
        source = build_audit_client(WORKFLOW) if role == "audit" else EXECUTOR
        tree = ast.parse(source)
        body = ast.Module(body=[tree.body[-1]], type_ignores=[])
        env = {"CODEX_PROFILE_NAME": profile, "CODEX_LAUNCH_ROLE": role,
               "GITHUB_REPOSITORY": "test/repo", "PR_NUMBER": "7", "GITHUB_RUN_ATTEMPT": "1",
               "REVIEW_HEAD_SHA": "a" * 40, "SKILLFORGE_TASK_PROMPT": "Review the fixed target"}
        namespace = {
            "os": os, "json": json, "ProfilePolicy": ProfilePolicy,
            "traceback": SimpleNamespace(print_exc=lambda: None),
            "WebSocketUnix": lambda path: None, "AppServerClient": lambda ws: self.server,
            "SOCKET_PATH": "unused", "WORKTREE": "/isolated/issue-worktree", "ISSUE_NUMBER": "1",
            "ISSUE_TITLE": "Test", "RUN_ID": "test", "THREAD_FILE": str(thread_file),
            "log": lambda event, **fields: self.logs.append({"event": event, **fields}),
            "write_atomic": lambda path, value: Path(path).write_text(value),
            "status_is_active": lambda status: status == "active" or status == {"type": "active"},
        }
        for key in ("ERROR_FILE", "TURN_FILE", "READY_FILE", "COMPLETED_FILE"):
            namespace[key] = str(self.home / key)
        with patch.dict(os.environ, env):
            exec(compile(body, "launcher-lifecycle", "exec"), namespace)

    def methods(self):
        return [method for method, _ in self.server.calls]

    def params(self, method):
        return next(params for name, params in self.server.calls if name == method)

    def test_critical_start_confirms_runtime_and_preserves_permissions(self):
        self.launch()
        self.assertEqual(self.params("thread/start")["config"],
                         {CAP_KEY: 3, "model_reasoning_effort": "ultra"})
        self.assertIs(self.params("thread/start")["allowProviderModelFallback"], False)
        turn = self.params("turn/start")
        self.assertEqual((turn["model"], turn["effort"]), ("gpt-6-astra", "ultra"))
        self.assertEqual(turn["approvalPolicy"], "on-request")
        self.assertEqual(turn["approvalsReviewer"], "auto_review")
        self.assertEqual(turn["sandboxPolicy"], {"type": "workspaceWrite",
                         "writableRoots": ["/isolated/issue-worktree"], "networkAccess": True})
        confirmation = next(log for log in self.logs if log["event"] == "model_policy_confirmed")
        self.assertEqual(confirmation["source"], str(self.home / "codex-critical.config.toml"))
        self.assertEqual(confirmation["confirmedEffort"], "ultra")
        self.assertEqual(confirmation["requestedEffort"], "ultra")
        self.assertEqual(confirmation["role"], "execution")
        self.assertNotIn("config", confirmation)

    def test_inactive_resume_preserves_explicit_higher_selection(self):
        self.launch(profile="codex-general", existing=True)
        resume = self.params("thread/resume")
        self.assertNotIn("model", resume)
        self.assertEqual(resume["config"], {CAP_KEY: 3})
        self.assertEqual(self.params("turn/start")["effort"], "ultra")
        self.assertNotIn("thread/start", self.methods())

    def test_inactive_lower_or_different_selection_requires_explicit_adoption(self):
        for model, effort in (("gpt-6-astra", "xhigh"), ("gpt-5.6-luna", "low")):
            with self.subTest(model=model, effort=effort):
                self.server.calls.clear()
                self.server.saved = {"model": model, "reasoningEffort": effort}
                with self.assertRaisesRegex(ValueError, "Explicitly adopt"):
                    self.launch(existing=True)
                self.assertNotIn("turn/start", self.methods())

    def test_active_thread_is_never_resumed_or_steered(self):
        self.server.status = {"type": "active"}
        with self.assertRaisesRegex(RuntimeError, "already active"):
            self.launch(existing=True)
        self.assertNotIn("thread/resume", self.methods())
        self.assertNotIn("turn/start", self.methods())

    def test_thread_that_becomes_active_during_resume_is_not_steered(self):
        self.server.resumed_status = "active"
        with self.assertRaisesRegex(RuntimeError, "became active"):
            self.launch(existing=True)
        self.assertNotIn("turn/start", self.methods())

    def test_audit_uses_new_high_capability_thread_and_no_executor_entrypoint(self):
        self.launch(role="audit", profile="codex-review")
        self.assertEqual(self.methods().count("thread/start"), 1)
        self.assertFalse(set(self.methods()) & {"thread/read", "thread/resume", "thread/fork"})
        self.assertEqual(self.params("turn/start")["effort"], "ultra")
        self.assertTrue(any(log.get("role") == "audit" for log in self.logs))

    def test_audit_refuses_existing_thread(self):
        with self.assertRaisesRegex(RuntimeError, "never resume"):
            self.launch(role="audit", profile="codex-review", existing=True)
        self.assertFalse(set(self.methods()) & {"thread/read", "thread/resume", "thread/start", "turn/start"})

    def test_audit_rejects_auxiliary_and_general_profiles(self):
        self.profile("auxiliary", "low", model="gpt-5.6-luna")
        for profile in ("auxiliary", "codex-general"):
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, "at least max"):
                self.launch(role="audit", profile=profile)
        self.assertNotIn("thread/start", self.methods())

    def test_audit_allows_verified_max_profile(self):
        self.profile("codex-review", "max")
        self.launch(role="audit", profile="codex-review")
        self.assertEqual(self.params("turn/start")["effort"], "max")

    def test_unconfirmed_runtime_selection_never_starts_turn(self):
        self.server.start_override = {"model": "gpt-5.6-luna", "reasoningEffort": "low"}
        with self.assertRaisesRegex(ValueError, "No turn was started"):
            self.launch()
        self.assertNotIn("turn/start", self.methods())

    def test_profile_validation_rejects_paths_permissions_malformed_toml_and_cap(self):
        malformed = self.home / "malformed.config.toml"
        malformed.write_text("model = [")
        self.profile("permissions", "ultra", extra='sandbox_mode = "danger-full-access"\n')
        self.profile("wrong-cap", "ultra", cap=4)
        for profile in ("../auth", "missing", "malformed", "permissions", "wrong-cap"):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                self.launch(profile=profile)
        self.assertNotIn("thread/start", self.methods())

    def test_pagination_finds_capability_and_unavailable_effort_fails_closed(self):
        models = self.server.pages[0]
        self.server.pages = [{"data": [], "nextCursor": "1"}, models]
        self.launch()
        self.assertEqual([params["cursor"] for method, params in self.server.calls if method == "model/list"], [None, "1"])
        self.server.calls.clear()
        self.server.pages = [{"data": [], "nextCursor": None}]
        with self.assertRaisesRegex(ValueError, "no fallback"):
            self.launch()
        self.assertNotIn("thread/start", self.methods())

    def test_missing_effort_and_broken_pagination_never_launch(self):
        self.server.pages = [{"data": [{"model": "gpt-6-astra", "supportedReasoningEfforts": [
            {"reasoningEffort": "xhigh"}
        ]}], "nextCursor": None}]
        with self.assertRaisesRegex(ValueError, "no fallback"):
            self.launch()
        self.server.pages = [{"data": [], "nextCursor": "0"}]
        with self.assertRaisesRegex(ValueError, "pagination"):
            self.launch()
        self.assertNotIn("thread/start", self.methods())

    def test_critical_profile_cannot_lower_itself(self):
        self.profile("codex-critical", "max")
        with self.assertRaisesRegex(ValueError, "at least ultra"):
            self.launch()
        self.assertNotIn("thread/start", self.methods())


if __name__ == "__main__":
    unittest.main()
