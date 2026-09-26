"""Offline tests: real Git/worktrees and a fake Devin executable; no model calls."""

import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import devin_runner as runner
import local_issue_worktree as worktrees
from executor_control import (ControlError, GitHub, LEASE_MARKER, SESSION_MARKER,
                              devin_settings, executor, read_record, select_executor)

BODY = '```execution\nexecutor = "devin"\n```\n'
REPO = "owner/repo"
SCRIPTS = Path(__file__).resolve().parent
REAL_TMUX = shutil.which("tmux")


def comment(marker, data, cid=1):
    return {"id": cid, "user": {"login": "github-actions[bot]", "type": "Bot"},
            "body": marker + "\n```json\n" + json.dumps(data, sort_keys=True) + "\n```\n"}


class FakeGitHub(GitHub):
    def __init__(self, body=BODY, state="execution-ready"):
        super().__init__(REPO, "fake-actions-secret")
        self.current_issue = {"number": 7, "state": "open", "body": body,
                              "labels": [{"name": state}]}
        self.records, self.calls, self.branch = [], [], None

    def request(self, method, path, payload=None, **kwargs):
        self.calls.append((method, path, copy.deepcopy(payload)))
        if path == "/issues/7":
            return copy.deepcopy(self.current_issue)
        if path == "/issues/8":
            return {**copy.deepcopy(self.current_issue), "number": 8}
        if path.startswith("/issues/7/comments?"):
            page = int(path.rsplit("=", 1)[1])
            return copy.deepcopy(self.records[(page - 1) * 100:page * 100])
        if path == "/git/ref/heads/codex/issue-7":
            return self.branch
        if method == "POST" and path == "/issues/7/comments":
            cid = max([c["id"] for c in self.records] + [0]) + 1
            self.records.append({**comment("", {}, cid), "body": payload["body"]})
            return copy.deepcopy(self.records[-1])
        if method == "PATCH" and path.startswith("/issues/comments/"):
            cid = int(path.rsplit("/", 1)[1])
            target = next(c for c in self.records if c["id"] == cid)
            target["body"] = payload["body"]
            return copy.deepcopy(target)
        raise AssertionError((method, path))

    def own(self):
        self.records.append(comment(LEASE_MARKER, {"executor": "devin"}))
        return self


class SettingsTests(unittest.TestCase):
    def test_codex_remains_default(self):
        for body in (None, "", "Use Devin in prose", '```codex\nmodel = "example"\n```', "```execution\n```"):
            self.assertEqual(executor(body), "codex")

    def test_explicit_executors(self):
        for value in ("codex", "devin"):
            self.assertEqual(executor(f'```execution\nexecutor = "{value}"\n```'), value)

    def test_quoted_and_nested_examples_do_not_select(self):
        for body in ("> " + BODY.replace("\n", "\n> "), "    " + BODY.replace("\n", "\n    "),
                     "````markdown\n" + BODY + "````", "~~~markdown\n" + BODY + "~~~"):
            self.assertEqual(executor(body), "codex")

    def test_duplicate_unterminated_invalid_and_oversized_blocks_fail(self):
        for body in (BODY + BODY, '```execution\nexecutor = "devin"', '```execution\nexecutor =\n```',
                     "x" * 1_048_577, 123):
            with self.subTest(body=str(body)[:40]), self.assertRaises(ControlError):
                executor(body)

    def test_unknown_executor_fields_and_values_fail(self):
        for text in ('executor = "other"', 'executor = false', 'command = "sh"', 'model = "example"'):
            with self.assertRaises(ControlError):
                executor("```execution\n" + text + "\n```")

    def test_devin_native_default_and_separate_model(self):
        self.assertEqual(devin_settings(BODY + '```codex\nmodel = "audit-model"\n```'), {})
        self.assertEqual(devin_settings('```devin\nmodel = "local-model"\n```'), {"model": "local-model"})

    def test_cloud_cost_mode_and_arbitrary_cli_arguments_rejected(self):
        for text in ('devin_mode = "normal"', 'max_acu_limit = 5', 'model_reasoning_effort = "xhigh"',
                     'command = "sh"', 'permission_mode = "dangerous"', 'model = "--cloud"',
                     'model = true', 'model = "a;touch /tmp/x"', 'model = ""'):
            with self.subTest(text=text), self.assertRaises(ControlError):
                devin_settings("```devin\n" + text + "\n```")


class RoutingTests(unittest.TestCase):
    def test_records_default_once(self):
        gh = FakeGitHub(body="")
        self.assertEqual(select_executor(gh, 7), "codex")
        self.assertEqual(select_executor(gh, 7), "codex")
        self.assertEqual(len(gh.records), 1)

    def test_selects_devin_and_refuses_hot_switch(self):
        gh = FakeGitHub()
        self.assertEqual(select_executor(gh, 7), "devin")
        gh.current_issue["body"] = ""
        with self.assertRaises(ControlError):
            select_executor(gh, 7)

    def test_delayed_events_do_not_release_holds(self):
        for state in ("queued", "blocked", "review-ready", "completed", "design-required", "investigation-required"):
            gh = FakeGitHub(state=state)
            self.assertEqual(select_executor(gh, 7, wait=True), "")
            self.assertEqual(gh.records, [])

    def test_closed_invalid_and_pr_targets(self):
        gh = FakeGitHub()
        gh.current_issue["state"] = "closed"
        self.assertEqual(select_executor(gh, 7), "")
        for change in ({"labels": []}, {"pull_request": {}}, {"number": 8}):
            gh = FakeGitHub()
            gh.current_issue.update(change)
            with self.assertRaises(ControlError):
                select_executor(gh, 7)

    def test_legacy_codex_and_cloud_are_not_adopted(self):
        for kind in ("branch", "dag", "active", "cloud"):
            gh = FakeGitHub()
            if kind == "branch":
                gh.branch = {"ref": "existing"}
            elif kind == "dag":
                gh.records.append({"body": "<!-- codex-epic-dag:v1 -->"})
            elif kind == "active":
                gh.current_issue["labels"] = [{"name": "in-progress"}]
            else:
                gh.records.append(comment(SESSION_MARKER, {"phase": "ready"}))
            with self.subTest(kind=kind), self.assertRaises(ControlError):
                select_executor(gh, 7, wait=True)

    def test_scheduler_reads_controlling_issue(self):
        gh = FakeGitHub(state="in-progress").own()
        self.assertEqual(select_executor(gh, 7), "")
        self.assertEqual(select_executor(gh, 7, wait=True), "devin")

    def test_all_comment_pages_are_checked(self):
        gh = FakeGitHub()
        gh.records = [{"id": i, "body": "ordinary"} for i in range(100)]
        gh.records.append(comment(LEASE_MARKER, {"executor": "codex"}, 101))
        with self.assertRaises(ControlError):
            select_executor(gh, 7)
        self.assertTrue(any("page=2" in path for _, path, _ in gh.calls))

    def test_duplicate_untrusted_and_malformed_records_fail(self):
        for kind in ("duplicate", "untrusted", "malformed"):
            gh = FakeGitHub().own()
            if kind == "duplicate":
                gh.records.append(copy.deepcopy(gh.records[0]))
            elif kind == "untrusted":
                gh.records[0]["user"]["login"] = "other"
            else:
                gh.records[0]["body"] = LEASE_MARKER + "\ninvalid"
            with self.assertRaises(ControlError):
                select_executor(gh, 7)


class ContextTests(unittest.TestCase):
    def body(self, fields):
        return [{"body": worktrees.CONTEXT_MARKER + "\n```yaml\n" + fields + "\n```"}]

    def test_default_and_canonical_pin(self):
        self.assertIsNone(worktrees.context([], 7))
        fields = 'epic_issue: 8\nintegration_branch: "codex/epic-8" # comment\nbase_sha: ' + "a" * 40
        self.assertEqual(worktrees.context(self.body(fields), 7)["base_sha"], "a" * 40)

    def test_duplicate_missing_alias_and_invalid_fields_fail(self):
        good = 'epic_issue: 8\nintegration_branch: integration\nbase_sha: ' + "a" * 40
        for text in (good + '\nepic_issue: 9', good.replace('epic_issue: 8', 'epic_issue: 7'),
                     good.replace('base_sha:', 'wrong_key:'), good.replace('a' * 40, '*alias'),
                     good.replace('integration_branch: integration', 'integration_branch: [x]')):
            # Invalid branch syntax is rejected by Git before preparation.
            if "[x]" in text:
                continue
            with self.assertRaises(ControlError):
                worktrees.context(self.body(text), 7)
        with self.assertRaises(ControlError):
            worktrees.context(self.body(good) * 2, 7)


class GitFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="local-devin-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root, self.remote = self.base / "durable", self.base / "remote.git"
        self.root.mkdir()
        self.command("git", "init", "-b", "main", self.root)
        self.command("git", "init", "--bare", self.remote)
        self.command("git", "-C", self.root, "config", "user.email", "test@example.invalid")
        self.command("git", "-C", self.root, "config", "user.name", "Offline test")
        (self.root / "file.txt").write_text("baseline\n")
        self.command("git", "-C", self.root, "add", "file.txt")
        self.command("git", "-C", self.root, "commit", "-m", "baseline")
        self.command("git", "-C", self.root, "remote", "add", "origin", f"https://github.com/{REPO}.git")
        self.command("git", "-C", self.root, "push", str(self.remote), "main")
        original_git = worktrees.git
        def local_git(root, *args, **kwargs):
            if args[:1] == ("fetch",):
                return original_git(root, "fetch", str(self.remote),
                                    "+refs/heads/*:refs/remotes/origin/*", **kwargs)
            return original_git(root, *args, **kwargs)
        self.git_patch = patch.object(worktrees, "git", side_effect=local_git)
        self.git_patch.start()
        self.addCleanup(self.git_patch.stop)

    def command(self, *args):
        value = subprocess.run([str(x) for x in args], text=True, capture_output=True)
        if value.returncode:
            self.fail(value.stderr)
        return value.stdout.strip()

    def prepare(self, activation=None):
        return worktrees.prepare(self.root, self.base / "worktrees", REPO, 7, "main", activation)


class WorktreeTests(GitFixture):
    def test_creates_real_separate_worktree_and_branch(self):
        path = self.prepare()
        self.assertNotEqual(path, self.root)
        self.assertEqual(self.command("git", "-C", path, "branch", "--show-current"), "codex/issue-7")
        self.assertTrue((path / ".git").is_file())

    def test_preserves_dirty_tracked_and_untracked_work(self):
        path = self.prepare()
        (path / "file.txt").write_text("unfinished\n")
        (path / "untracked.txt").write_text("keep\n")
        head = self.command("git", "-C", path, "rev-parse", "HEAD")
        self.assertEqual(self.prepare(), path)
        self.assertEqual((path / "file.txt").read_text(), "unfinished\n")
        self.assertTrue((path / "untracked.txt").exists())
        self.assertEqual(self.command("git", "-C", path, "rev-parse", "HEAD"), head)

    def test_adopts_registered_path_instead_of_creating_another(self):
        alternate = self.base / "other-location"
        self.command("git", "-C", self.root, "worktree", "add", "-b", "codex/issue-7", alternate)
        self.assertEqual(self.prepare(), alternate)

    def test_new_child_starts_at_exact_integration_pin(self):
        self.command("git", "-C", self.root, "checkout", "-b", "integration")
        (self.root / "file.txt").write_text("integration\n")
        self.command("git", "-C", self.root, "commit", "-am", "integration")
        pin = self.command("git", "-C", self.root, "rev-parse", "HEAD")
        self.command("git", "-C", self.root, "push", str(self.remote), "integration")
        self.command("git", "-C", self.root, "checkout", "main")
        path = self.prepare({"integration_branch": "integration", "base_sha": pin, "epic_issue": "8"})
        self.assertEqual(self.command("git", "-C", path, "rev-parse", "HEAD"), pin)

    def test_invalid_pin_or_branch_fails_without_issue_worktree(self):
        for activation in ({"integration_branch": "main", "base_sha": "a" * 40},
                           {"integration_branch": "bad..branch", "base_sha": "a" * 40}):
            with self.assertRaises(ControlError):
                self.prepare(activation)

    def test_unregistered_directory_is_never_deleted(self):
        path = self.base / "worktrees" / "owner-repo" / "issue-7"
        path.mkdir(parents=True)
        (path / "keep").write_text("important")
        with self.assertRaises(ControlError):
            self.prepare()
        self.assertTrue((path / "keep").exists())

    def test_wrong_origin_and_actions_paths_fail(self):
        with self.assertRaises(ControlError):
            worktrees.verify_repo(self.root, "different/repo")
        with self.assertRaises(ControlError):
            worktrees.durable_path(self.base / "_work" / "checkout")

    def test_wrong_worktree_branch_fails(self):
        path = self.prepare()
        with self.assertRaises(ControlError):
            worktrees.verify_worktree(self.root, path, REPO, "codex/issue-8")


class LocalPolicyTests(unittest.TestCase):
    def test_clean_environment_preserves_host_auth_not_actions_secrets(self):
        source = {"GH_TOKEN": "secret", "GITHUB_TOKEN": "secret", "CI": "true", "GITHUB_ACTIONS": "true",
                  "RUNNER_TRACKING_ID": "tracked", "ACTIONS_RUNTIME_TOKEN": "secret", "TMUX": "personal",
                  "CODEX_HOME": "private", "DEVIN_API_KEY": "cloud", "GIT_CONFIG_COUNT": "1",
                  "PATH": "/bin", "HOME": "/home/runner", "SSH_AUTH_SOCK": "/ssh-agent"}
        value = runner.clean_env(source)
        self.assertNotIn("secret", str(value))
        self.assertEqual(value["RUNNER_TRACKING_ID"], "")
        self.assertEqual(value["SSH_AUTH_SOCK"], "/ssh-agent")
        self.assertNotIn("TMUX", value)
        self.assertNotIn("CODEX_HOME", value)

    def test_cli_resumes_only_explicit_id_and_keeps_permissions(self):
        job = {"prompt": "/tmp/prompt", "export": "/tmp/export", "settings": {"model": "exact-id"},
               "session_id": "local-session"}
        command = runner.cli_command("/bin/devin", job)
        self.assertIn("--print", command)
        self.assertIn("--resume", command)
        self.assertIn("exact-id", command)
        for forbidden in ("--cloud", "--continue", "--permission-mode", "--sandbox", "dangerous", "false"):
            self.assertNotIn(forbidden, command)

    def test_export_requires_exact_session_and_supported_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export.json"
            runner.atomic_json(path, {"session_id": "local-session"})
            self.assertEqual(runner.export_id(path), "local-session")
            with self.assertRaises(ControlError):
                runner.export_id(path, "other")
            runner.atomic_json(path, {"unknown_schema": "x"})
            with self.assertRaises(ControlError):
                runner.export_id(path)

    def test_prompt_contains_shared_workflow_and_no_cloud_handoff(self):
        text = runner.prompt(REPO, 7)
        for required in ("SKILLFORGE_LOCAL_RUNNER=1", "codex-epic-scheduler", "spec-driven-codex-loop",
                         "codex-execution-context:v1", "review-ready", "Never merge", "max_parallel_workers"):
            self.assertIn(required, text)


class LaunchFixture(GitFixture):
    def setUp(self):
        super().setUp()
        self.bin = self.base / "bin"
        self.bin.mkdir()
        cli = self.bin / "devin"
        cli.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, sys, time
args = sys.argv[1:]
if "--help" in args:
    print("--print --prompt-file --export --resume --respect-workspace-trust")
    sys.exit(0)
if args[:2] == ["auth", "status"]:
    sys.exit(0)
path = pathlib.Path(args[args.index("--export") + 1])
sid = args[args.index("--resume") + 1] if "--resume" in args else "local-test-session"
path.with_suffix(".observed.json").write_text(json.dumps({"env": dict(os.environ), "args": args, "cwd": os.getcwd()}))
time.sleep(float(os.environ.get("FAKE_DEVIN_DELAY", "0")))
print("fake CLI completed", flush=True)
if not os.environ.get("FAKE_MISSING_EXPORT"):
    path.write_text(json.dumps({"schema_version": "ATIF-v1.6", "session_id": sid, "steps": []}))
sys.exit(int(os.environ.get("FAKE_DEVIN_EXIT", "0")))
''')
        cli.chmod(0o755)
        for name in ("gh", "tmux"):
            stub = self.bin / name
            stub.write_text("#!/bin/sh\nexit 0\n")
            stub.chmod(0o755)
        self.env_patch = patch.dict(os.environ, {"PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
                                                 "GITHUB_TOKEN": "ephemeral-test-secret",
                                                 "RUNNER_TRACKING_ID": "tracked-test", "FAKE_DEVIN_DELAY": "0"})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.gh = FakeGitHub().own()
        self.state_dir = self.base / "state" / "issue-7" / "devin"
        self.log_dir = self.base / "logs"
        self.processes, self.tmux_calls = {}, []
        original_run = runner.run
        def fake_tmux(args, **kwargs):
            if args[0] != "tmux":
                return original_run(args, **kwargs)
            self.tmux_calls.append((list(args), dict(kwargs.get("env", {}))))
            socket, operation = args[2], args[5]
            process = self.processes.get(socket)
            if operation == "list-panes":
                status = 1 if process is None else 0
                text = "0\n" if process is not None and process.poll() is None else "1\n"
                return subprocess.CompletedProcess(args, status, text, "")
            if operation == "kill-session":
                if process and process.poll() is None:
                    raise AssertionError("Tried to kill an active tmux session")
                self.processes.pop(socket, None)
                return subprocess.CompletedProcess(args, 0, "", "")
            if operation == "new-session":
                receipt = runner.read_json(self.state_dir / "state.json")
                self.assertEqual(receipt["phase"], "pending")
                self.assertIsNotNone(read_record(self.gh.records, runner.LOCAL_MARKER)[1])
                command = args[args.index("-c") + 2]
                process = subprocess.Popen(shlex.split(command), env=kwargs["env"],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           start_new_session=True)
                self.processes[socket] = process
                return subprocess.CompletedProcess(args, 0, "", "")
            raise AssertionError(args)
        self.transport_patch = patch.object(runner, "run", side_effect=fake_tmux)
        self.transport_patch.start()
        self.addCleanup(self.transport_patch.stop)
        self.addCleanup(self.finish_processes)

    def finish_processes(self):
        for process in self.processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)

    def launch(self, run_id="101", **kwargs):
        return runner.launch(self.gh, 7, self.root, self.base / "worktrees", self.state_dir,
                             self.log_dir, "main", run_id, **kwargs)

    def receipt(self):
        return runner.read_json(self.state_dir / "state.json")

    def finished(self):
        self.finish_processes()
        return self.receipt()


class LifecycleTests(LaunchFixture):
    def test_creates_real_worktree_and_records_cli_completion(self):
        result = self.launch()
        record = self.finished()
        self.assertEqual(result["result"], "launched-local-cli")
        self.assertEqual(record["phase"], "finished")
        self.assertEqual(record["exit_code"], 0)
        self.assertEqual(record["session_id"], "local-test-session")
        self.assertIn("fake CLI completed", Path(result["log"]).read_text())
        observed = runner.read_json(self.state_dir / "run-101" / "conversation.observed.json")
        self.assertEqual(observed["cwd"], record["worktree"])
        self.assertEqual(observed["env"]["SKILLFORGE_LOCAL_RUNNER"], "1")
        self.assertNotIn("ephemeral-test-secret", json.dumps(observed))
        self.assertEqual(observed["env"]["RUNNER_TRACKING_ID"], "")

    def test_replay_does_not_start_second_cli(self):
        self.launch()
        self.finished()
        self.assertEqual(self.launch()["result"], "already-acknowledged")
        self.assertEqual(sum(args[5] == "new-session" for args, _ in self.tmux_calls), 1)

    def test_new_turn_resumes_exact_session_and_preserves_unfinished_file(self):
        self.launch()
        record = self.finished()
        unfinished = Path(record["worktree"]) / "unfinished.txt"
        unfinished.write_text("keep")
        self.launch("102")
        self.finished()
        observed = runner.read_json(self.state_dir / "run-102" / "conversation.observed.json")
        self.assertIn("--resume", observed["args"])
        self.assertIn("local-test-session", observed["args"])
        self.assertTrue(unfinished.exists())

    def test_active_turn_is_not_steered(self):
        with patch.dict(os.environ, {"FAKE_DEVIN_DELAY": "0.8"}):
            self.launch()
        self.assertEqual(self.launch("102")["result"], "already-active")

    def test_scheduler_wait_is_bounded(self):
        with patch.dict(os.environ, {"FAKE_DEVIN_DELAY": "0.8"}):
            self.launch()
        with self.assertRaisesRegex(ControlError, "bounded"):
            self.launch("102", wait=True, wait_seconds=0)

    def test_hold_during_wait_prevents_followup(self):
        with patch.dict(os.environ, {"FAKE_DEVIN_DELAY": "0.8"}):
            self.launch()
        def hold(_):
            self.gh.current_issue["labels"] = [{"name": "blocked"}]
        self.assertEqual(self.launch("102", wait=True, sleep=hold)["result"], "skipped-current-state")

    def test_scheduler_resumes_after_previous_turn_exits(self):
        with patch.dict(os.environ, {"FAKE_DEVIN_DELAY": "0.4"}):
            self.launch()
        def finish(_):
            self.finish_processes()
        self.assertEqual(self.launch("102", wait=True, sleep=finish)["result"], "launched-local-cli")
        self.assertEqual(self.finished()["session_id"], "local-test-session")

    def test_cloud_receipt_is_not_migrated(self):
        self.gh.records.append(comment(SESSION_MARKER, {"phase": "ready"}, 2))
        with self.assertRaisesRegex(ControlError, "cloud"):
            self.launch()
        self.assertEqual(self.processes, {})

    def test_wrong_host_refuses_launch(self):
        self.gh.records.append(comment(runner.LOCAL_MARKER, {"transport": "cli-tmux", "host": "other"}, 2))
        with self.assertRaisesRegex(ControlError, "another local host"):
            self.launch()

    def test_missing_local_record_is_not_a_new_session(self):
        self.launch()
        self.finished()
        (self.state_dir / "state.json").unlink()
        with self.assertRaisesRegex(ControlError, "missing"):
            self.launch("102")

    def test_codex_thread_state_blocks_local_devin(self):
        self.state_dir.parent.mkdir(parents=True)
        (self.state_dir.parent / "app-server-thread-id").write_text("codex-thread")
        with self.assertRaisesRegex(ControlError, "Codex session state"):
            self.launch()

    def test_unknown_or_incomplete_receipt_never_relaunches(self):
        self.launch()
        record = self.finished()
        record["phase"] = "pending"
        runner.atomic_json(self.state_dir / "state.json", record)
        with self.assertRaisesRegex(ControlError, "Incomplete"):
            self.launch("102")

    def test_cli_error_is_recorded_not_task_success(self):
        with patch.dict(os.environ, {"FAKE_DEVIN_EXIT": "3", "FAKE_DEVIN_DELAY": "0.3"}):
            self.launch()
        record = self.finished()
        self.assertEqual(record["phase"], "failed")
        self.assertEqual(record["exit_code"], 3)
        with self.assertRaises(ControlError):
            self.launch()

    def test_missing_export_requires_reconciliation(self):
        with patch.dict(os.environ, {"FAKE_MISSING_EXPORT": "1", "FAKE_DEVIN_DELAY": "0.3"}):
            self.launch()
        self.assertEqual(self.finished()["phase"], "needs-reconciliation")
        with self.assertRaises(ControlError):
            self.launch("102")

    def test_tmux_receives_clean_environment_and_isolated_socket(self):
        self.launch()
        self.finished()
        args, env = next((a, e) for a, e in self.tmux_calls if a[5] == "new-session")
        self.assertTrue(args[2].startswith("sf-devin-"))
        self.assertIn("/dev/null", args)
        self.assertIn("remain-on-exit", args)
        self.assertEqual(env["RUNNER_TRACKING_ID"], "")
        self.assertNotIn("GITHUB_TOKEN", env)
        self.assertNotIn("TMUX", env)


class RealTmuxTests(LaunchFixture):
    def test_real_tmux_detachment_and_exact_resume_with_fake_cli(self):
        if not REAL_TMUX:
            if os.environ.get("REQUIRE_TMUX_TEST") == "1":
                self.fail("tmux is required for the transport integration gate")
            self.skipTest("tmux is not installed; CI requires this real transport test")
        self.transport_patch.stop()
        # The fake executable remains Devin only; tmux is the actual OS binary.
        (self.bin / "tmux").unlink()
        (self.bin / "tmux").symlink_to(REAL_TMUX)
        result = self.launch()
        self.addCleanup(lambda: subprocess.run(runner.tmux_args(self.receipt()["socket"], "kill-server"),
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        def wait_final():
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if self.receipt()["phase"] in runner.TERMINAL:
                    # Let supervisor leave the pane and release its lifetime lock.
                    time.sleep(0.2)
                    return
                time.sleep(0.05)
            self.fail("Local tmux CLI did not finish")
        wait_final()
        self.assertEqual(self.receipt()["phase"], "finished")
        panes = subprocess.run(runner.tmux_args(result["socket"], "list-panes", "-t", "=" + result["session"] + ":0",
                                                "-F", "#{pane_dead}"), text=True, capture_output=True)
        self.assertEqual(panes.stdout.strip(), "1")
        self.launch("102")
        wait_final()
        self.assertEqual(self.receipt()["session_id"], "local-test-session")
        observed = runner.read_json(self.state_dir / "run-102" / "conversation.observed.json")
        self.assertIn("--resume", observed["args"])
        self.assertNotIn("ephemeral-test-secret", json.dumps(observed))


class WorkflowTests(unittest.TestCase):
    def test_local_launcher_has_no_cloud_or_ephemeral_checkout(self):
        source = (SCRIPTS.parent / "workflows/devin-execute-ready.yml").read_text()
        self.assertIn("runs-on: [self-hosted, codex]", source)
        self.assertIn("SKILLFORGE_REPO_ROOT", source)
        self.assertIn('"$GITHUB_SHA:.github/scripts/$name"', source)
        for forbidden in ("DEVIN_API_KEY", "DEVIN_ORG_ID", "secrets:", "actions/checkout"):
            self.assertNotIn(forbidden, source)

    def test_single_dispatcher_and_fixed_independent_audit(self):
        sources = {p.name: p.read_text() for p in (SCRIPTS.parent / "workflows").glob("*.yml")}
        if "codex-issue-state.yml" not in sources:
            self.skipTest("Full repository workflow set is checked in CI")
        self.assertEqual([name for name, text in sources.items() if "types: [labeled]" in text],
                         ["codex-issue-state.yml"])
        dispatcher = sources["codex-issue-state.yml"]
        self.assertIn("needs.select-executor.outputs.executor == 'devin'", dispatcher)
        self.assertIn("uses: ./.github/workflows/codex-review-ready.yml", dispatcher)
        self.assertNotIn("DEVIN_API_KEY", dispatcher)


if __name__ == "__main__":
    unittest.main()
