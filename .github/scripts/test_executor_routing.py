"""Offline routing/Devin lifecycle regressions: no external or paid calls."""

import copy
import io
import json
from pathlib import Path
import unittest
import urllib.error

import devin_runner as runner
from executor_control import (ControlError, GitHub, JsonAPI, LEASE_MARKER,
                              NoRedirect, SESSION_MARKER, block, devin_settings,
                              executor, read_record, select_executor)


REPO = "owner/repo"
SCOPE = f"skillforge:{REPO}:issue-7"
URL = "https://app.devin.ai/sessions/abc"
BODY = '```execution\nexecutor = "devin"\n```\n'


def comment(marker, record, cid=1):
    return {"id": cid, "user": {"login": "github-actions[bot]", "type": "Bot"},
            "body": marker + "\n```json\n" + json.dumps(record, sort_keys=True) + "\n```\n"}


class FakeGitHub(GitHub):
    def __init__(self, body=BODY, state="execution-ready"):
        super().__init__(REPO, "fake-actions-secret")
        self.current_issue = {"number": 7, "state": "open", "body": body,
                              "labels": [{"name": state}]}
        self.records = []
        self.calls = []
        self.branch = None
        self.fail_save = False

    def request(self, method, path, payload=None, **kwargs):
        self.calls.append((method, path, copy.deepcopy(payload)))
        if path == "/issues/7":
            return copy.deepcopy(self.current_issue)
        if path.startswith("/issues/7/comments?"):
            page = int(path.rsplit("=", 1)[1])
            return copy.deepcopy(self.records[(page - 1) * 100:page * 100])
        if path == "/git/ref/heads/codex/issue-7":
            return copy.deepcopy(self.branch)
        if self.fail_save:
            raise ControlError("simulated GitHub record failure")
        if method == "POST" and path == "/issues/7/comments":
            cid = max([r["id"] for r in self.records] + [0]) + 1
            self.records.append({"id": cid, "user": {"login": "github-actions[bot]", "type": "Bot"},
                                 "body": payload["body"]})
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

    def session(self, *, phase="ready", settings=None, run="100"):
        self.records.append(comment(SESSION_MARKER, {
            "phase": phase, "org_id": "org-test", "scope": SCOPE,
            "settings": settings or {"max_acu_limit": 10}, "last_run_id": run,
            "session_id": "devin-abc", "url": URL,
        }, 2))
        return self


class FakeDevin:
    org = "org-test"

    def __init__(self, gh, status="running", detail="finished"):
        self.gh, self.calls, self.fail = gh, [], False
        self.response = {"session_id": "devin-abc", "org_id": self.org,
                         "url": URL, "tags": [SCOPE], "devin_mode": "normal",
                         "status": status, "status_detail": detail}

    def request(self, method, path, payload=None):
        self.calls.append((method, path, copy.deepcopy(payload)))
        if method == "POST":
            # This assertion exercises ordering: receipt must precede paid call.
            _, receipt = read_record(self.gh.records, SESSION_MARKER)
            assert receipt["phase"] == "pending"
            if self.fail:
                raise ControlError("simulated ambiguous provider failure")
        return copy.deepcopy(self.response)

    def posts(self):
        return [call for call in self.calls if call[0] == "POST"]


class SettingsTests(unittest.TestCase):
    def test_codex_is_backward_compatible_default(self):
        for body in (None, "", "Use Devin in prose", '```codex\nmodel = "gpt-6-sol"\nmodel_reasoning_effort = "xhigh"\n```'):
            with self.subTest(body=body):
                self.assertEqual(executor(body), "codex")

    def test_explicit_executors(self):
        for value in ("codex", "devin"):
            self.assertEqual(executor(f'```execution\nexecutor = "{value}"\n```'), value)

    def test_empty_execution_block_is_native(self):
        self.assertEqual(executor("```execution\n```"), "codex")

    def test_only_top_level_configuration_is_used(self):
        for body in ("> ```execution\n> executor = \"devin\"\n> ```",
                     "    ```execution\n    executor = \"devin\"\n    ```",
                     "````markdown\n" + BODY + "````",
                     "~~~markdown\n" + BODY + "~~~"):
            with self.subTest(body=body):
                self.assertEqual(executor(body), "codex")

    def test_tilde_and_longer_closing_fence(self):
        self.assertEqual(executor('  ~~~execution\nexecutor = "devin"\n  ~~~~'), "devin")

    def test_duplicate_unterminated_and_malformed_fail(self):
        for body in (BODY + BODY, '```execution\nexecutor = "devin"',
                     '```execution\nexecutor = \n```'):
            with self.subTest(body=body), self.assertRaises(ControlError):
                executor(body)

    def test_no_unknown_executor_or_command_fields(self):
        for settings in ('executor = "unknown"', 'executor = false', 'executor = ["devin"]',
                         'executor = "Devin"', 'executor = ""', 'model = "x"',
                         'command = "echo unsafe"', 'executor = "devin"\n[auth]\ntoken = "x"'):
            with self.subTest(settings=settings), self.assertRaises(ControlError):
                executor("```execution\n" + settings + "\n```")

    def test_body_and_block_limits(self):
        for body in (123, "x" * 1_048_577, '```execution\n#' + "x" * 8192 + '\n```'):
            with self.assertRaises(ControlError):
                executor(body)

    def test_provider_settings_stay_separate(self):
        body = BODY + '```codex\nmodel = "gpt-6-sol"\nmodel_reasoning_effort = "xhigh"\n```'
        self.assertEqual(devin_settings(body, 10), {"max_acu_limit": 10})
        self.assertEqual(block(body, "codex")["model"], "gpt-6-sol")

    def test_devin_modes_and_cost_ceiling(self):
        for mode in ("normal", "fast", "lite", "ultra", "fusion"):
            self.assertEqual(devin_settings(f'```devin\ndevin_mode = "{mode}"\nmax_acu_limit = 5\n```', 10),
                             {"devin_mode": mode, "max_acu_limit": 5})

    def test_rejects_unbounded_cost_and_codex_parameters(self):
        for value in ('max_acu_limit = 11', 'max_acu_limit = 0', 'max_acu_limit = -1',
                      'max_acu_limit = true', 'max_acu_limit = "3"', 'max_acu_limit = 3.5',
                      'devin_mode = "sol"', 'devin_mode = []', 'model = "gpt-6-sol"',
                      'model_reasoning_effort = "xhigh"', 'bypass_approval = true',
                      'session_secrets = []', 'platform = "other-host"'):
            with self.subTest(value=value), self.assertRaises(ControlError):
                devin_settings("```devin\n" + value + "\n```", 10)
        for ceiling in (0, -1, True, "10", 100001):
            with self.assertRaises(ControlError):
                devin_settings("", ceiling)


class RoutingTests(unittest.TestCase):
    def test_selects_and_records_codex_default_once(self):
        gh = FakeGitHub(body="")
        self.assertEqual(select_executor(gh, 7), "codex")
        self.assertEqual(select_executor(gh, 7), "codex")
        self.assertEqual(len(gh.records), 1)

    def test_explicit_devin_routing(self):
        gh = FakeGitHub()
        self.assertEqual(select_executor(gh, 7), "devin")
        self.assertEqual(read_record(gh.records, LEASE_MARKER)[1], {"executor": "devin"})

    def test_does_not_automatically_switch_executors(self):
        gh = FakeGitHub().own()
        gh.current_issue["body"] = ""
        with self.assertRaises(ControlError):
            select_executor(gh, 7)

    def test_legacy_work_cannot_be_adopted_as_devin(self):
        for kind in ("branch", "dag", "active"):
            gh = FakeGitHub()
            if kind == "branch":
                gh.branch = {"ref": "refs/heads/codex/issue-7"}
            elif kind == "dag":
                gh.records.append({"id": 9, "body": "<!-- codex-epic-dag:v1 -->"})
            else:
                gh.current_issue["labels"] = [{"name": "in-progress"}]
            with self.subTest(kind=kind), self.assertRaises(ControlError):
                select_executor(gh, 7, wait=True)
            self.assertEqual(len([c for c in gh.calls if c[0] == "POST"]), 0)

    def test_delayed_event_preserves_holds(self):
        for state in ("queued", "blocked", "design-required", "investigation-required", "completed", "review-ready"):
            gh = FakeGitHub(state=state)
            with self.subTest(state=state):
                self.assertEqual(select_executor(gh, 7, wait=True), "")
                self.assertEqual(gh.records, [])

    def test_closed_issue_is_not_launched(self):
        gh = FakeGitHub()
        gh.current_issue["state"] = "closed"
        self.assertEqual(select_executor(gh, 7), "")

    def test_scheduler_uses_the_resolved_controlling_issue(self):
        gh = FakeGitHub(state="in-progress").own()
        self.assertEqual(select_executor(gh, 7, wait=True), "devin")
        self.assertEqual(select_executor(gh, 7, wait=False), "")

    def test_invalid_labels_and_prs_fail_closed(self):
        for update in ({"labels": []}, {"labels": [{"name": "queued"}, {"name": "execution-ready"}]},
                       {"pull_request": {}}, {"number": 8}):
            gh = FakeGitHub()
            gh.current_issue.update(update)
            with self.subTest(update=update), self.assertRaises(ControlError):
                select_executor(gh, 7)

    def test_paginated_comments_are_fully_checked(self):
        gh = FakeGitHub()
        gh.records = [{"id": i, "body": "ordinary comment"} for i in range(100)]
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
                gh.records[0]["user"]["login"] = "someone-else"
            else:
                gh.records[0]["body"] = LEASE_MARKER + "\ninvalid"
            with self.subTest(kind=kind), self.assertRaises(ControlError):
                select_executor(gh, 7)

    def test_failed_receipt_never_claims_success(self):
        gh = FakeGitHub()
        gh.fail_save = True
        with self.assertRaises(ControlError):
            select_executor(gh, 7)


class LifecycleTests(unittest.TestCase):
    def start(self, gh=None, **kwargs):
        gh = gh or FakeGitHub().own()
        devin = FakeDevin(gh)
        result = runner.launch(gh, devin, 7, "101", 10, **kwargs)
        return gh, devin, result

    def test_create_has_a_durable_receipt_and_only_devin_fields(self):
        gh, devin, result = self.start()
        self.assertEqual(result["result"], "created")
        payload = devin.posts()[0][2]
        self.assertEqual(payload["max_acu_limit"], 10)
        self.assertNotIn("devin_mode", payload)  # Organization-native mode.
        for forbidden in ("model", "model_reasoning_effort", "session_secrets", "bypass_approval", "platform"):
            self.assertNotIn(forbidden, payload)
        self.assertNotIn("fake-actions-secret", json.dumps(payload))
        self.assertEqual(read_record(gh.records, SESSION_MARKER)[1]["phase"], "ready")

    def test_same_actions_run_is_not_replayed(self):
        gh, devin, _ = self.start()
        self.assertEqual(runner.launch(gh, devin, 7, "101", 10)["result"], "already-acknowledged")
        self.assertEqual(len(devin.posts()), 1)

    def test_resumes_only_recorded_session(self):
        gh = FakeGitHub().own().session()
        _, devin, result = self.start(gh)
        self.assertEqual(result["result"], "resumed")
        self.assertEqual(devin.posts()[0][1], "/sessions/devin-abc/messages")
        self.assertEqual(set(devin.posts()[0][2]), {"message"})

    def test_active_normal_turn_is_not_steered(self):
        gh = FakeGitHub().own().session()
        devin = FakeDevin(gh, detail="working")
        self.assertEqual(runner.launch(gh, devin, 7, "101", 10)["result"], "already-active")
        self.assertEqual(devin.posts(), [])

    def test_scheduler_waits_then_resumes(self):
        gh = FakeGitHub(state="in-progress").own().session()
        devin = FakeDevin(gh, detail="working")
        waits = []
        def finish(seconds):
            waits.append(seconds)
            devin.response["status_detail"] = "finished"
        result = runner.launch(gh, devin, 7, "101", 10, wait=True, sleep=finish)
        self.assertEqual(result["result"], "resumed")
        self.assertEqual(waits, [10])

    def test_scheduler_wait_is_bounded(self):
        gh = FakeGitHub(state="in-progress").own().session()
        devin = FakeDevin(gh, detail="working")
        with self.assertRaises(ControlError):
            runner.launch(gh, devin, 7, "101", 10, wait=True, clock=lambda: 0, wait_seconds=0)
        self.assertEqual(devin.posts(), [])

    def test_hold_during_wait_cancels_launch(self):
        gh = FakeGitHub(state="in-progress").own().session()
        devin = FakeDevin(gh, detail="working")
        def hold(_):
            gh.current_issue["labels"] = [{"name": "blocked"}]
        self.assertEqual(runner.launch(gh, devin, 7, "101", 10, wait=True, sleep=hold)["result"],
                         "skipped-current-state")
        self.assertEqual(devin.posts(), [])

    def test_ambiguous_create_is_never_retried_automatically(self):
        gh = FakeGitHub().own()
        devin = FakeDevin(gh)
        devin.fail = True
        with self.assertRaises(ControlError):
            runner.launch(gh, devin, 7, "101", 10)
        self.assertEqual(read_record(gh.records, SESSION_MARKER)[1]["phase"], "pending")
        devin.fail = False
        with self.assertRaises(ControlError):
            runner.launch(gh, devin, 7, "102", 10)
        self.assertEqual(len(devin.posts()), 1)

    def test_ambiguous_resume_is_not_resent(self):
        gh = FakeGitHub().own().session()
        devin = FakeDevin(gh)
        devin.fail = True
        with self.assertRaises(ControlError):
            runner.launch(gh, devin, 7, "101", 10)
        devin.fail = False
        with self.assertRaises(ControlError):
            runner.launch(gh, devin, 7, "102", 10)
        self.assertEqual(len(devin.posts()), 1)

    def test_missing_lease_or_failed_receipt_prevents_paid_request(self):
        for kind in ("lease", "receipt"):
            gh = FakeGitHub()
            if kind == "receipt":
                gh.own()
                gh.fail_save = True
            devin = FakeDevin(gh)
            with self.subTest(kind=kind), self.assertRaises(ControlError):
                runner.launch(gh, devin, 7, "101", 10)
            self.assertEqual(devin.posts(), [])

    def test_configuration_change_does_not_migrate_session(self):
        gh = FakeGitHub(body=BODY + '```devin\ndevin_mode = "ultra"\n```').own().session()
        devin = FakeDevin(gh)
        with self.assertRaises(ControlError):
            runner.launch(gh, devin, 7, "101", 10)
        self.assertEqual(devin.calls, [])

    def test_mode_mismatch_is_recorded_not_reported_as_success(self):
        gh = FakeGitHub(body=BODY + '```devin\ndevin_mode = "ultra"\n```').own()
        devin = FakeDevin(gh)
        with self.assertRaises(ControlError):
            runner.launch(gh, devin, 7, "101", 10)
        record = read_record(gh.records, SESSION_MARKER)[1]
        self.assertEqual(record["session_id"], "devin-abc")
        self.assertEqual(record["phase"], "pending")

    def test_wrong_session_organization_scope_and_url_fail(self):
        for update in ({"session_id": "different"}, {"org_id": "org-other"}, {"tags": []},
                       {"url": "https://attacker.invalid/sessions/abc"}, {"is_archived": True}):
            gh = FakeGitHub().own().session()
            devin = FakeDevin(gh)
            devin.response.update(update)
            with self.subTest(update=update), self.assertRaises(ControlError):
                runner.launch(gh, devin, 7, "101", 10)
            self.assertEqual(devin.posts(), [])

    def test_approvals_quota_and_unknown_status_are_not_bypassed(self):
        for status, detail in (("running", "waiting_for_approval"), ("suspended", "usage_limit_exceeded"),
                               ("suspended", "user_request"), ("error", None), ("unrecognized", None)):
            gh = FakeGitHub().own().session()
            devin = FakeDevin(gh, status, detail)
            with self.subTest(status=status, detail=detail), self.assertRaises(ControlError):
                runner.launch(gh, devin, 7, "101", 10)
            self.assertEqual(devin.posts(), [])

    def test_busy_unknown_detail_is_not_idleness(self):
        self.assertTrue(runner.busy({"status": "running", "status_detail": "future-value"}))

    def test_native_inactive_session_can_resume(self):
        for status, detail in (("exit", None), ("suspended", "inactivity"), ("running", "waiting_for_user")):
            self.assertFalse(runner.busy({"status": status, "status_detail": detail}))

    def test_identifiers_cannot_inject_paths_or_prompts(self):
        for sid in ("../x", "x/y", "x\n", "", None):
            with self.assertRaises(ControlError):
                runner.session_id(sid)
        for repo in ("../repo", "owner/..", "owner/repo/other", "owner/\nrepo"):
            with self.assertRaises(ControlError):
                GitHub(repo, "token")
        self.assertEqual(runner.session_id("abc"), "devin-abc")
        self.assertEqual(runner.session_id("devin-abc"), "devin-abc")

    def test_prompt_preserves_epic_and_audit_contracts(self):
        value = runner.prompt(REPO, 7)
        for required in ("codex/issue-7", "codex-epic-scheduler", "spec-driven-codex-loop",
                         "codex-execution-context:v1", "review-ready", "Never merge",
                         "max_parallel_workers", "Parent settings do not"):
            self.assertIn(required, value)


class HttpTests(unittest.TestCase):
    def test_missing_credentials_are_explicit(self):
        with self.assertRaises(ControlError):
            runner.Devin("org-test", "")
        with self.assertRaises(ControlError):
            runner.Devin("../bad", "fake")

    def test_redirects_cannot_forward_authorization(self):
        with self.assertRaises(ControlError):
            NoRedirect().redirect_request(None, None, 302, "", {}, "https://attacker.invalid")

    def test_request_uses_timeout_no_retry_and_redacted_errors(self):
        api = JsonAPI("https://api.devin.ai", "do-not-print-token")
        calls = []
        class Opener:
            def open(self, request, timeout):
                calls.append((request, timeout))
                raise urllib.error.HTTPError(request.full_url, 401, "do-not-print-token", {}, None)
        api.opener = Opener()
        with self.assertRaises(ControlError) as error:
            api.request("POST", "/sessions", {"prompt": "test"})
        self.assertNotIn("do-not-print-token", str(error.exception))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 30)

    def test_read_404_is_the_only_optional_failure(self):
        api = JsonAPI("https://api.github.com", "fake")
        class Opener:
            def open(self, request, timeout):
                raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)
        api.opener = Opener()
        self.assertIsNone(api.request("GET", "/ref", missing_ok=True))
        with self.assertRaises(ControlError):
            api.request("POST", "/ref", {}, missing_ok=True)

    def test_invalid_and_oversized_json_fail(self):
        api = JsonAPI("https://api.github.com", "fake")
        for raw in (b"invalid", b"x" * 4_194_305):
            class Opener:
                def open(self, request, timeout):
                    return io.BytesIO(raw)
            api.opener = Opener()
            with self.assertRaises(ControlError):
                api.request("GET", "/test")


class WorkflowContractTests(unittest.TestCase):
    def test_single_label_entrypoint_and_fixed_fresh_audit(self):
        workflows = Path(__file__).resolve().parents[1] / "workflows"
        sources = {p.name: p.read_text() for p in workflows.glob("*.yml")}
        listeners = [name for name, body in sources.items() if "types: [labeled]" in body]
        self.assertEqual(listeners, ["codex-issue-state.yml"])
        dispatcher = sources["codex-issue-state.yml"]
        self.assertIn("needs.select-executor.outputs.executor == 'codex'", dispatcher)
        self.assertIn("needs.select-executor.outputs.executor == 'devin'", dispatcher)
        self.assertIn("uses: ./.github/workflows/codex-review-ready.yml", dispatcher)
        self.assertIn("needs: [route, select-executor]", dispatcher)
        self.assertNotIn("secrets: inherit", dispatcher)

    def test_devin_launcher_never_targets_local_codex_host(self):
        workflows = Path(__file__).resolve().parents[1] / "workflows"
        source = (workflows / "devin-execute-ready.yml").read_text()
        self.assertIn("workflow_call:", source)
        self.assertIn("runs-on: ubuntu-latest", source)
        self.assertIn("cancel-in-progress: false", source)
        self.assertNotIn("self-hosted", source)
        self.assertNotIn("issues:\n    types:", source)
        for variable in ("DEVIN_API_KEY", "DEVIN_ORG_ID", "DEVIN_MAX_ACU_LIMIT"):
            self.assertIn(variable, source)


if __name__ == "__main__":
    unittest.main()
