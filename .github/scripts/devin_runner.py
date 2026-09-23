"""Bounded Devin v3 launcher. No Actions credentials enter a Devin session.

A durable pending receipt precedes every non-idempotent API write. Ambiguous
failures stop for reconciliation instead of creating duplicate paid sessions.
"""

import json
import os
import re
import time
from urllib.parse import urlsplit

from executor_control import (ControlError, GitHub, JsonAPI, SESSION_MARKER,
                              devin_settings, eligible, executor, positive_int,
                              read_record, require_lease)


def identifier(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", value):
        raise ControlError(f"Invalid {name}")
    return value


def session_id(value):
    value = identifier(value, "Devin session ID")
    return value if value.startswith("devin-") else "devin-" + value


def session_url(value):
    if not isinstance(value, str) or any(c.isspace() for c in value):
        raise ControlError("Invalid Devin session URL")
    url = urlsplit(value)
    if (url.scheme != "https" or url.netloc != "app.devin.ai"
            or not re.fullmatch(r"/sessions/[A-Za-z0-9_-]+", url.path)
            or url.query or url.fragment):
        raise ControlError("Unexpected Devin session URL")
    return value


def prompt(repo, number):
    # No issue title/body, shell snippet, credential, or configurable endpoint is
    # interpolated. The agent reads the live authorized contract via its own SCM.
    return f"""Execute the controlling issue https://github.com/{repo}/issues/{number}.
Repository: https://github.com/{repo}. Executor: Devin, not Codex.
Read AGENTS.md and skills/execution-runner-selection/SKILL.md from the current
default branch, then the live issue and its canonical top-level comments.
Use your configured GitHub integration and persistent VM, never an Actions token,
Codex credentials, local Codex socket, or another executor's worktree.
Verify exact repository origin, single workflow state and exclusive ownership.
Keep the existing branch convention codex/issue-{number}; it is a workflow
identifier, not a request to run Codex. Preserve unfinished work on resume.
The dispatcher-owned executor/session comments are read-only to you. Never edit,
delete or duplicate them. Do not spawn other product implementation sessions.
If execution_mode is epic-dag, use skills/codex-epic-scheduler/SKILL.md; otherwise
use skills/spec-driven-codex-loop/SKILL.md. Resolve and verify canonical
codex-execution-context:v1 base/PR target before any implementation edit.
Only the epic scheduler may activate queued children. Parent settings do not
propagate to children. Respect dependency edges, holds and max_parallel_workers.
Implementation must end at a ready PR and review-ready handoff to the independent
Codex audit. Never merge, enable auto-merge, close issues, mark completed, or
continue GitHub mutations after review-ready. Follow repository validation gates.
Do not bypass approvals or silently change model/mode, executor or ownership.
If required repository access or environment is unavailable, report the precise
blocker; do not fabricate test results or choose a different executor.
"""


class Devin(JsonAPI):
    def __init__(self, org, token):
        self.org = identifier(org, "DEVIN_ORG_ID")
        if not self.org.startswith("org-"):
            raise ControlError("DEVIN_ORG_ID must be an organization ID beginning org-")
        super().__init__(f"https://api.devin.ai/v3/organizations/{self.org}", token)


def validate_session(value, org, scope, expected_id=None):
    if not isinstance(value, dict) or value.get("org_id") != org:
        raise ControlError("Devin session organization was not confirmed")
    sid = session_id(value.get("session_id"))
    if expected_id is not None and sid != session_id(expected_id):
        raise ControlError("Devin returned a different session")
    if not isinstance(value.get("tags"), list) or scope not in value["tags"]:
        raise ControlError("Devin session is not bound to this repository and issue")
    session_url(value.get("url"))
    if value.get("is_archived"):
        raise ControlError("Devin session is archived; explicit recovery required")
    return sid


def busy(value):
    status, detail = value.get("status"), value.get("status_detail")
    if status in {"new", "claimed", "resuming"}:
        return True
    if status == "running":
        if detail == "waiting_for_approval":
            raise ControlError("Devin is awaiting approval; the launcher cannot bypass it")
        if detail in {"finished", "waiting_for_user"}:
            return False
        return True  # Unknown running details are not evidence of idleness.
    if status == "exit" or status == "suspended" and detail == "inactivity":
        return False
    raise ControlError("Devin is not safely resumable (error, hold, quota or unknown state)")


def launch(gh, devin, number, run_id, ceiling, *, wait=False,
           sleep=time.sleep, clock=time.monotonic, wait_seconds=900):
    number = str(number)
    if not re.fullmatch(r"[1-9][0-9]*", number):
        raise ControlError("Invalid controlling issue number")
    run_id = identifier(run_id, "Actions run ID")
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ControlError("Invalid Actions run ID")
    scope = f"skillforge:{gh.repo}:issue-{number}"

    def current():
        issue = gh.issue(number)
        if not eligible(issue, wait):
            return None
        if executor(issue["body"]) != "devin":
            raise ControlError("Executor selection changed before launch")
        require_lease(gh, number, "devin")
        return devin_settings(issue["body"], ceiling)

    settings = current()
    if settings is None:
        return {"result": "skipped-current-state"}
    comment, record = read_record(gh.comments(number), SESSION_MARKER)
    sid = None
    if record is not None:
        if (record.get("org_id") != devin.org or record.get("scope") != scope
                or record.get("settings") != settings):
            raise ControlError("Session owner/settings changed; explicit idle-session migration required")
        if record.get("phase") != "ready":
            raise ControlError("Pending Devin request: reconcile its recorded run before retrying; no duplicate sent")
        sid = session_id(record.get("session_id"))
        session_url(record.get("url"))
        if record.get("last_run_id") == run_id:
            return {"result": "already-acknowledged", "url": record["url"]}
        deadline = clock() + wait_seconds
        while True:
            session = devin.request("GET", f"/sessions/{sid}")
            validate_session(session, devin.org, scope, sid)
            if "devin_mode" in settings and session.get("devin_mode") != settings["devin_mode"]:
                raise ControlError("Session mode differs from the explicit request; no fallback selected")
            if not busy(session):
                break
            if not wait:
                return {"result": "already-active", "url": session["url"]}
            if clock() >= deadline:
                raise ControlError("Existing Devin turn is still active; bounded scheduler wait expired")
            sleep(10)
            refreshed = current()
            if refreshed is None:
                return {"result": "skipped-current-state"}
            if refreshed != settings:
                raise ControlError("Devin settings changed while waiting")

    # Revalidate after polling and immediately before any paid API write.
    refreshed = current()
    if refreshed is None:
        return {"result": "skipped-current-state"}
    if refreshed != settings:
        raise ControlError("Devin settings changed before launch")
    pending = {"phase": "pending", "org_id": devin.org, "scope": scope,
               "settings": settings, "last_run_id": run_id}
    if sid:
        pending.update(session_id=sid, url=record["url"])
    comment = gh.save_record(number, SESSION_MARKER, pending, comment)
    if sid:
        result = devin.request("POST", f"/sessions/{sid}/messages",
                               {"message": prompt(gh.repo, number)})
    else:
        result = devin.request("POST", "/sessions", {
            "prompt": prompt(gh.repo, number), "title": f"{gh.repo} issue #{number}",
            "tags": ["skillforge", scope, f"skillforge-run:{run_id}"],
            "session_links": [f"https://github.com/{gh.repo}/issues/{number}"],
            "resumable": True, **settings,
        })
    sid = validate_session(result, devin.org, scope, sid)
    # Save the session identity even when mode confirmation fails, so an operator
    # can reconcile the already-created session without creating another.
    pending.update(session_id=sid, url=result["url"])
    comment = gh.save_record(number, SESSION_MARKER, pending, comment)
    if "devin_mode" in settings and result.get("devin_mode") != settings["devin_mode"]:
        raise ControlError("Devin did not confirm the requested mode; reconcile the recorded session")
    gh.save_record(number, SESSION_MARKER, {**pending, "phase": "ready"}, comment)
    return {"result": "resumed" if record else "created", "url": result["url"]}


def main():
    # Settings are admin-controlled workflow env, not issue-supplied commands.
    if os.environ.get("GITHUB_REF") != "refs/heads/" + os.environ["DEFAULT_BRANCH"]:
        raise ControlError("Devin launches must use the repository default branch")
    raw_limit = os.environ.get("DEVIN_MAX_ACU_LIMIT", "")
    if not re.fullmatch(r"[1-9][0-9]{0,5}", raw_limit):
        raise ControlError("Set repository variable DEVIN_MAX_ACU_LIMIT to a positive integer before enabling Devin")
    ceiling = positive_int(int(raw_limit), "DEVIN_MAX_ACU_LIMIT")
    result = launch(
        GitHub(os.environ["GITHUB_REPOSITORY"], os.environ.get("GITHUB_TOKEN")),
        Devin(os.environ.get("DEVIN_ORG_ID", ""), os.environ.get("DEVIN_API_KEY")),
        os.environ["ISSUE_NUMBER"], os.environ["GITHUB_RUN_ID"], ceiling,
        wait=os.environ.get("WAIT_FOR_EXISTING_TURN") == "true",
    )
    print(json.dumps(result, sort_keys=True))
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as output:
        output.write(f"Devin launcher: **{result['result']}**. Launch acknowledgement is not task completion.\n\n")
        if "url" in result:
            output.write(f"Session: {result['url']}\n")


if __name__ == "__main__":
    try:
        main()
    except (ControlError, KeyError) as exc:
        raise SystemExit(str(exc)) from None
