"""Select an issue's executor, independently of provider-owned model settings.

The hosted dispatcher owns the executor lease; provider launchers own sessions.
Only the trusted workflow revision may run this code with Actions credentials.
"""

import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.request


LEASE_MARKER = "<!-- skillforge-executor:v1 -->"
SESSION_MARKER = "<!-- skillforge-devin-session:v1 -->"
STATES = {"queued", "execution-ready", "in-progress", "review-ready",
          "design-required", "investigation-required", "blocked", "completed"}
MODES = {"normal", "fast", "lite", "ultra", "fusion"}


class ControlError(RuntimeError):
    """An actionable control-plane error, safe to print without credentials."""


def block(body, name):
    """Read one top-level TOML fence, never quoted/indented/nested examples."""
    if body is None:
        return {}
    if not isinstance(body, str) or len(body.encode("utf-8")) > 1_048_576:
        raise ControlError("Invalid or oversized issue body")
    fence, selected, found, lines = None, False, False, []
    for line in body.splitlines():
        if fence:
            if re.fullmatch(r" {0,3}" + re.escape(fence[0]) +
                            "{" + str(len(fence)) + r",}[ \t]*", line):
                fence, selected = None, False
            elif selected:
                lines.append(line)
            continue
        opening = re.fullmatch(r" {0,3}(`{3,}|~{3,})([^\r\n]*)", line)
        if opening:
            fence, info = opening.groups()
            selected = info.strip() == name
            if selected:
                if found:
                    raise ControlError(f"Duplicate {name} settings block")
                found = True
    if selected:
        raise ControlError(f"Unterminated {name} settings block")
    text = "\n".join(lines)
    if len(text.encode("utf-8")) > 8192:
        raise ControlError(f"Oversized {name} settings block")
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        raise ControlError(f"Malformed TOML in {name} settings block") from None


def executor(body):
    settings = block(body, "execution")
    if set(settings) - {"executor"}:
        raise ControlError("execution permits only executor; model settings belong to provider blocks")
    value = settings.get("executor", "codex")
    if not isinstance(value, str) or value not in {"codex", "devin"}:
        raise ControlError("executor must be codex or devin; no fallback selected")
    return value


def positive_int(value, name):
    if type(value) is not int or not 1 <= value <= 100_000:
        raise ControlError(f"{name} must be an integer between 1 and 100000")
    return value


def devin_settings(body, ceiling):
    ceiling = positive_int(ceiling, "DEVIN_MAX_ACU_LIMIT")
    settings = block(body, "devin")
    if set(settings) - {"devin_mode", "max_acu_limit"}:
        raise ControlError("devin permits only devin_mode and max_acu_limit; Codex model/effort are not Devin settings")
    mode = settings.get("devin_mode")
    if "devin_mode" in settings and (not isinstance(mode, str) or mode not in MODES):
        raise ControlError("Unsupported devin_mode; no fallback selected")
    limit = positive_int(settings.get("max_acu_limit", ceiling), "max_acu_limit")
    if limit > ceiling:
        raise ControlError("Issue max_acu_limit exceeds the repository DEVIN_MAX_ACU_LIMIT")
    return {**settings, "max_acu_limit": limit}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ControlError("API redirects are refused; credentials remain on their original host")


class JsonAPI:
    def __init__(self, base, token):
        if not token:
            raise ControlError("Required API credential is missing")
        self.base, self.token = base, token
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, method, path, payload=None, *, missing_ok=False):
        if not path.startswith("/") or path.startswith("//") or ".." in path:
            raise ControlError("Invalid relative API path")
        request = urllib.request.Request(
            self.base + path,
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            method=method,
            headers={"Authorization": "Bearer " + self.token,
                     "Accept": "application/json", "Content-Type": "application/json",
                     "User-Agent": "skillforge-executor-router",
                     "X-GitHub-Api-Version": "2022-11-28"},
        )
        try:
            with self.opener.open(request, timeout=30) as response:
                raw = response.read(4_194_305)
        except urllib.error.HTTPError as exc:
            if missing_ok and method == "GET" and exc.code == 404:
                return None
            raise ControlError(f"API {method} failed with HTTP {exc.code}; writes are not automatically retried") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ControlError(f"API {method} transport failed; a write may have succeeded. Reconcile before retrying") from None
        if len(raw) > 4_194_304:
            raise ControlError("Oversized API response")
        try:
            return json.loads(raw)
        except (ValueError, UnicodeError):
            raise ControlError("Invalid JSON API response; reconcile any pending write") from None


class GitHub(JsonAPI):
    def __init__(self, repo, token):
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
                or any(part in {".", ".."} for part in repo.split("/"))):
            raise ControlError("Invalid repository identity")
        super().__init__(f"https://api.github.com/repos/{repo}", token)
        self.repo = repo

    def issue(self, number):
        if not re.fullmatch(r"[1-9][0-9]*", str(number)):
            raise ControlError("Invalid controlling issue number")
        issue = self.request("GET", f"/issues/{number}")
        if (not isinstance(issue, dict) or issue.get("number") != int(number)
                or "pull_request" in issue or "body" not in issue):
            raise ControlError("Expected the current controlling issue, not a pull request")
        return issue

    def comments(self, number):
        comments = []
        for page in range(1, 101):
            batch = self.request("GET", f"/issues/{number}/comments?per_page=100&page={page}")
            if not isinstance(batch, list) or any(not isinstance(c, dict) for c in batch):
                raise ControlError("Invalid comments response")
            comments.extend(batch)
            if len(batch) < 100:
                return comments
        raise ControlError("Comment pagination limit exceeded; cannot establish unique ownership")

    def save_record(self, number, marker, record, previous=None):
        body = marker + "\n```json\n" + json.dumps(record, sort_keys=True) + "\n```\n"
        path = f"/issues/comments/{previous['id']}" if previous else f"/issues/{number}/comments"
        self.request("PATCH" if previous else "POST", path, {"body": body})
        observed, saved = read_record(self.comments(number), marker)
        if observed is None or saved != record:
            raise ControlError("Control-plane record write was not uniquely confirmed")
        return observed


def workflow_state(issue):
    labels = issue.get("labels")
    if not isinstance(labels, list):
        raise ControlError("Missing issue labels")
    states = [label.get("name") for label in labels if isinstance(label, dict)
              and label.get("name") in STATES]
    if len(states) != 1:
        raise ControlError("Expected exactly one workflow-state label")
    return states[0]


def eligible(issue, wait):
    return (issue.get("state") == "open" and workflow_state(issue) in
            ({"execution-ready", "in-progress"} if wait else {"execution-ready"}))


def read_record(comments, marker):
    matches = [c for c in comments if marker in (c.get("body") or "")]
    if not matches:
        return None, None
    if len(matches) != 1:
        raise ControlError("Duplicate executor/session records; explicit repair required")
    comment = matches[0]
    if (comment.get("user", {}).get("login") != "github-actions[bot]"
            or comment.get("user", {}).get("type") != "Bot"
            or type(comment.get("id")) is not int):
        raise ControlError("Executor/session record is not owned by the Actions dispatcher")
    match = re.fullmatch(re.escape(marker) + r"\n```json\n([^`]+)\n```\n?", comment["body"])
    try:
        data = json.loads(match[1]) if match else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise ControlError("Malformed executor/session record")
    return comment, data


def require_lease(gh, number, selected):
    comment, record = read_record(gh.comments(number), LEASE_MARKER)
    if comment is None or record != {"executor": selected}:
        raise ControlError("Executor ownership changed or is missing; no automatic provider switch")
    return comment


def select_executor(gh, number, wait=False):
    issue = gh.issue(number)
    # A delayed label event is not authorization to release a current hold.
    if not eligible(issue, wait):
        return ""
    selected = executor(issue["body"])
    comments = gh.comments(number)
    comment, lease = read_record(comments, LEASE_MARKER)
    if comment is not None:
        if lease != {"executor": selected}:
            raise ControlError("Executor already claimed; coordinate an idle ownership transfer before switching")
        return selected
    if selected == "devin":
        # Do not adopt pre-router Codex work just because it lacks a lease.
        if (workflow_state(issue) == "in-progress"
                or any("<!-- codex-epic-dag:v1 -->" in (c.get("body") or "") for c in comments)
                or gh.request("GET", f"/git/ref/heads/codex/issue-{number}", missing_ok=True) is not None):
            raise ControlError("Existing execution has no executor lease; explicitly reconcile legacy ownership first")
    gh.save_record(number, LEASE_MARKER, {"executor": selected})
    return selected


def main():
    if sys.argv[1:] != ["select"]:
        raise ControlError("usage: executor_control.py select")
    selected = select_executor(
        GitHub(os.environ["GITHUB_REPOSITORY"], os.environ.get("GITHUB_TOKEN")),
        os.environ["ISSUE_NUMBER"], os.environ.get("WAIT_FOR_EXISTING_TURN") == "true",
    )
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"executor={selected}\n")
    print(f"Executor: {selected or 'not launched (current state is not executable)'}")


if __name__ == "__main__":
    try:
        main()
    except (ControlError, KeyError) as exc:
        raise SystemExit(str(exc)) from None
