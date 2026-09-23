"""Local Devin CLI turns in isolated tmux sessions, not Devin Cloud.

Actions acknowledges a detached local supervisor. Durable receipts, explicit
session IDs and real Git worktrees survive the job; no Actions token does.
"""

from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from executor_control import (ControlError, GitHub, SESSION_MARKER, devin_settings,
                              eligible, executor, read_record, require_lease)
from local_issue_worktree import (context, durable_path, git, lock, prepare, run,
                                 verify_repo, verify_worktree)

LOCAL_MARKER = "<!-- skillforge-devin-local:v1 -->"
TERMINAL = {"finished", "failed"}


def clean_env(source):
    denied = {"GH_TOKEN", "GITHUB_TOKEN", "CI", "GITHUB_ACTIONS", "TMUX", "TMUX_PANE",
              "GIT_ASKPASS", "SSH_ASKPASS", "DEVIN_API_KEY", "DEVIN_ORG_ID", "DEVIN_MAX_ACU_LIMIT"}
    result = {k: v for k, v in source.items() if k not in denied and
              not k.startswith(("ACTIONS_", "RUNNER_", "CODEX_", "SKILLFORGE_CODEX_", "GIT_CONFIG_"))}
    result["RUNNER_TRACKING_ID"] = ""
    return result


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def read_json(path, *, optional=False):
    path = Path(path)
    if optional and not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        raise ControlError(f"Unreadable local receipt: {path.name}; reconcile before retrying") from None
    if not isinstance(value, dict):
        raise ControlError("Local receipt must be an object")
    return value


def session_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,159}", value):
        raise ControlError("Missing or invalid exported Devin session ID; never resume the latest session")
    return value


def export_id(path, expected=None):
    # CLI --export is ATIF; unsupported/missing schemas require reconciliation,
    # not a guessed private database location or a 'latest session' fallback.
    value = read_json(path)
    sid = session_id(value.get("session_id"))
    if expected is not None and sid != expected:
        raise ControlError("Devin exported a different session than the requested resume")
    return sid


def process_identity(pid):
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except (OSError, ValueError, IndexError, TypeError):
        return None


def tmux_args(socket, *args):
    return ["tmux", "-L", socket, "-f", "/dev/null", *args]


def active(state_dir, record, env):
    try:
        with lock(state_dir / "turn.lock", blocking=False):
            pass
    except BlockingIOError:
        return True
    if not record:
        return False
    for name in ("worker", "child"):
        pid, start = record.get(name + "_pid"), record.get(name + "_start")
        if pid and start and process_identity(pid) == start:
            return True
    panes = run(tmux_args(record["socket"], "list-panes", "-t", "=" + record["session"] + ":0",
                          "-F", "#{pane_dead}"), env=env, check=False)
    return panes.returncode == 0 and any(line == "0" for line in panes.stdout.splitlines())


def cli_command(binary, job):
    command = [binary, "--print", "--prompt-file", job["prompt"], "--export", job["export"],
               "--respect-workspace-trust", "true"]
    if job.get("session_id"):
        command += ["--resume", session_id(job["session_id"])]
    if "model" in job["settings"]:
        command += ["--model", job["settings"]["model"]]
    return command


def prompt(repo, number):
    return f"""Execute https://github.com/{repo}/issues/{number} using local Devin CLI.
You are in the prepared persistent issue worktree on this host, not Devin Cloud.
Read AGENTS.md and skills/execution-runner-selection/SKILL.md, then the live
controlling issue and its relevant top-level comments using persistent host auth.
Respect SKILLFORGE_LOCAL_RUNNER=1, SKILLFORGE_ISSUE_WORKTREE and
SKILLFORGE_ISSUE_BRANCH. Do not create another worktree or change to the durable
coordination clone. Preserve unfinished changes on resume. Re-read current state
and exclusive ownership before editing. Dispatcher-owned executor/local-host and
session receipts are read-only to you. Never modify, delete or duplicate them.
If execution_mode is epic-dag use skills/codex-epic-scheduler/SKILL.md; otherwise
use skills/spec-driven-codex-loop/SKILL.md. Verify canonical
codex-execution-context:v1, pinned base and PR target before implementation edits;
reconcile a reused branch according to that skill, never reset valid issue work.
Only the scheduler activates queued children; preserve dependencies, holds and
max_parallel_workers. Parent executor/model settings do not propagate to children.
Use only local tools and persistent host Git/gh credentials. Do not use /handoff,
--cloud, a cloud VM, the Codex App Server, or another issue's session or worktree.
Never recover Actions tokens, change host authentication or bypass approvals.
Finish at a ready PR and review-ready handoff to the independent Codex audit.
Never merge, enable auto-merge, close issues, mark completed, or mutate GitHub
after review-ready. A successful CLI exit is not task completion or test evidence.
Report concrete missing tools/auth/model/test infrastructure; do not switch executor.
"""


def worker(job_path):
    job = read_json(job_path)
    state_path = Path(job["state"])
    env = clean_env(os.environ)
    with lock(state_path.parent / "turn.lock", blocking=False):
        record = read_json(state_path)
        if record.get("run_id") != job["run_id"] or record.get("phase") != "pending":
            raise ControlError("Turn receipt is not the expected pending launch")
        child = None
        try:
            verify_worktree(Path(job["root"]), job["worktree"], job["repo"], job["branch"])
            record.update(worker_pid=os.getpid(), worker_start=process_identity(os.getpid()))
            # Receipt is already pending before Popen; no retry after uncertain launch.
            atomic_json(state_path, record)
            env.update(SKILLFORGE_LOCAL_RUNNER="1", SKILLFORGE_EXECUTOR="devin",
                       SKILLFORGE_ISSUE_WORKTREE=job["worktree"],
                       SKILLFORGE_ISSUE_BRANCH=job["branch"],
                       SKILLFORGE_REPO_ROOT_RESOLVED=job["root"],
                       GITHUB_REPOSITORY=job["repo"], ISSUE_NUMBER=job["number"])
            with open(job["log"], "ab", buffering=0) as output, open(job["log"], "rb") as mirror:
                mirror.seek(0, os.SEEK_END)
                # A log file, not a pipe/PTY, prevents background command children
                # from holding the supervisor's output stream open after CLI exit.
                child = subprocess.Popen(cli_command(job["binary"], job), cwd=job["worktree"],
                                         env=env, stdin=subprocess.DEVNULL,
                                         stdout=output, stderr=subprocess.STDOUT)
                def interrupt(signum, frame):
                    if child.poll() is None:
                        child.terminate()
                    raise ControlError("Local supervisor interrupted; reconcile the session before retrying")
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                    signal.signal(sig, interrupt)
                record.update(phase="running", child_pid=child.pid,
                              child_start=process_identity(child.pid))
                atomic_json(state_path, record)
                while True:
                    data = mirror.read(65536)
                    if data:
                        # tmux scrollback is convenience; the local log is durable.
                        with suppress(BrokenPipeError, OSError):
                            os.write(sys.stdout.fileno(), data)
                    elif child.poll() is not None:
                        break
                    else:
                        time.sleep(0.1)
                rc = child.wait()
                record["exit_code"] = rc
                record["session_id"] = export_id(job["export"], job.get("session_id"))
                record["phase"] = "finished" if rc == 0 else "failed"
        except BaseException as exc:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            record.update(phase="needs-reconciliation", error=type(exc).__name__)
            atomic_json(state_path, record)
            raise
        atomic_json(state_path, record)
        return record["exit_code"]


def launch(gh, number, root, worktree_base, state_dir, log_dir, default_branch, run_id,
           *, wait=False, sleep=time.sleep, clock=time.monotonic, wait_seconds=900):
    if not re.fullmatch(r"[1-9][0-9]*", str(number)) or not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ControlError("Invalid issue number or Actions run ID")
    number = str(number)
    root = verify_repo(root, gh.repo)
    env = clean_env(os.environ)
    binary = shutil.which("devin", path=env.get("PATH"))
    if not binary or not shutil.which("tmux", path=env.get("PATH")):
        raise ControlError("Install/authenticate local Devin CLI and tmux for the runner user first")
    help_text = run([binary, "--help"], env=env).stdout
    for flag in ("--print", "--prompt-file", "--export", "--resume", "--respect-workspace-trust"):
        if flag not in help_text:
            raise ControlError(f"Installed Devin CLI lacks required capability {flag}")
    run([binary, "auth", "status"], env=env)
    # Persistently configured gh must work without the short-lived Actions token.
    run(["gh", "auth", "status", "--hostname", "github.com"], env=env)
    state_dir, log_dir = durable_path(state_dir), durable_path(log_dir)
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    machine = Path("/etc/machine-id").read_text().strip()
    host = hashlib.sha256(f"{machine}:{os.getuid()}:{root}".encode()).hexdigest()
    binding = {"transport": "cli-tmux", "host": host}
    state_path = state_dir / "state.json"

    def current():
        issue = gh.issue(number)
        if not eligible(issue, wait):
            return None
        if executor(issue["body"]) != "devin":
            raise ControlError("Executor selection changed before local launch")
        require_lease(gh, number, "devin")
        comments = gh.comments(number)
        if any(SESSION_MARKER in (c.get("body") or "") for c in comments):
            raise ControlError("Legacy cloud session receipt exists; explicit migration required")
        _, owner = read_record(comments, LOCAL_MARKER)
        if owner is not None and owner != binding:
            raise ControlError("Issue is bound to another local host/user/clone; do not start a replacement")
        return devin_settings(issue["body"]), context(comments, number), owner

    with lock(state_dir / "launch.lock", blocking=False):
        selected = current()
        if selected is None:
            return {"result": "skipped-current-state"}
        record = read_json(state_path, optional=True)
        if record and (record.get("repo") != gh.repo or record.get("number") != number
                       or record.get("host") != host):
            raise ControlError("Local receipt belongs to a different issue/host")
        deadline = clock() + wait_seconds
        while active(state_dir, record, env):
            if not wait:
                return {"result": "already-active"}
            if clock() >= deadline:
                raise ControlError("Active Devin turn exceeded bounded scheduler wait; retry later")
            sleep(10)
            selected = current()
            if selected is None:
                return {"result": "skipped-current-state"}
            record = read_json(state_path, optional=True)
        if record and record.get("phase") not in TERMINAL:
            raise ControlError("Incomplete local receipt; reconcile logs/session before retrying")
        if record and record.get("run_id") == run_id:
            if record["phase"] == "failed":
                raise ControlError("This Actions run already ended with CLI failure; inspect local logs")
            return {"result": "already-acknowledged"}
        if selected[2] is not None and not record:
            raise ControlError("Local session receipt is missing on the bound host; do not create another session")
        # Never adopt a pre-existing Codex session as local Devin work.
        codex_state = state_dir.parent
        if any((codex_state / name).exists() for name in
               ("app-server-thread-id", "app-server-client.pid", "codex.pid")):
            raise ControlError("Codex session state exists; explicit idle ownership transfer required")
        settings, activation, owner = selected
        if record is None and git(root, "show-ref", "--verify", "--quiet",
                                  f"refs/heads/codex/issue-{number}", check=False).returncode == 0:
            raise ControlError("Unowned local issue branch exists; explicitly reconcile legacy work")
        if activation:
            gh.issue(activation["epic_issue"])
        worktree = prepare(root, worktree_base, gh.repo, number, default_branch, activation)
        if record and record.get("worktree") != str(worktree):
            raise ControlError("Recorded session worktree moved; explicit idle migration required")
        refreshed = current()
        if refreshed is None:
            return {"result": "skipped-current-state"}
        if refreshed != selected:
            raise ControlError("Issue settings/context/ownership changed during worktree preparation")
        turn_dir = state_dir / f"run-{run_id}"
        # Exclusive mkdir refuses replay of even a partially written launch bundle.
        turn_dir.mkdir(mode=0o700)
        for name in ("devin_runner.py", "executor_control.py", "local_issue_worktree.py"):
            shutil.copyfile(Path(__file__).with_name(name), turn_dir / name)
        (turn_dir / "prompt.txt").write_text(prompt(gh.repo, number))
        socket = "sf-devin-" + hashlib.sha256(f"{gh.repo}:{number}".encode()).hexdigest()[:20] + "-" + run_id
        session = f"issue-{number}"
        if record:
            session_id(record.get("session_id"))
            # Only our verified inactive pane, never a user's tmux server.
            run(tmux_args(record["socket"], "kill-session", "-t", "=" + record["session"]), env=env, check=False)
        pending = {"phase": "pending", "run_id": run_id, "repo": gh.repo, "number": number,
                   "host": host, "socket": socket, "session": session,
                   "worktree": str(worktree), "settings": settings}
        if record:
            pending["session_id"] = record["session_id"]
        job = {**pending, "state": str(state_path), "root": str(root),
               "branch": f"codex/issue-{number}", "binary": binary,
               "prompt": str(turn_dir / "prompt.txt"), "export": str(turn_dir / "conversation.json"),
               "log": str(log_dir / f"issue-{number}-{run_id}-devin.log")}
        atomic_json(turn_dir / "job.json", job)
        atomic_json(state_path, pending)
        if owner is None:
            gh.save_record(number, LOCAL_MARKER, binding)
        # The dedicated tmux server and all descendants receive sanitized env.
        # No shell interpolation of issue prose; prompt is a private local file.
        command = shlex.join([sys.executable, str(turn_dir / "devin_runner.py"),
                              "worker", str(turn_dir / "job.json")])
        run(tmux_args(socket, "new-session", "-d", "-s", session, "-c", str(worktree),
                      command, ";", "set-option", "-w", "-t", "=" + session + ":0",
                      "remain-on-exit", "on"), env=env)
        for _ in range(100):
            observed = read_json(state_path)
            if observed.get("phase") in {"running", "finished"}:
                return {"result": "launched-local-cli", "socket": socket, "session": session,
                        "log": job["log"], "state": str(state_path)}
            if observed.get("phase") in {"failed", "needs-reconciliation"}:
                raise ControlError("Devin CLI failed to start/finish; inspect local receipt and log")
            sleep(0.2)
        raise ControlError("Local launch acknowledgement timed out; reconcile before retrying")


def main():
    os.umask(0o077)
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        return worker(sys.argv[2])
    if sys.argv[1:] != ["launch"]:
        raise ControlError("usage: devin_runner.py launch | worker JOB_JSON")
    if os.environ.get("GITHUB_REF") != "refs/heads/" + os.environ["DEFAULT_BRANCH"]:
        raise ControlError("Local Devin launches require the trusted default-branch workflow")
    repo, number = os.environ["GITHUB_REPOSITORY"], os.environ["ISSUE_NUMBER"]
    if not re.fullmatch(r"[1-9][0-9]*", number):
        raise ControlError("Invalid controlling issue number")
    gh = GitHub(repo, os.environ.get("GITHUB_TOKEN"))
    repo_key = repo.replace("/", "-")
    home = Path.home()
    result = launch(gh, number, os.environ["SKILLFORGE_REPO_ROOT"],
                    os.environ.get("SKILLFORGE_WORKTREE_ROOT", str(home / ".skillforge/worktrees")),
                    home / ".skillforge/run" / repo_key / f"issue-{number}" / "devin",
                    Path(os.environ.get("SKILLFORGE_LOG_ROOT", str(home / ".skillforge/logs"))) / repo_key,
                    os.environ["DEFAULT_BRANCH"], os.environ["GITHUB_RUN_ID"],
                    wait=os.environ.get("WAIT_FOR_EXISTING_TURN") == "true")
    print(json.dumps(result, sort_keys=True))
    # No private host paths/log contents in public Actions summaries.
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as output:
        output.write(f"Local Devin CLI: **{result['result']}**. Launch is not task completion.\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ControlError, KeyError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"Local Devin runner stopped: {type(exc).__name__}: {exc}") from None
