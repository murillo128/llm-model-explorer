# Execution runners

Executor selection is independent of model selection. Both implementations now
run **locally on the existing self-hosted machine**: Codex through its shared
App Server, Devin through its installed CLI inside an isolated tmux session.
There is no Devin Cloud API launcher, cloud VM, `/handoff`, or `--cloud` fallback.

The single label entrypoint remains `.github/workflows/codex-issue-state.yml`.
Existing issues without an override still use Codex. The final independent audit
remains a fresh Codex session regardless of the implementation executor.

## Select an executor

Use at most one top-level TOML `execution` fence in the controlling issue body:

````markdown
```execution
executor = "devin"
```
````

An optional separate `devin` fence accepts `model = "<local CLI model identifier>"`.
Use an identifier exposed by `devin models list` for the installed account; the
launcher passes it as one `--model` argument without translating Codex aliases or
adding a fallback. Omit it for native Devin selection, including saved-session
selection on resume. Model availability and any fuzzy-name resolution belong to
the installed CLI; the launcher does not claim to have verified a resulting model.

For Codex, select `executor = "codex"` and keep its existing optional `codex`
model/effort/profile block unchanged. Missing `execution`, an empty fence, or an
omitted `executor` retains Codex. Only exactly `codex` and `devin` are accepted.
Prose, comments, quoted examples, nested fences and parent settings do not select
an executor. Duplicates, malformed TOML and unsupported fields fail closed.

Devin Cloud fields such as `devin_mode` and `max_acu_limit` are no longer accepted.
They are not CLI settings and are never silently ignored or translated. Do not
put credentials, commands, paths, permission modes or arbitrary flags in issues.
An issue cannot disable workspace trust, select dangerous permissions, or redirect
the process to a cloud service through these settings.

Each child and parent selects independently. Mixed Codex/Devin epics retain the
same DAG and parallelism policy; the parent controls only its scheduler turns.
A child-completion wake resolves its canonical parent before selecting that
parent's executor. Selection changes do not activate issues or release holds.

## Physical runner and worktrees

Both execution workflows target `[self-hosted, codex]` because that is the existing
physical runner label. It does not force the implementation to be Codex. No new
runner registration, label rename or service restart is required by this change.
Provision Devin CLI and tmux under the same non-root user as the existing runner.

`SKILLFORGE_REPO_ROOT` must name the persistent coordination clone with the exact
repository origin, outside Actions `_work`. Optional `SKILLFORGE_WORKTREE_ROOT`
and `SKILLFORGE_LOG_ROOT` retain their existing meanings. Never replace the shared
Codex App Server, touch its authentication or use its socket to run Devin.

The Devin launcher performs real Git worktree preparation before starting the
agent. It verifies origin, branch and common Git directory, serializes worktree
creation, and uses the same `codex/issue-N` branch and default
`~/.skillforge/worktrees/<owner>-<repo>/issue-N` path as the existing workflow.
A registered issue worktree is reused, including its uncommitted/untracked work.
It never resets, cleans, prunes worktrees, or substitutes the coordination clone.
An unregistered occupied path or an unowned legacy local branch requires repair.

A new child worktree is created at the canonical activation `base_sha`, after
checking its integration branch and parent. The parser accepts the documented
three-scalar YAML context, including simple quoted scalars and comments; aliases,
object constructors, duplicate/unknown keys and unsupported layouts fail closed.
A reused issue branch is preserved as-is. The execution skill still owns any
required pinned-base reconciliation and final integration freshness before edits
and handoff; the infrastructure does not merge or discard valid issue commits.

Before resume, the saved host/user/clone binding and worktree path must match.
Moving to another host, clone, worktree or executor is an explicit idle migration,
not something that an issue-body edit or a missing state file can authorize.

## Shell, detachment and lifecycle

The trusted default-branch workflow loads launcher scripts by its exact
`GITHUB_SHA`, not from the issue branch. It snapshots the supervisor and prompt
under durable state before starting tmux, so Actions temporary-file cleanup cannot
remove the running program. It does not execute an Actions checkout as product
state and does not execute issue-provided shell snippets.

Each turn uses an isolated tmux server/socket and one issue-named session. The
supervisor runs the installed local CLI in the verified worktree using `--print`,
`--prompt-file` and `--export`; it does not merely background an interactive TUI.
The shell commands executed by Devin therefore run on the local machine, with its
installed tools and available model caches/GPU resources, subject to host policy.
No resources or repository contents are transferred to a cloud VM.

The new tmux server and descendants receive an empty `RUNNER_TRACKING_ID` and no
Actions token, `GH_TOKEN`, `GITHUB_TOKEN`, Actions runtime credentials, CI markers,
Codex environment overrides or personal tmux connection. The detached agent uses
persistent host Git/gh authentication and its own local Devin authentication.
Authentication files are neither copied nor replaced. Keep ephemeral tokens out
of shell startup files too: Devin may read the user's shell configuration.

A local launch lock serializes dispatch; a separate lifetime lock, process identity
and tmux pane checks protect the active turn. A durable `pending` receipt precedes process creation. The supervisor
records its PID, child PID/start identity, CLI exit code, export and final phase.
Unknown/incomplete receipts and lost acknowledgements require reconciliation;
there is no automatic duplicate CLI launch or provider fallback.

Actions exits after local process acknowledgement. **That is not task success.**
The supervisor continues, records output directly to a local log and mirrors it
to tmux, and records the CLI exit even after the job finishes. Print-mode stdin is
closed; background command children cannot hold an output pipe open indefinitely. `remain-on-exit` preserves the finished pane for
inspection. A later authorized turn removes only its verified inactive prior
session and uses a new isolated socket, avoiding stale tmux environments.

The CLI's ATIF export supplies the session ID used by the next explicit `--resume`.
The launcher never uses `--continue`, a latest-session guess, private CLI database
paths or a fabricated session ID. Missing/unsupported exports or a mismatched ID
fail closed. The first real installed-CLI execution must validate this export
contract; fake CLI tests do not prove compatibility with a particular installation.

Ordinary duplicate events do not steer an active turn. Scheduler follow-ups may
wait up to 15 minutes, rechecking current state and ownership, then resume the
same session. Holds cancel the follow-up. A timeout requires another authorized
wake; this change adds no cron or unconditional polling service. SIGTERM/interrupt
terminates the supervised CLI and leaves a reconciliation receipt rather than
claiming success. Host reboot does not preserve processes; worktree/logs/receipts
remain, and interrupted execution requires review before resuming.

## Local prerequisites and permissions

The existing Linux runner user needs Python 3.11+, Git, tmux, an installed Devin
CLI exposing `--print`, `--prompt-file`, `--export`, `--resume` and
`--respect-workspace-trust`, and persistent Git/gh access. `devin auth status` and
`gh auth status --hostname github.com` must succeed without an Actions token.
The launcher checks prerequisites but does not install software or log in.
No `DEVIN_API_KEY`, organization variable or ACU-limit secret is required.

Workspace trust and native command permissions are intentionally preserved:
`--respect-workspace-trust true` is explicit and no bypass/permission-mode flag is
added. The exact issue worktree must be trusted in Devin before non-interactive
execution, and native policies must allow the required commands. A brand-new,
untrusted worktree or an approval requirement may cause the CLI to stop; tmux does
not make print mode interactive or approve commands. Configure trust/allow rules
as the runner user, within the intended scope, before an authorized retry. Never
solve this with a blanket dangerous mode or by switching to the cloud.

Keep the Codex audit runner available. A `codex` block on a Devin-owned issue
configures its independent Codex audit, not the Devin implementation. Shared
historical skill/branch/comment names remain workflow identifiers, not instructions
to invoke a Codex helper from Devin. Provider-specific model/delegation settings
remain scoped to that provider.

## Inspection and recovery

Devin state is under
`~/.skillforge/run/<owner>-<repo>/issue-N/devin/`. `state.json` contains the current
socket/session, session ID when confirmed, worktree, run, PIDs and phase. Each
`run-<Actions-run-id>/` retains its supervisor snapshot, `job.json`, prompt and
ATIF export. Logs are in
`${SKILLFORGE_LOG_ROOT:-~/.skillforge/logs}/<owner>-<repo>/issue-N-<run>-devin.log`.
Keep these artifacts private; they are not uploaded to GitHub automatically.

For a recorded socket `S` and session `issue-N`, `tmux -L S attach -t issue-N`
shows the running or retained output. Detach normally rather than terminating the
session. In print mode this is monitoring, not an approval UI. Read the log and
receipt for the CLI result; GitHub issue/PR state remains the task authority.

The dispatcher owns the existing `skillforge-executor:v1` lease and the new
`skillforge-devin-local:v1` host-binding comment. Agents/schedulers must not edit,
delete or duplicate them. The old `skillforge-devin-session:v1` marker is retained
only as a migration guard: any cloud receipt blocks local launch, including a
pending cloud request. No old session is automatically killed or adopted.

For interrupted/missing-export/failed-trust launches, inspect the receipt, local
log, tmux pane and `devin list --format json` in the exact worktree. Verify whether
a session exists and whether any process remains active before repair. A human
may reconcile the existing local receipt in place only with a verified explicit
session ID and inactive ownership; preserve the prior run evidence. If no CLI
session was ever created, verify that negative result before resetting the initial
launch state/binding. Never delete a pending receipt merely to force another call,
use a last-session guess, or reattach a second Devin process to an active session.

## Validation

```sh
python3 -m unittest discover -s .github/scripts -p 'test_executor_routing.py' -v
python3 -m unittest discover -s .github/scripts -p 'test_codex_profile.py' -v
```

The path-scoped CI includes real temporary Git repositories/worktrees, a fake
local Devin executable, supervisor completion/resume and a required real tmux
transport test. It covers current holds, host/owner identity, legacy cloud/Codex
refusal, native model settings, dirty work preservation, explicit-session resume,
missing exports, bounded waits, credential removal and nonzero CLI exits.
It performs no paid model request and does not validate the user's host setup.

CLI references (checked 2026-09-23):
[commands and flags](https://docs.devin.ai/cli/reference/commands),
[permissions](https://docs.devin.ai/cli/reference/permissions),
[local models](https://docs.devin.ai/cli/models).
