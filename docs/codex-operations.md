# Codex operating policy

This guide covers the CLI and shared App Server on the runner machine.
Desktop installation and configuration are outside this maintenance scope. Runtime settings do not authorize
issue execution, release holds or replace the existing audit/ownership rules.
See [the original maintenance evidence](./codex-maintenance-evidence.md) for the
0.154.0 host observations and limitations. That dated record predates the optional
issue-override policy below; its mandatory launcher defaults are superseded.

## Default behavior: native local configuration

No issue or epic needs a model, reasoning effort, profile file or repository
variable. With no override, execution and audit leave model/effort unset in
`thread/start` and `turn/start`; the serving Codex installation resolves its normal
local configuration for that worktree. There is no implicit `codex-critical`,
`codex-review`, Astra-only restriction or Ultra minimum in the launcher.

Normal `thread/resume` retains the saved session selection, just as native CLI
resume does. It is not rejected because it differs from today's local default or
an old hardcoded profile. Changing local defaults does not retroactively change a
loaded session. An explicit issue override changes the next permitted turn of an
inactive session; it never steers an active turn. Omitted fields stay native.

“Local CLI configuration” means the user/project configuration seen by the shared
App Server, including its effective `CODEX_HOME` and trusted worktree layers. It
cannot inherit transient flags from another terminal or another user's CLI.
No user config files are rewritten, and no server restart is performed.

## Optional per-issue or per-epic override

Put at most one top-level fenced `codex` block in the **controlling issue body**.
Its contents are TOML. For example, explicitly choose the model and effort:

````markdown
```codex
model = "gpt-6-astra"
model_reasoning_effort = "ultra"
```
````

Alternatively, select an installed local profile; explicit fields in the same
block take precedence over that profile's model/effort:

````markdown
```codex
profile = "codex-general"
model_reasoning_effort = "ultra"
```
````

All three keys are optional. No block or an empty block means native behavior.
An effort-only override retains the model; a model-only override retains the
native effort. The resulting explicit model/effort pair must be supported by the
live catalog. Supply both fields when changing model would otherwise retain an
incompatible effort. An explicit lower effort is an intentional override, not a
silent downgrade; absent overrides never lower saved settings.

The block configures execution and fresh audit of that controlling issue. On an
epic parent it configures the scheduler's own session. Children are independent
controlling issues: there is no implicit inheritance from parent prose, links or
comments. Give a child its own block only when it needs an override. This avoids
silently changing other running tasks or inventing scheduling metadata.

Only this dedicated fence is parsed, not model names in prose, quotes or examples
nested inside other fences. Duplicate or unterminated blocks, invalid TOML,
unknown keys, non-string/empty values and unsafe profile names fail explicitly.
Settings cannot alter permissions, commands, authentication, network policy,
paths or agent instructions. Explicit unavailable models/efforts or missing
profiles fail before a model turn; they do not silently fall back to local values.

The old `CODEX_EXECUTION_PROFILE` and `CODEX_AUDIT_PROFILE` repository variables are
no longer consulted. They are not a hidden default between an issue and local
configuration. Removing the issue block returns to native behavior; if a resumed
session already saved a previous explicit selection, native resume retains it.

## Optional reusable profiles and agents

Templates live in [`config/codex`](../config/codex/). Personal copies belong under
the runner user's effective `CODEX_HOME`; the inspected host reported
`/home/sergio/.codex`. Native standalone profiles are
`<CODEX_HOME>/<name>.config.toml`, selected interactively with `--profile <name>`.
They are conveniences, not prerequisites for automatic launches.

| Purpose | Optional profile or agent | Model | Effort |
| --- | --- | --- | --- |
| Ordinary development | `codex-general` profile | `gpt-6-astra` | `xhigh` |
| Critical owner | `codex-critical` profile | `gpt-6-astra` | `ultra` |
| Audit controller | `codex-review` profile | `gpt-6-astra` | `ultra` |
| Bounded exploration | `bounded-explorer` agent | `gpt-5.6-terra` | `medium` |
| Prescribed commands and CI/logs | `command-runner` agent | `gpt-5.6-luna` | `low` |
| Independent technical reviewer | `independent-reviewer` agent | `gpt-6-astra` | `ultra` |

These are recommended opt-in roles, not a reason to reject an unannotated issue.
The owner may choose available roles/models and escalate ambiguous work. The
independent reviewer must still have fresh context and sufficient capability;
an implementation helper is not the final reviewer. The installed custom
reviewer remains Ultra, independently of the audit controller's native default.

Before installing on another host, identify the actual runner user, CLI binary
and server home, and compare any existing destination. Do not overwrite differing
files or follow an unexpected symlink. Create missing profile/agent files with
mode 0600 under the selected home and its `agents/` directory. Retain a local
manifest of paths, previous absence/backup and SHA-256 values. Never copy or back
up `auth.json`, credentials or complete environments as configuration evidence.

Examples for explicit new CLI sessions:

```sh
codex --profile codex-general
codex --profile codex-critical
codex --profile codex-review
```

The server adapter deliberately forwards only profile model/effort and the
existing auxiliary cap. A selected launcher profile may contain only `model`,
`model_reasoning_effort`, and optional `[agents]` with
`max_concurrent_threads_per_session = 3`. Other native profile features are not
implicitly granted by selecting a profile in an issue. The adapter does not
claim to be a general implementation of native profile/config layering.

## Precedence, delegation and permissions

Let Codex own user/project configuration resolution instead of hand-parsing the
global `config.toml`. The repository `.codex/config.toml` requests only auxiliary
cap 3, not a model. Explicit issue model/effort override optional profile values;
unspecified values remain native. For a loaded inactive thread, overrides are
sent through `turn/start` rather than assuming `thread/resume` changed its saved
model or effort. A selected custom agent file may override spawn settings for
values it defines: choose another authorized role to elevate a fixed helper.

Use `agent_type` with the custom role name and `fork_turns: "none"`. Verify native
role discovery in a new session; TOML parsing alone does not prove discovery.
The original evidence demonstrates role precedence and the three-child limit on
new sessions. The cap is per session, not a host-wide budget or permission for
several product implementers. Keep one product change owner and respect the
epic's independent `max_parallel_workers` and other host users.

Custom roles request read-only sandboxing, but observed parent workspace-write
permissions can prevail. A role file is not an enforced write barrier. Use an
isolated stable snapshot or a separate read-only session when enforcement is
needed. Generated results belong only in authorized temporary locations with
isolated ports/caches. Do not weaken sandbox, approvals or network controls.

The owner designs invariants, independent oracles and expectations and judges
correctness. Helpers run prescribed commands and report exact command, snapshot
SHA, exit status and relevant evidence. They cannot edit source, expectations or
config, suppress failures, reinterpret specs, mutate Git/GitHub or delegate.
Pass commands fenced and exact, with cwd separately; inspect the executed command.
Run at most one heavy suite at a time on the shared host. Preserve the existing
final independent audit rather than adding a duplicate review.

## Launcher implementation and observability

Both workflows fetch the current controlling issue body with short-lived
Actions authentication before detaching. Only the validated settings dictionary
is written atomically to a mode-0600 snapshot; neither the issue body nor token
is retained there. A failed issue read is an error, not “no override.” The
detached client still removes Actions credentials and uses the shared App Server.
The helper and audit generator come from the trusted published workflow commit,
not the implementation worktree. The dispatcher, worktrees, ownership and labels
are unchanged; no issues are activated or retried by a configuration change.

`model_policy_resolved` logs mode (`native` or `issue-override`), optional profile,
observed model/effort, intended turn overrides, thread and worktree. A resumed
thread's observed selection is the **old** selection, not proof that its next
turn applied an override. Null effort stays null; no confirmation is fabricated.
New explicit thread settings must match the returned selection. Explicit targets
are checked against the paginated live model catalog before starting a turn.
Native defaults remain Codex-owned and do not require the hardcoded catalog of a
particular model family. Runtime errors still fail normally.

The cap is logged as requested, not confirmed for old loaded threads. Such a
thread may ignore a new cap supplied on resume; use a coordinated idle window and
verified new session when full adoption is required. No atomic cross-client lock
exists between idle checking and turn start. Keep the existing duplicate guards,
coordinate thread ownership, and never steer a thread another actor is using.
Audit uses a fresh detached worktree/thread; explicit settings do not grant
reviewers permission to implement or change the reviewed head.

## Desktop project assignment

Execution, epic-scheduler resumes and fresh audit threads use the same project
assignment code. The launcher discovers existing projects with experimental
`project/list`, matching an exact canonical root to the verified persistent
repository checkout (`SKILLFORGE_REPO_ROOT_RESOLVED`), not the issue/review
worktree, project name or a path prefix. It follows bounded pagination and only
uses a unique match; it never creates or edits projects or guesses by name.

Discovery happens before start/resume. The resulting fresh thread snapshot must
explicitly report `projectId: null` before `thread/metadata/update` assigns the
project, immediately before `turn/start`. Existing non-null assignments, including
manual Desktop moves, are preserved. Missing `projectId` is unknown, not permission
to overwrite it. Assignment does not change `cwd`, environments, writable roots,
models, permissions, branches or audit isolation. It does not resume audit history,
migrate old sessions in bulk, restart the server or interrupt active work. Existing
unassigned execution threads are handled only on their next permitted resume.
As with turn launch, the protocol provides no atomic cross-client lock: avoid
concurrent manual metadata edits during the small snapshot-to-update window.

`thread_project_assigned` is logged only when the metadata response confirms the
same thread and requested project. `thread_project_preserved` records an existing
assignment. Missing/ambiguous matches, unsupported experimental RPCs or fields,
invalid pagination and unconfirmed writes produce `thread_project_warning` in the
local event log and continue without retrying thread creation. Transport failures
and core thread/turn failures remain errors. The target project must already exist
on the serving App Server with the repository root; offline tests do not establish
that the runner version exposes the API or that Desktop displays the association.

## Validation and activation

The optional-settings regression suite exercises actual executor and generated
audit lifecycle code with an offline server double, plus issue parsing and
credential-separated settings preparation:

```sh
python3 -m unittest discover -s .github/scripts -p 'test_codex_profile.py' -v
```

It covers native launches with no profile files, legacy/null-effort resumes,
inline/profile overrides on new and inactive sessions, profile precedence,
malformed/unavailable choices, fresh audit and active-turn guards, bounded
catalog handling, and settings-only preparation. Project regressions cover new and
resumed execution, fresh audits, canonical root matching, manual assignments,
bounded/ambiguous discovery, unsupported APIs and confirmation of metadata writes.
Offline tests are not a claim of live-host validation. Check both new and resumed behavior on an isolated idle
smoke thread before claiming runtime adoption; do not use real issue transitions,
interrupt manual #114 or release #124/#125 as a smoke test.

## Version maintenance and rollback

The retained host evidence reports CLI and shared server 0.154.0 with no update
or restart. A newer PATH binary does not replace a running server. Record the
serving executable, version and home independently from the interactive CLI.
Check stable updates between important executions, not before each issue; do not
mix installers, change channels, or create a cron/daemon/scheduler for upgrades.

During a coordinated idle window, preserve the prior standalone `current`
symlink target, release directory and configuration manifest, then use the
installation's supported update mechanism when appropriate. Activating the
server version is a separate operation requiring that same safe window. The
observed `codex-remote-control.service` is a bootstrap unit: its `active (exited)`
state does not prove that the server stopped. Never kill/restart it to clear a
model-profile mismatch.

Rollback restores only the recorded, unchanged configuration files or previous
release target; preserve later user edits for comparison. Revert repository
changes with a reviewed commit, not a reset of shared history. Old evidence of
issue holds is historical: re-read current state before future operations.

References: [App Server](https://developers.openai.com/codex/app-server/),
[native configuration](https://developers.openai.com/codex/config-advanced/),
[custom agents](https://developers.openai.com/codex/subagents/).
