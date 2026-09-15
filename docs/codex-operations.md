# Codex operating policy

This guide covers the CLI and shared App Server on the runner machine.
**Desktop is completely outside this maintenance task's scope.**
It neither changes product specifications nor authorizes execution of an issue.
See [maintenance evidence](./codex-maintenance-evidence.md) for dated observations,
runtime confirmations, validation results, upstream provenance and limitations.

## Scope and installation

Personal reusable templates live in [`config/codex`](../config/codex/).
Installed profiles and agents live under the runner user's effective `CODEX_HOME`;
on the inspected host this is `/home/sergio/.codex`.
The existing global `config.toml` selection, Astra Ultra, remains unchanged.
The repository's [project config](../.codex/config.toml) sets only the auxiliary cap.
It deliberately leaves model and reasoning effort to explicit selections.

Codex 0.154.0 and the official documentation use standalone profile files:
`<CODEX_HOME>/<name>.config.toml`, selected with `--profile <name>`.
Legacy `[profiles.<name>]` tables and the top-level `profile` selector are retired.
See [native profiles](https://learn.chatgpt.com/docs/config-file/config-advanced).

To install on another machine, first identify the actual runner user, CLI binary
and App Server `codexHome`. Compare existing destinations before copying anything.
Do not overwrite an existing profile or role merely because its name matches.
From a reviewed checkout, this example creates missing files with mode `0600`
and refuses to replace differing files; set the destination explicitly:

```sh
CODEX_POLICY_DEST=/absolute/runner/codex-home python3 - <<'PY'
import os
from pathlib import Path

destination = Path(os.environ["CODEX_POLICY_DEST"])
if not destination.is_absolute() or not destination.is_dir():
    raise SystemExit("Select an existing absolute Codex home")
pairs = [(p, destination / p.name)
         for p in Path("config/codex/profiles").glob("*.toml")]
pairs += [(p, destination / "agents" / p.name)
          for p in Path("config/codex/agents").glob("*.toml")]
for source, target in pairs:
    if target.is_symlink():
        raise SystemExit(f"Inspect symlink before installation: {target}")
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise SystemExit(f"Compare existing file before installation: {target}")
for source, target in pairs:
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.exists():
        continue
    with target.open("xb") as output:
        os.chmod(target, 0o600)
        output.write(source.read_bytes())
PY
```

Record newly created paths, SHA-256 values and previous absence in a local
rollback manifest. A partial installation can be resumed after inspecting the
failure. Do not back up credentials, `auth.json`, tokens or complete environments.

## Model choices

| Purpose | Native profile or agent | Model | Effort |
| --- | --- | --- | --- |
| Ordinary development, explicit opt-in | `codex-general` profile | `gpt-6-astra` | `xhigh` |
| Critical owner | `codex-critical` profile | `gpt-6-astra` | `ultra` |
| Audit controller | `codex-review` profile | `gpt-6-astra` | `ultra` |
| Bounded exploration and initial diagnosis | `bounded-explorer` agent | `gpt-5.6-terra` | `medium` |
| Prescribed commands, CI status and logs | `command-runner` agent | `gpt-5.6-luna` | `low` |
| Independent technical reviewer | `independent-reviewer` agent | `gpt-6-astra` | `ultra` |

The owner may choose an available authorized role/model and raise capability
when ambiguity warrants it. Critical semantics, isolation, concurrency, security,
numerical computation and complex integration require Astra Ultra.
Independent review requires Astra Max or Ultra; the installed role selects Ultra.
If a helper is unavailable, disclose an authorized alternative or retain the
high-capability owner. Required critical capability must never silently fall back.

Use these commands for a new CLI session in the intended worktree:

```sh
codex --profile codex-general
codex --profile codex-critical
codex --profile codex-review
codex --profile codex-general -c model_reasoning_effort='"ultra"'
```

Selecting `codex-general` explicitly opts into xhigh; it does not change global
defaults. An explicit higher CLI effort takes precedence over the profile.
The audit profile configures capability; the repository audit/review skills still
determine authority, independence, target selection and publication behavior.

## Precedence and native roles

Configuration precedence, highest first, is CLI overrides, trusted project
configuration, selected profile, user configuration, cloud defaults, system
configuration, then built-in defaults. Project model/effort settings would shadow
profiles, which is why this project's config contains neither.
See [configuration layers](https://learn.chatgpt.com/docs/config-file/config-basic).

On spawn, explicit model/effort settings precede `[agents]` defaults and inherited
parent values. A selected custom agent file then overrides values it defines,
including model and effort. Choose another authorized role to elevate a fixed
helper; do not assume a conflicting spawn override defeats that role's file.
No economical default is configured for all subagents.
See [custom agents](https://learn.chatgpt.com/docs/agent-configuration/subagents).

Use `agent_type` with the agent's `name` and `fork_turns: "none"` for these roles.
V2 exposes `agent_type` when valid custom agents have been discovered; no V1
switch is required ([0.154 source](https://github.com/openai/codex/blob/rust-v0.154.0/codex-rs/core/src/tools/spec_plan.rs#L1306)).
Malformed role files can be warned about and ignored.
Check actual discovery in a new session rather than relying on TOML parsing alone.
The recorded native-role smoke test identified `bounded-explorer` in the child
runtime source and confirmed Terra medium; model self-description is not evidence.
The native CLI profile smoke test completed with `--strict-config`, read-only
sandboxing and no model tool use. The actual trusted maintenance worktree loaded
the project cap; an untrusted isolated project correctly skipped project config.

## Ownership, permissions and concurrency

Each profile and the project config request
`agents.max_concurrent_threads_per_session = 3`: at most three spawned auxiliary
threads, excluding the primary. `agents.max_threads` is a legacy alias.
This limit is per session, not a host-wide resource budget or permission for
multiple product implementers. Retain one product change owner.

All three custom roles request read-only sandboxing. The live inheritance test
confirmed that an explicit parent workspace-write policy takes precedence and
leaves the child workspace-writable. A role file is therefore not an independent
write boundary; use a separate read-only session/snapshot when enforcement is
required, and keep generated output in an explicitly authorized temporary area.
Managed requirements also apply. Do not weaken sandboxing, approvals or network
controls when a helper cannot perform a command.

The owner designs invariants, independent oracles and test expectations, and
decides whether computation is preserved. Helpers run prescribed checks and
report failures; they cannot edit source, assertions, expected results or config,
suppress failures, reinterpret specifications, mutate Git/GitHub, or delegate.
Pass prescribed commands as fenced, exact command lines, with the working
directory stated separately; inspect the actual executed command in the result.
An auxiliary needs a stable isolated snapshot, its exact SHA, an authorized
temporary output directory and separate ports/caches where relevant.
Coordinate with other host users and run at most one heavy suite at a time.

The final independent reviewer starts with fresh context and has not participated
in implementation. An implementation helper does not replace that reviewer.
Keep the existing final audit boundary; do not add a duplicate final audit.

## Automatic execution and audit

Execution defaults to `codex-critical`; audit defaults to `codex-review`.
The repository variables `CODEX_EXECUTION_PROFILE` and `CODEX_AUDIT_PROFILE`
permit explicit selection. Use `CODEX_EXECUTION_PROFILE=codex-general` only for
ordinary work; restore critical before work requiring Ultra. No issue-number
conditions are embedded in the launcher.

App Server's installed start/resume schema has configuration overrides but no
native configuration-profile selector. The small
[`codex_profile.py`](../.github/scripts/codex_profile.py) adapter reads the profile
from the server-reported home and forwards only model, effort and the cap.
This file-to-protocol adaptation is repository integration, not a claimed native
server profile feature. Existing launcher permission choices remain authoritative.
See the [App Server protocol](https://learn.chatgpt.com/docs/app-server).

The adapter checks the live paginated model catalog and rejects unavailable or
insufficient capability before a turn. It confirms start/resume model and effort,
then supplies that confirmed selection to `turn/start`. Structured launch evidence
records requested/confirmed settings, role, source file, worktree and thread ID.
The cap is logged as requested; that log alone does not establish enforcement.

The two launch workflows and audit-client extraction script have small local
changes relative to their pinned Skillforge source blobs; provenance is recorded
in the evidence document. The adapter is extracted from the published workflow
commit, so an implementation worktree cannot replace it. No generic skill is
forked. Shared-server use, duplicate guards, serialization, issue worktrees,
dispatcher routing and workflow transitions remain intact.

## Existing sessions and safe adoption

Treat new, resumed and currently loaded sessions separately. An inactive loaded
thread was observed to ignore model/config values supplied to `thread/resume`.
An explicit `turn/start` model/effort selection on an isolated inactive test
thread took effect and was confirmed by a later resume response.

Automatic resume preserves the saved model/effort, accepts an already higher
authorized selection, and refuses insufficient capability. It does not silently
replace the selection. For an older thread, arrange an idle window, explicitly
adopt the desired model/effort, inspect runtime confirmation, then retry.
Prefer a new session when a reliable cap and complete new configuration are needed:
an already-loaded thread may retain its earlier cap despite resume overrides.

No atomic compare-and-set operation protects idle-check-to-launch across clients.
Existing duplicate guards reduce overlap but cannot prove that another client
will not start the same thread. Do not use a thread another actor may be steering;
coordinate ownership and an idle window. Never reconfigure an active turn.

## Version maintenance and rollback

The inspected standalone CLI and shared App Server report 0.154.0; this was the
latest stable CLI in the official changelog on 2026-09-15. No update was needed.
Retained standalone release directories are rollback versions, not proof of
multiple active installations. Record both the CLI's resolved path and the binary
of the serving process; a newer PATH binary does not update a running server.
See [release notes](https://learn.chatgpt.com/docs/changelog).

Check for updates between important executions, not before every issue. During
a coordinated window with all affected sessions idle, preserve the previous
standalone `current` symlink target, release directory and configuration manifest,
then use the existing standalone `codex update` mechanism if a stable update is
appropriate. Do not mix package managers, change channels or install prereleases.
Do not add cron, a daemon or a scheduler for maintenance.

The existing user unit `codex-remote-control.service` bootstraps the shared server;
its observed `Type=oneshot`/`RemainAfterExit` state was `active (exited)` while the
server process continued running. Updating
the CLI and activating a server version are separate operations. This task does
not restart or kill that service; plan activation in the same coordinated window
and verify its new binary/version afterward. Desktop remains outside scope.

For rollback, restore the recorded standalone symlink target while preserving
both releases, then activate the previous server version only in an idle window.
Remove only files this installation created whose SHA-256 still matches the
manifest; leave subsequently edited files for comparison. Restore overwritten
files only from their recorded backups. The global user config was not rewritten.
Revert repository changes through a separate reviewed change if already merged.

## Activation hold

The maintenance PR prepares future launches; merging or activating it is outside
this task's authorization. Issue #128 remains `blocked` under the user's manual hold,
without `review-ready`, which would trigger the audit controller's merge path.
The observed holds on #124 and #125 remain in place. Preserve #124's
`max_parallel_workers: 1`; internal helpers do not change epic scheduling.
The evidence snapshot records #114 as `in-progress`; re-read current labels before
later work. Do not release blockers, activate epics or interrupt other executors.
