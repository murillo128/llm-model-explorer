# Codex maintenance evidence — 2026-09-15

Controlling issue: [#128](https://github.com/murillo128/llm-model-explorer/issues/128).
This is maintenance preparation with a manual activation/merge hold. The operating
procedure is in [Codex operations](codex-operations.md). Compact runtime metadata
and native tool results are in [the evidence JSON](evidence/codex-2026-09-15.json).
No model self-description is used to establish model identity.

## Before and after inventory

| Surface | Before | After |
| --- | --- | --- |
| PATH CLI | Standalone `codex-cli 0.154.0` | Same binary/version |
| CLI origin | `~/.local/bin/codex` → `~/.codex/packages/standalone/current/bin/codex` | Unchanged |
| Active standalone release | `0.154.0-x86_64-unknown-linux-musl` | Unchanged |
| Shared App Server | PID 5244; executable resolves to that same release | Same PID/binary; no restart |
| App Server home | `initialize.codexHome = /home/sergio/.codex` | Unchanged |
| CLI home | `CODEX_HOME` unset; user home resolves to `~/.codex` | Unchanged |
| User default | `gpt-6-astra`, `model_reasoning_effort = "ultra"` | Original config bytes restored/verified |
| Personal profiles/roles | None | Three opt-in profiles and three native agents |
| Project config | Absent | Cap 3 in this maintenance branch only |
| Runner account | `sergio`; four repository runners online/busy | Services/env unchanged |
| Version activation | Current version already latest documented stable | No update prepared or required |

The runner worker environment identifies the durable clone as
`/usr/local/src/llm-model-explorer`, persistent worktrees under
`~/.skillforge/worktrees`, and the shared socket at
`~/.codex/app-server-control/app-server-control.sock`. Execution and audit join that
server; the CLI on PATH is not a replacement server for launcher requests.
`initialize.userAgent` also reports version 0.154.0. The existing enabled
`codex-remote-control.service` bootstraps the daemon; a proxy and daemon PID updater
were also observed. Several subprocesses use the same executable.

Eight older standalone releases remain (0.146.0, 0.149.1, 0.152.0, 0.153.0–0.153.4).
They are retained rollback directories, not eight active installations. Filtered
npm/dpkg/snap/flatpak checks found no second package-managed Codex installation.
No Codex-specific user timer or repository update mechanism was found. Desktop
was explicitly removed from scope by the user.

Only named non-secret environment keys and configuration fields were inspected.
No credentials, `auth.json`, tokens or environment dumps were accessed. Native
workspace-write smoke creation added trust for its own temporary fixture; that
single entry was removed and the original user-config SHA-256 restored exactly.
No existing application session received a maintenance start/resume/steer request.

## Native profiles, models and precedence

The actual server catalog advertised `gpt-6-astra` and `gpt-5.6-terra` with
`low`, `medium`, `high`, `xhigh`, `max`, `ultra`; Luna advertised those through `max`.
The installed generated protocol represents effort as a nonempty string. Its
`model/list` catalog, not a selector display label, establishes literal `ultra`.

Installed opt-in profiles are `codex-general` (Astra xhigh), `codex-critical` and
`codex-review` (Astra Ultra), each requesting three auxiliary threads. Native
agents are `bounded-explorer` (Terra medium), `command-runner` (Luna low) and
`independent-reviewer` (Astra Ultra). No economical inherited default is configured.

The native CLI `codex exec --strict-config --profile codex-general --sandbox
read-only` completed a no-tools smoke turn. Reading/resuming only that inactive
smoke session confirmed Astra xhigh. Its native exec approval policy was `never`;
launcher `on-request`/automatic approval review policy is unchanged.

`config/read` from the real linked maintenance worktree reported cap 3 with a
project-layer origin. An untrusted isolated Git project correctly skipped its
project config. Personal roles were then discovered in a new session from the
user configuration folder without trusting that project.

## Bounded validation

- **14 offline launcher tests passed**, exercising actual executor lifecycle and
  generated audit code: fresh starts, preserved higher inactive selections,
  refused lower/different selections, active-thread guards, fresh audit isolation,
  insufficient reviewer capability, invalid/missing profile, malformed TOML,
  permission-key rejection, unavailable effort, pagination and response mismatch.
- Both workflow files parse as YAML; all four shell blocks pass `bash -n`;
  generated executor/audit Python compiles; `git diff --check` passes.
- All three profiles pass installed `app-server --strict-config --stdio` parsing
  in credential-free isolated homes. These short parser processes received no
  model turns and were not a fallback for the shared runner server.
- The installed parser rejects an unknown field, malformed TOML and cap zero.
  `--strict-config` is unsupported by `features`/`debug`; those attempts were
  rejected, so they are not counted as successful validation.
- Critical owner and fresh reviewer used Astra Ultra on a tiny fixed fixture;
  explorer used Terra medium; the prescribed command helper used Luna low and
  executed `python3 -B check.py` with exit 0. The owner fixture's independent
  expected value was literal `5` for `add(2, 3)`.
- Native custom-role tests supplied conflicting explicit spawn settings: an
  explorer requested Astra Ultra but runtime confirmed Terra medium; a reviewer
  requested Luna low but runtime confirmed Astra Ultra. Child `source` metadata
  independently identifies the selected role and fresh parent relationship.
- Three custom children ran concurrently. The exact native fourth-spawn result
  was `collab spawn failed: agent thread limit reached`. A nonexistent role was
  rejected with `unknown agent_type`. Only these smoke tool metadata/results were
  retained from the smoke transcript; prompts and encrypted message data are not
  part of repository evidence.
- New execution and audit adapter calls against the shared server confirmed
  Astra Ultra from the actual installed profile files, in the real maintenance
  worktree. They created only isolated idle smoke threads, not issue workflows.
- The fixture was a committed temporary snapshot at
  `f5038db763cb5d50f9204a76394d13c8849948cf`; no sockets, model files, full product
  suites, GitHub transitions or shared test ports were needed.

## Limitations observed, not hidden

A loaded inactive thread ignored model/effort overrides in `thread/resume`.
Explicit `turn/start` selection changed the smoke thread to xhigh and back to
Ultra; subsequent resume responses confirmed both. An older loaded thread may
also ignore a newly requested cap. New-session cap enforcement is demonstrated;
legacy-session adoption needs an idle window and verified reload/new session.
The adapter logs the cap as requested, not confirmed.

An explicit parent workspace-write policy prevailed over the command agent's
read-only file setting. Role instructions remain necessary; use a separate
read-only session when an enforced write boundary is required. This diagnostic
also caught one owner prompt adding an extra `.` command argument. The command
was not counted as an exact-command pass; fenced exact command lines and separate
cwd instructions were added to the guide, followed by a corrected helper run.
Direct App Server input to a V2 child is rejected; follow-up goes through its owner.

Native start/resume confirmations do not supply an atomic compare-and-set guard
against another client simultaneously changing a thread. Existing serialization
and active-thread checks are preserved. Availability failure is tested through
catalog validation/mocks; no intentionally invalid billable model turn was sent.
The smoke reviewer reviewed only the fixture and is not a final audit of this PR.

## Workflow provenance and pending activation

The original executor, audit workflow and audit generator were byte-identical to
[Skillforge commit 747bf4766dec7c9e75384c31b601b5523b2dbb78](https://github.com/murillo128/skillforge/tree/747bf4766dec7c9e75384c31b601b5523b2dbb78):

| File | Original Git blob |
| --- | --- |
| `codex-execute-ready.yml` | `bbf95c4f4b7c8e1853e92bbdd67ba28b0b6b631a` |
| `codex-review-ready.yml` | `85b8cf23267bb28939c6aba43a8bc3b060ccfaeb` |
| `prepare_pr_audit.py` | `998d6f24823eabcec662a8cd76cc8f29502ae4c0` |

No automatic materializer was found. Local divergence is limited to profile
selection/verification and its focused helper/tests; generic skills, dispatch
labels, worktree ownership and final-audit authority are preserved. Python 3.11+
is required for `tomllib`; this host uses Python 3.12.3.

Observed labels at final validation: #114 `in-progress`; #119–123 `queued`;
#124/#125 `blocked`; #128 `blocked` for manual maintenance review. No labels on
existing issues were changed. #124 retains `max_parallel_workers: 1`.

Personal profiles and roles are installed for explicit new-session use. The
project config and launcher changes remain in the separate maintenance PR,
awaiting later review and explicit integration authority. No merge, auto-merge,
epic activation, daemon restart or stable-channel change occurred.
Local rollback manifest and detailed evidence remain under
`~/.codex-maintenance/2026-09-15/`; restore/remove only recorded, unchanged files.
