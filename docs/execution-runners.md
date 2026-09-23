# Execution runners

Executor selection and model selection are independent. The single label
entrypoint remains `.github/workflows/codex-issue-state.yml`. Existing issues
without an executor override continue to use the local Codex App Server.
This policy applies to ordinary execution and epic scheduler turns; final audit
continues to use a fresh, independent Codex session.

## Select an executor in the controlling issue

Use at most one top-level fenced `execution` block containing TOML:

````markdown
```execution
executor = "codex"
```

```codex
model = "gpt-6-sol"
model_reasoning_effort = "xhigh"
```
````

The `codex` model above is an example, not a launcher default or availability
claim. The live Codex catalog must accept any explicit model/effort pair.
The existing optional `codex` block and native/profile/resume behavior remain
unchanged; see [Codex operations](codex-operations.md).

To choose Devin instead:

````markdown
```execution
executor = "devin"
```

```devin
devin_mode = "normal"
max_acu_limit = 5
```
````

Both provider blocks are optional. Missing `execution`, an empty block, or an
omitted `executor` selects `codex`. Supported executors are exactly `codex` and
`devin`; no typo, unavailable capability, missing credential or failed launch
silently falls back to another executor. Executor selection is not a workflow
state or a model alias. Never put `executor` inside `codex`, or model/effort,
commands, paths, credentials or permissions inside `execution`.

Only the current controlling issue **body** is parsed. Prose, quoted examples,
indented code, nested Markdown fences, labels, comments and parent settings do
not select an executor. Duplicate, unterminated, oversized, malformed or unknown
settings fail closed. A failed issue read is not a default selection.

Each epic child selects independently, allowing a mixed Codex/Devin DAG. The
parent's executor controls only that parent's scheduler turns. A child completion
or `queued` wake resolves the canonical parent first, then reads the **parent's**
executor; it never uses the triggering child's selection for the scheduler.
Changing configuration does not activate an issue, release a hold, change the DAG
or increase `max_parallel_workers`.

## Provider capabilities and final audit

Codex still uses `[self-hosted, codex]`, the persistent clone/worktrees and shared
App Server. Its `codex` block configures its execution and fresh audit settings.
Do not rename the existing runner label, kill/restart the App Server, alter host
authentication or reinstall local profiles to select another executor.

Devin uses a hosted Actions control-plane job and the organization-scoped v3 API;
the implementation runs in Devin's persistent VM with its configured repository
integration. No self-hosted `devin` Actions runner or local Codex socket is needed.
Devin exposes `devin_mode`, not arbitrary Codex model IDs or reasoning efforts.
Supported values are `normal`, `fast`, `lite`, `ultra` and `fusion`; availability
and approval policy remain organization-owned. Omission keeps the organization
mode default. No local model alias is translated or silently substituted.

`review-ready` still routes directly to `.github/workflows/codex-review-ready.yml`
regardless of the implementation executor. A `codex` block on a Devin-owned issue
therefore configures its independent Codex audit, not its Devin implementation.
Choosing the audit provider is outside this change. Review capability, exact-head
CI, integration freshness and merge authority remain unchanged. Keep the Codex
audit runner available even for a Devin-only implementation wave.

Both executors use the existing skills, `codex/issue-N` branch convention,
`codex-epic-dag:v1` graph and `codex-execution-context:v1` activation context.
Those historical names are workflow identifiers, not instructions to run Codex.
Only `SKILLFORGE_LOCAL_RUNNER=1` makes the local host/worktree lease mandatory;
Devin must not manufacture local-runner variables or copy Codex credentials.

## One-time Devin prerequisites

A repository administrator configures:

| GitHub setting | Purpose |
| --- | --- |
| Secret `DEVIN_API_KEY` | Devin v3 organization service-user credential, normally `cog_...`. |
| Variable `DEVIN_ORG_ID` | Exact organization ID beginning `org-`. |
| Variable `DEVIN_MAX_ACU_LIMIT` | Required positive integer ceiling for each session; there is no unbounded default. |

The optional issue `max_acu_limit` may reduce, but never exceed, the repository
ceiling. The service user needs session use/read/manage capabilities for creation,
status checks and follow-up messages. Enable repository read/write and GitHub
issue/PR operations through Devin's own authorized integration. A successful API
launch does not establish that this integration or model files/test infrastructure
exist in the VM. Host-local weights, CUDA resources and caches are not transferred;
designs requiring them need an explicitly suitable execution environment.

No secret is put in issue bodies, session prompts, tags, comments, command-line
arguments or `session_secrets`. Only `DEVIN_API_KEY` is forwarded to the reusable
Devin launcher; it is sent to the fixed API host as an Authorization header. The
short-lived Actions token is used for GitHub control-plane records only. The
launcher never forwards it into Devin. Redirects are refused, write requests are
not retried automatically, and API errors do not print response bodies or tokens.
Organization approval/security defaults are preserved; no bypass is requested.

The dispatcher and Devin launcher require the default-branch workflow. Their
checkouts read trusted infrastructure only, with persisted Git credentials
disabled. Neither executes issue-provided shell commands or PR-head code.

## Ownership, retries and recovery

Before execution, the hosted selector re-reads the live state and establishes
one Actions-owned `<!-- skillforge-executor:v1 -->` comment with a JSON executor
lease. Repeated same-provider selections reuse it. A conflicting, duplicate,
malformed or non-Actions-owned record stops the launch rather than choosing an
owner. Selector jobs are serialized per resolved controlling issue.

A legacy issue without a lease may continue on Codex. A first Devin selection
refuses an already `in-progress` issue, an existing issue branch or an initialized
epic DAG. These remote checks cannot prove the absence of unpublished legacy
Codex work on a host: inspect/coordinate that host before any manual transfer.
Changing an issue block alone is **not** a hot executor/session migration.

Devin uses one separate Actions-owned `<!-- skillforge-devin-session:v1 -->`
comment for organization, issue scope, session identity, effective creation
settings, last Actions run and `pending`/`ready` acknowledgement state. These two
launcher records are infrastructure-owned: executors and schedulers must not edit,
delete or duplicate them. They do not replace the canonical epic DAG or child
activation context and do not grant implementation/merge authority.

Every paid create/resume request is preceded by a verified durable `pending`
receipt. A successful response is identity-checked and recorded `ready`. Replaying
the same Actions run does not repeat it. A timeout, lost response, failed receipt
write or mode mismatch stops for manual reconciliation; it never starts a second
paid session to compensate. The create request includes a `skillforge-run:<run>`
tag to locate an unrecorded response in Devin. A launch acknowledgement is not
issue success, PR readiness or proof that validation ran.

Ordinary dispatch does not steer an active session. Scheduler follow-ups may wait
up to 15 minutes, repeatedly checking live holds, ownership and settings, then
resume the same inactive session. Quota/user holds, approval waits, archived,
error and unknown unsafe states are not bypassed. A bounded wait timeout requires
a later authorized retry; this change installs no cron or extra wake-up service.

A Devin mode or ceiling change for an existing session fails rather than claiming
that a message changed creation-only settings. For recovery or provider transfer,
a human first verifies/stops the old turn through its native control plane,
preserves unpublished work and branch/PR ownership, and explicitly reconciles the
existing records in place. Do not merely delete a pending receipt or lease to
force another launch. Restore exactly one compatible record only after proving
whether the old request/session exists; retain session evidence in the handoff.
Do not restart, migrate or relabel active tasks as a provisioning smoke test.

## Validation

Run the offline regression suite without any provider credentials:

```sh
python3 -m unittest discover -s .github/scripts -p 'test_executor_routing.py' -v
python3 -m unittest discover -s .github/scripts -p 'test_codex_profile.py' -v
```

The path-scoped `executor-routing-ci.yml` checks both suites without starting an
agent. New tests cover parser defaults/isolation, provider leases, legacy-owner
refusal, current holds, pagination, API identity, duplicate events, pending
receipts, busy/resumed sessions, bounded waits, cost caps, errors and credential
boundaries. Offline doubles and YAML parsing are not live Devin/Codex validation.
The first explicitly authorized real task is the end-to-end environment check.

API contract references (checked 2026-09-23):
[create session](https://docs.devin.ai/api-reference/v3/sessions/post-organizations-sessions),
[get session](https://docs.devin.ai/api-reference/v3/sessions/get-organizations-session),
[send a message](https://docs.devin.ai/api-reference/v3/sessions/post-organizations-sessions-messages).
