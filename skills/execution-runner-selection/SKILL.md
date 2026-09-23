---
name: execution-runner-selection
description: Select Codex or Devin independently of model settings, preserve issue ownership and audit boundaries, and verify the appropriate launch prerequisites without activating work.
---

# Execution Runner Selection

## Responsibility

Use when designing or changing an issue's executor, or inspecting its launch
configuration. The canonical configuration, provider capabilities, prerequisites
and recovery rules are in `docs/execution-runners.md`. This skill does not grant
issue activation, host provisioning, credential changes, ownership transfer or
merge authority.

## Selection procedure

1. Read the exact controlling issue and its current single workflow-state label.
2. Choose the executor explicitly requested by the user, or retain native Codex
   when none is specified. Do not infer a provider from a model name or prose.
3. Put at most one top-level `execution` TOML block in that issue body with
   `executor = "codex"` or `executor = "devin"`.
4. Keep model/effort/profile in the separate optional `codex` block. Devin accepts
   its separate optional `devin_mode` and bounded `max_acu_limit` in `devin`;
   it does not accept Codex model IDs or reasoning efforts.
5. Inspect existing executor/session records and active ownership before changing
   a started issue. A body edit cannot migrate an active or saved session. Route
   conflicts through an explicit idle-session transfer, never delete the lease
   or pending receipt merely to force a launch.
6. Verify the selected environment can satisfy the issue's declared validation.
   A local CUDA/model-cache dependency is not automatically available in Devin.
   Report missing required infrastructure without choosing another executor.
7. Preserve current labels/holds unless the user separately authorizes activation.
   Describe the actual selection and any unverified live prerequisites honestly.

## Epics and audit

Each child is an independent controlling issue. A parent's setting selects only
its scheduler; it never propagates to children. For a user-requested whole-epic
choice, intentionally update every requested child rather than imply inheritance.
Mixed-provider children retain the same DAG, integration branch, dependency,
mutex and `max_parallel_workers` rules. The scheduler alone activates `queued`.

Historical `codex/issue-N`, `codex-epic-dag:v1`, `codex-execution-context:v1` and
skill names remain shared workflow identifiers. Devin follows the same execution
or epic-scheduler skill, but does not impersonate a local Codex runner.

Final `review-ready` audit remains a fresh independent Codex session. A `codex`
block on a Devin issue is for this audit only. Do not perform self-review or grant
Devin implementation sessions merge/completion authority.

## Infrastructure boundary

Codex provisioning stays in `skills/codex-local-runner/SKILL.md`. Devin needs the
repository secret `DEVIN_API_KEY`, variables `DEVIN_ORG_ID` and
`DEVIN_MAX_ACU_LIMIT`, and an authorized repository integration in its own VM.
Never copy secrets into issues, prompts or repository files. Do not launch paid
sessions, relabel real issues, alter local authentication or restart existing
turns merely to test configuration.

The Actions dispatcher alone owns `skillforge-executor:v1` and
`skillforge-devin-session:v1` comments. Product executors and epic schedulers read
them as ownership evidence, never modify them. A `pending` receipt means a request
may already have succeeded; require reconciliation, not an automatic second call.
