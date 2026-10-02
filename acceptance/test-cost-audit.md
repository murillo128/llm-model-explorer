# Consolidated test cost and retained coverage — epic #278

This report records observed results for #272–#277. The accepted product contracts
remain in `docs/spec/`. Full local reference and CUDA success are separate from
CPU fixture success. No test-count or percentage quota was used.

## Revisions, environment and measurement scope

- Activation baseline: `f014a3447d9f86da7707417a1bd34347ad1cf27a`.
- Final child activation/integration base: `25a8d7c3be08e4e74383a988cef74feafeef8414`.
- Isolated selected/full browser and integration snapshot:
  `6aca73f8bd11034b6e1b4b6ddfaf7cbacb461f35`.
- Routing guard: `33ba0d5d1f0e8d5e2141ae6ba9cc5fc8ffbacd84`, narrowing
  infrastructure exceptions to exact known paths. Independent negative controls
  fail on the preceding selector and pass after the correction. The native files selected by both
  exercised plans remain identical. Subsequent discovery safeguards are
  recorded below; application source and assertions remain unchanged.

The host is the same 16-logical-CPU Linux host (62 GiB RAM), Node 24.14.0/npm
11.9.0, Python 3.12.14, Playwright 1.63.0/Chromium 153 revision 1243,
PyTorch 2.14.0+cpu and SwiftShader. Component browsers use two workers;
production acceptance uses one headed worker. Retries are zero. References are
not supplied. Dependencies/locks match the activation baseline. Local uv is
0.12.16; CI's configured 0.12.13 is unchanged. Warm dependency installation was
shared, outside the reported native browser clocks; fresh builds and browser/server
setup/body/teardown are included. There is no adopted one-time fixture cache:
per-case generation and fresh backend/cache/session startup remain included.

Heavy suites are serialized on isolated archives with separate ports/reports.
One restricted-environment attempt failed WebGL resource allocation (15 passes,
43 failures); an unchanged host probe and the complete host tokenizer route pass.
Those failed environment results are retained and excluded from timing comparisons.
No assertions, deadlines, tolerances or retry settings were changed.

- Final discovery/CLI-filter safeguards: `33eb17a75c557c8ef6a38103ade3493600437d64`.
  Native registration proves every existing case/project pair unchanged, adds a
  new nested untagged spec to DPR 1, and preserves literal regex-special filenames.

## Retained owners and expanded work

| Boundary | Activation | Optimized full selection | Explanation |
| --- | ---: | ---: | --- |
| UI component browser | 776 | 521 | #273 removes 244 duplicate project invocations; #274 moves 11 temporal invocations to their cheap owner and compacts four non-scale graph cases |
| Production browser | 128 | 93 (61 DPR 1 + 32 DPR 2) | #276 relocates three repeated distribution inputs with complete independent TCP/component owners and retains density-specific native cases |
| UI unit | 1,038 | 1,043 | Five added temporal/history cases, including inclusive 180 ms cutoff and burst/interruption/reset oracles |
| HTTP/orchestration/report pytest | 45 | 69 collected | Five report tests, seven additional entrypoint tests and twelve selector test methods; real HTTP owners remain intact |
| Backend | 1,964 per Python | Unchanged | Both accepted main Python versions (3.12/3.14), storage/numerical/session/native exporter and installed-wheel proofs remain |
| Independent API | 118 instance / 262 architecture / 132 wire fixtures | Unchanged | 365 references resolved; bindings checked separately by the UI |

#275 deliberately adopts **no fixture-copy optimization**: measured immutable
inputs were generally cheaper to generate than copy, and one plausible architecture
profile's isolated advantage did not establish cumulative savings. There is no
shared live backend, session/cache state or cached test result. Its generation/copy
measurements and scoped lifecycle evidence remain in the
[activation report at the integration base](https://github.com/murillo128/llm-model-explorer/blob/25a8d7c3be08e4e74383a988cef74feafeef8414/acceptance/test-cost-audit.md).
No setup saving is attributed to that child.

Exact retained proof is documented in
[the domain/compact-fixture map](../ui/evidence/domain-test-ownership.md) and
[the production/density map](integration-test-ownership.md). Scale-sensitive
48-instance graphs, intrinsic fractional DPR, native scrollbars, pointer capture,
keyboard/focus, scalar/texture ownership, progressive publication, cancellation,
sharing, failure cleanup, port/parameter identity and independent numerical values
remain at their real owners. Newly admitted CLM/Kev/LoRA/formula/routed-port cases
are included. The original temporal/window/title faults fail independent controls;
no known failed case was removed by routing.

## Responsibility routing

[The conservative ownership and invocation map](validation-routing.md) is backed
by one stdlib selector. Its full/focused native collections are:

| Representative surface | Component main / PR | Product main / PR |
| --- | ---: | ---: |
| Tokenizer | 58 / 45 | 46 / 32 |
| Architecture Explorer | 156 / 121 | 93 / 61 |
| Rendering/reusable matrix | 476 / 349 | 93 / 61 |
| Shared/API/unknown | 521 / 383 | 93 / 61 |

Tokenizer's focused route retains real network/tokenization/embedding owners;
rendering retains tensor, embedding and architecture weight-modal consumers.
Architecture integration is intentionally broad because native weight streams and
shared startup cross boundaries. Selections union all owners. Fast checks stay
complete for each selected owner; this child deletes no product assertion.

All four product workflows inspect exact complete Git diffs before conditional
steps: current PR base/merge-base and head, push before/after including every
commit, both rename names and deletions. Unknown/missing/non-forward/malformed
context falls back to full coverage with reasons and revision context. Invalid
explicit plans and empty required groups fail. Same-repository restrictions,
contents-only permission, isolated ports, superseded-PR cancellation, logical
check names and Python/worker budgets remain. Each logical gate is one job; failed,
cancelled or missing selected commands cannot be aggregated into success. Explicit
non-applicability is shown separately from executed tests in job summaries.

Normative API/UI/backend/product specs, API/generated inputs, example exporters
and shared fixture/config/lock inputs reach their consumers. Operational Markdown
uses documentation/infrastructure owners. Manual workflows and final epic PRs
force full selection, independently of paths. `check.sh` remains complete;
main integration continues to avoid duplicated component/API checks.

## Measured outcome and validation

The 2026-10-02 continuation authorizes a bounded harness repair. Extend the
existing real-WebGL probe false/true cases with independent native-draw and
observer-only framebuffer-query counters. Pre-upload and non-matrix draws must
not record first render; the first eligible draw records it once. Later draws
must still reach native GL. With capture off they must avoid redundant observer
queries/readbacks; with capture on they must refresh exact pixels and counters.
Existing upload/count/resource/reader/error and resize/DPR/replacement/loss
assertions remain. Demonstrate the unnecessary-query assertion red against the
old probe, then green after the fast path. Compare retained camera/native-pane
cases on quiet isolated snapshots before the single authorized repaired full gate.

**Acceptance is unmet. Issue #277 returns to `investigation-required`; the PR
remains draft.** One complete selected integration attempt exhausted the unchanged
25-minute budget. No second full application attempt, extra timing-baseline run,
timeout increase or coverage reduction is used to obtain a green outcome.

| Personally executed check | Observed result | Wall time / qualification |
| --- | --- | --- |
| Isolated tokenizer route, all native projects | 58 passed, zero skips/failures/retries | Native 82.532 s; selector/Xvfb envelope 83.075 s |
| Shared/API complete component route | 521 passed, zero skips/failures/retries | Native 742.602 s; envelope 743.526 s, fresh build 0.400 s |
| Complete HTTP gate in selected integration | 58 passed / 10 capability skips | 167.58 s, 68 cases on the measured snapshot; final nested-discovery method makes current collection 69 |
| Complete integration command with all DPR selections | **Exit 124 at 1500.001 s** | Partial browser log: 56 passed / 30 skips; 7 lack completed evidence |

The deadline interrupts `product.spec.ts`'s DPR-2 `integrated camera gestures,
exact selection, aligned scales and adaptive inspection retain scalar storage`.
Its trace ends during forced-close teardown; no completed camera assertion result
is available for that density. Native browser JSON never finalized. Six later
cases also lack completed evidence. The 1000×700 pane case passed at both densities,
and the historical camera case passed at DPR 1; neither repairs the incomplete
whole gate. Issue-owned backend ports closed; two surviving static servers were
verified against the isolated snapshot and terminated.

The pending set also includes stale-view fencing, DPR-change inspection, two
inventory captures, adaptive magnifier and linked two-card scientific parity.
These remain selected, mandatory assertions. They are not converted into skips.
All 30 observed browser skips correspond to missing actual references or existing
unsupported tiny MoE fixtures; HTTP's 10 skips retain its reference/CUDA rules.
CPU fixture success is not complete actual-reference or CUDA acceptance.

Component native attempt sums include setup/body/teardown: **1424.791 s** total,
**77.141 s** Before Hooks, **47.294 s** After Hooks and **1300.356 s** body/remainder.
Tokenizer sums are **159.499 s**, **2.487 s**, **4.130 s**, **152.882 s**, respectively.
Hook/remainder sums are not elapsed pipeline time. Fixture generation remains
inside native setup; production generation can overlap spawn-to-ready and must
not be added twice. Current complete production phase sums are **unavailable**,
not zero. No one-time fixture preparation was adopted.

The successful activation UI artifact reports 1316.178 s native wall, 2571.637 s
attempt sum, 151.594 s Before Hooks and 72.937 s After Hooks for 776 invocations.
Its other heavy CI jobs overlapped. The optimized 742.602 s is an observed
comparison, not a reproducible causal whole-pipeline percentage: the permitted
extra quiet baseline sample was not started once correctness/budget failed.
The cancelled baseline application run remains an invalid speedup denominator.
#276's complete standalone optimized browser run (63 passed / 30 skips, 1353.341 s)
is supporting unchanged-application evidence, not successful execution of this
full HTTP-plus-browser command.

Reproducible affected work reduction is established by native collection and the
real passing tokenizer route: 58 component invocations instead of 521 full, while
shared/API selects all 521 and all 93 product invocations. There are **29 added
cheap cases** relative to activation (five UI temporal, five report, seven
entrypoint, twelve selector); selector/entrypoint startup costs about 3 s.
Scoped controlled #274 and #276 timing comparisons remain in their linked proof
maps and are not summed/double-counted as cumulative savings. **Complete final
acceptance within budget and a cumulative timing-success conclusion are unmet.**

Before execution can finish, investigate the measured full-gate cost/interruption
and establish an authorized repair with retained owners, then obtain fresh complete
exact-target validation inside the existing budget. Do not merely rerun unchanged
suites, increase timeouts, drop DPR/native cases or waive final integration CI.

The new selected-command regression fails against the pinned original entrypoint
(`--plan` rejected), then passes after implementation. Unknown nested scripts are
incorrectly exempted by the preceding selector and select full coverage after its
exact-path correction. Native entrypoint tests execute selected commands and actual
workflow shell invocations with stubs, including pytest/build/browser failure
propagation. Temporary Git histories establish complete pushed ranges, current
PR merge-base, rename/deletion and invalid diff behavior; stubs establish orchestration
only, not application correctness.

Local cheap checks: full UI binding/type/lint/unit/build checks (1,043 cases),
independent API validator, selector/entrypoint/report tests and scoped Ruff
lint/format pass. Infrastructure regressions pass (127 cases).
`skills/test-quality/SKILL.md` stays byte-identical. Backend/API/spec trees and
locks match the activation baseline; its successful backend 3.12/3.14 CI and wheel
smoke remain applicable unchanged-source evidence, not a newly rerun backend suite.
The baseline [UI CI](https://github.com/murillo128/llm-model-explorer/actions/runs/36906375279)
passed, while [application CI](https://github.com/murillo128/llm-model-explorer/actions/runs/36906375007)
was cancelled. That cancellation is never a percentage-speedup denominator.

The tested local archives/ports, native lists, HTML/JSON/JUnit, harness phases,
command logs and timing summaries remain outside Git in `/tmp/issue-277-evidence`.
The concise report is repository evidence. Source and exercised-plan identity are
reconciled after report-only edits. Epic-child product CI deferral is retained;
this evidence does not waive complete applicable CI on the final integration PR.
