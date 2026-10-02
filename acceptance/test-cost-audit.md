# Consolidated test cost and retained coverage — epic #278

This report records observed results for #272–#277. The accepted product contracts
remain in `docs/spec/`. Full local reference and CUDA success are separate from
CPU fixture success. No test-count or percentage quota was used.

Current execution is the revised 2026-10-02 risk-based contract for #277. The
historical three complete attempts below remain failed evidence of their revisions.
The new routine/extended/full portfolio is implemented and locally validated.
Complete routine acceptance finishes normally in 561.084 s, all 451 components
pass, and required primary/extended/installed-wheel compatibility checks pass.
Current commands, exact revisions and deliberate coverage trades are in the final
revised portfolio section below; earlier tables retain their historical meaning.

## Revisions, environment and measurement scope

- Activation baseline: `f014a3447d9f86da7707417a1bd34347ad1cf27a`.
- Final child activation/integration base: `25a8d7c3be08e4e74383a988cef74feafeef8414`.
- Isolated selected/full browser and integration snapshot:
  `6aca73f8bd11034b6e1b4b6ddfaf7cbacb461f35`.
- Authorized harness repair: `11b555125d35f2f17482583b3e229c7947c1bd12`
  stops redundant post-first-render observation with pixel capture disabled.
  `985cff82e697316221ee55f8bcccf2d641d433ec` batches settled-frame observations
  and is the earlier repaired complete integration target. The user-authorized
  UI consolidation below changes test ownership and selection after that target.
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
The repair's before/after/batched samples and repaired complete gate all set
`PYTHONDONTWRITEBYTECODE=1`; the earlier `6aca73` full-gate driver did not.
This environment difference is disclosed, not assigned a measured timing cause.

- Final discovery/CLI-filter safeguards: `33eb17a75c557c8ef6a38103ade3493600437d64`.
  Native registration proves every existing case/project pair unchanged, adds a
  new nested untagged spec to DPR 1, and preserves literal regex-special filenames.

## Retained owners and expanded work

| Boundary | Activation | Selection before UI consolidation | Explanation |
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

## Harness repair and earlier complete attempts

The 2026-10-02 authorized continuation repairs test-harness overhead. The existing
real-WebGL probe cases now use independent native-draw and observer-only
framebuffer-query counters. Pre-upload and non-matrix draws do not record first
render; the first eligible draw records it once and later native draws continue.
The new capture-disabled assertion fails against the old probe for the intended
reason (queries increase from 3 to 4), while the other two cases pass. All three
pass after the fast path. Capture-enabled subsequent pixels still match independent
native `readPixels`, change with the transfer function, and refresh counters without
new snapshot allocation. Scalar/count/resource/reader/error observations and the
existing resize/DPR/replacement/context-loss assertions remain active.

The fast path alone shows little elapsed benefit. The bounded follow-through
batches native scroll, origin, rectangle and document observations from the
successful settled-frame poll, camera profile/inspection rectangles, and simultaneous
texture/reader cleanup. Raw native scroll/DPR remains the independent origin oracle;
wheel/pinch/drag/scroll actions, intermediate preview states, resize/frame barriers
and all geometric/resource expectations remain. No observation is reused across an
action or viewport change. Node-side step marks locate phases without browser calls.

Three serialized isolated samples retain the same eight native-pane, camera and
adaptive-inspection invocations, both DPRs, fresh backend/cache/session setup and
all assertions. Before uses `3f9acfe` with the new regression and timing marks;
fast-path uses `11b5551`; batching uses `985cff8`.

| Retained eight-case sample | Result | Native wall / command envelope |
| --- | --- | --- |
| Before repair | 8 passed, zero skips/retries | 461.946 / 462.502 s |
| Fast path | 8 passed, zero skips/retries | 458.770 / 459.280 s |
| Fast path and settled-frame batching | 8 passed, zero skips/retries | 365.394 / 365.943 s |

| Eight-case phase sums (seconds) | Before | Fast path | Batched |
| --- | ---: | ---: | ---: |
| Native attempt durations | 459.983 | 457.051 | 363.435 |
| Backend spawn to ready | 70.139 | 70.155 | 63.149 |
| Browser setup | 1.447 | 1.397 | 1.344 |
| Test body | 386.251 | 383.285 | 296.997 |
| Teardown | 1.469 | 1.497 | 1.355 |
| Fixture generation (overlaps setup; not added twice) | 0.319 | 0.324 | 0.322 |

The focused command is `xvfb-run -a npm run test:acceptance -- product.spec.ts
--grep 'production native pane geometry|integrated camera gestures|polish magnifier
follows edges' --reporter=list,json` from each isolated snapshot's `ui/` directory.
Probe proof uses `xvfb-run -a npm run test:browser -- acceptance-probe.spec.ts
--reporter=list,json,html`. These subsets are profiling/regression evidence,
not substitutions for the complete gate.

The batched sample is 96.552 s lower than before in native wall time. This is a
focused observed comparison, not a whole-pipeline saving or proof that all of the
difference comes from batching. Per-case setup/body/teardown and step timings are
retained; backend startup varies too. Separate fresh builds take 0.43–0.46 s.
Affected typecheck/lint and diff checks pass. Native discovery preserves all
93 product and 521 component case/project pairs, tags, expected statuses and
timeouts. Full API/shared routing still selects every retained product pair.

**At `985cff8`, acceptance was unmet and issue #277 returned to
`investigation-required` with a draft PR.** The original complete attempt and the single explicitly authorized
repaired-code complete attempt both exhausted the unchanged 25-minute budget.
The repaired attempt follows passing focused proof and measured benefit; it is
not an unchanged rerun. No unchanged complete retry or extra timing-baseline run
followed that failure. The later explicit consolidation authority is recorded
below. No timeout increase was used.

| Personally executed check | Observed result | Wall time / qualification |
| --- | --- | --- |
| Isolated tokenizer route, all native projects | 58 passed, zero skips/failures/retries | Native 82.532 s; selector/Xvfb envelope 83.075 s |
| Shared/API complete component route | 521 passed, zero skips/failures/retries | Native 742.602 s; envelope 743.526 s, fresh build 0.400 s |
| Original complete HTTP gate (`6aca73`) | 58 passed / 10 capability skips | 167.58 s; 68 cases before the added nested-discovery method |
| Original complete integration command (`6aca73`) | **Exit 124 at 1500.001 s** | Partial browser log: 56 passed / 30 skips; 7 lack completed evidence |
| Repaired complete HTTP gate (`985cff8`) | 59 passed / 10 capability skips | 178.67 s; finalized native JUnit, 69 cases |
| Repaired complete integration command (`985cff8`) | **Exit 124 at 1500.003 s** | Partial browser log: 56 passed / 30 skips; 7 lack completed evidence |

The repaired command selects from the complete 18-path Git diff against
`25a8d7c`, then runs `acceptance/check-integration.sh --main-ci --plan <plan>`
under one `timeout 1500` envelope. Selection, HTTP lint/pytest, build, discovery,
browser setup/body/teardown share that envelope; the budget is never reset.

Both deadlines interrupt `product.spec.ts`'s DPR-2 `integrated camera gestures,
exact selection, aligned scales and adaptive inspection retain scalar storage`.
The repaired partial trace reaches native lower bound (13.331 s), wheel/pinch
(19.105 s), and matrix selection (25.676 s) before interruption during axis
selection. The final `mouse.up` reports a closed browser after timeout; this is
not a completed camera assertion failure. Native browser JSON never finalized.
Six later cases also lack completed evidence. The 1000×700 pane case and focused
camera case pass at both densities; neither repairs the incomplete whole gate.
Original surviving static servers were verified and terminated. Repaired cleanup
terminates only processes verified as owned by its isolated snapshot; all six
owned ports (30060–30065) are closed, with no owned backend/static survivors.
Cleanup takes 0.184 s, for 1500.187 s including cleanup.

The pending set also includes stale-view fencing, DPR-change inspection, two
inventory captures, adaptive magnifier and linked two-card scientific parity.
These remain selected, mandatory assertions. They are not converted into skips.
All 30 observed browser skips correspond to missing actual references or existing
unsupported tiny MoE fixtures; HTTP's 10 skips retain its reference/CUDA rules.
CPU fixture success is not complete actual-reference or CUDA acceptance.

The cancelled activation application job has 14 s before acceptance and 21 s
after it (artifact/report teardown): 35 s outside the acceptance step. These
observed CI costs require headroom under the unchanged 25-minute workflow limit;
they are not a new passing CI measurement or a speedup denominator. The repaired
local gate already exceeds its own envelope without that extra CI work.

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

The remaining problem is complete HTTP-plus-browser cost within the existing
budget. The passing local repair does not establish that outcome, and the partial
trace does not establish a product defect or a single causal bottleneck. Before
execution can finish, investigate the remaining complete-gate cost and establish
a concrete authorized repair with retained owners, then obtain fresh complete
exact-target validation inside the existing budget with CI setup/report headroom.
That repaired complete-attempt authorization was exhausted. The later user-authorized
consolidation below permits specific relocations and one changed-code complete
validation; it does not permit unchanged retries, larger timeouts or a final-CI waiver.

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

## User-authorized UI consolidation — issue #277

The user requested removing all five audited duplication groups. The controlling
issue records that authority and the precise retained owners before implementation;
it supersedes the earlier fixed-93-pair restriction only for these relocations.
The preceding measurements describe their named revisions, not this new selection.

| Removed repetition / plausible defect | Independent remaining owner and gap closed |
| --- | --- |
| `architecture.spec.ts::openParameter` title/body click, exact selection, camera/layout/scope/no-request replay for every weight | `architecture-card-actions.spec.ts` and `architecture-inspection.spec.ts` retain literal generic gesture/modal assertions. One real production graph journey at both DPRs carries these gestures across the live boundary; every numeric inspection still dispatches exact graph IDs and checks actual stream/source values. |
| Component expansion/isolation, dimensions, search and explorer-state replay for each family; deterministic binding replay at DPR 2 | One production graph journey samples every declared component variant at both DPRs, retaining real port/trunk hits and literal preference/state assertions. Per-family real graph/compact projection, source conservation, exhaustive layout/reachability and native/decoded numeric bindings remain. Every actual-reference pair, size and prerequisite is unchanged. |
| Same-generator dense 28/30-layer component stress | Keep the larger 30-layer dense case, independently different 24-layer hybrid and 24+12 two-stack cases. Full node/edge size, last instance, bounded document, culling and layout/memory evidence remain. Fixture metadata used elsewhere is unchanged; titles describe topology. |
| CLM/Kev exporter→graph→weight HTTP→visible-canvas browser repeats | New mandatory `acceptance/test_native_packages.py` cases use fresh real exports and production CLI/TCP. Exact native head/factor IDs and float32 bytes are checked against independently read physical Safetensors. Existing exporter/admission/algebra/cache tests remain. Generic browser owners retain expand, parameter dispatch, real canvas and lifecycle. Supplied invalid reference roots fail. |
| Native adapted-hierarchy UI repeat and metadata-only real LoRA/QLoRA browser assertions | New mandatory TCP composition case checks literal child order, formulas/scalars/ports and physical-source A/B streams. The existing generic inspection case is extended for nested expand/isolate/Back, visible reshape/transpose/softmax/scale, factor=2, axis=-1, ports and distinct A/B dispatch. It passed before deleting the old browser case. Complete real-reference target/shape/algebra remains in `test_lora_reference.py`; actual base/factor browser bindings remain. |

Native discovery against `e73cc34` records product **93 → 85** (59 DPR 1,
26 DPR 2), component **521 → 520** (370 desktop, 138 narrow, 12 native
scrollbars), and three added mandatory native TCP cases. Product removes seven
deterministic `full graph, concrete bindings and logical weight modal` DPR-2
pairs (SmolLM2, Qwen3, Qwen3.5, V-JEPA2, DeepSeek V2, GLM4 MoE Lite, Kimi Linear)
and the DPR-1 CLM/Kev/adapted-hierarchy pairs; it adds the generic production
journey at DPR 1 and 2. Three removed deterministic pairs were already optional
unsupported-fixture skips, not runnable savings. All actual-reference case/project
pairs compare equal. The four component full-size titles become three topology
titles, and the existing factor-inspection title becomes the nested-source title;
there is no additional project deselection. Exact lists stay outside Git in
`consolidation/collection-delta.json` under the existing evidence directory.

The four formerly repeated runnable deterministic families took 162.6 s across
both DPRs in the preceding partial native log; the three relocated package cases
took 37.1 s. These are complete-case costs, not deletion savings: retained bindings,
new TCP proof and the new production journey still cost time. No cumulative
speedup is inferred by summing removed-case durations.

Focused proof so far: the production journey at both DPRs plus all retained
deterministic bindings completes normally (6 passed, 3 unchanged unsupported-fixture
skips, 94.602 s native wall). The generic nested-source replacement passes on host
WebGL before deletion of the old case. CLM/Kev TCP cases pass; the first composition
attempt exposes an incorrect test provenance lookup, corrected to the existing
exact `rule`/`source` contract, then the affected composition case passes. An initial
restricted generic attempt cannot allocate WebGL; host execution passes without
expectation changes. Native selector/entrypoint/report controls pass (33 tests,
60 subtests) after the retained full-command expectation removes the three relocated
browser filenames. HTTP discovery includes all three new cases: 72 collected.

The newly authorized complete changed-code gate is recorded after execution.
The unchanged 1500-second envelope must include
selection, HTTP, build and production-browser setup/body/teardown, with workflow
setup/report headroom. Normal native JSON/JUnit completion, every mandatory case
successful, truthful optional skips and owned process/port cleanup remain required.
No product source, scalar/count/resource/error observer, renderer, retained fixture
size, worker/retry/timeout/tolerance or actual-reference prerequisite is changed.

### Complete consolidated result

The exact isolated target is `21c83bd13b0732c0df8233b48efac894b77e6711`, against
the unchanged integration tip `25a8d7c3be08e4e74383a988cef74feafeef8414`.
Selection consumes its complete 32-path Git diff, including deletions, and selects
all product owners. No source changes occur during validation. The selected
command is the same `acceptance/check-integration.sh --main-ci --plan <plan>`
inside one `timeout 1500` envelope; browser discovery confirms all 85 pairs.

| Personally executed consolidated check | Observed result |
| --- | --- |
| Retained generic card/component/connection/summary/inspection/isolation and full-size owners, all configured applicable projects | 52 passed, zero skips/failures/retries; finalized JSON, 201.507 s native wall |
| Generic production journey at both densities plus retained deterministic native bindings | 6 passed, 3 existing unsupported-fixture skips; finalized JSON, 94.602 s native wall |
| Native selector/entrypoint/report controls | 33 passed, 60 subtests passed, 5.31 s; the first run detects the three stale relocated filenames in the full-command expectation |
| Complete HTTP in the consolidated gate | 62 logical tests passed, 10 existing capability/reference skips, 209.01 s; finalized JUnit has zero failures/errors and 132 entries including 60 subtests |
| Complete consolidated HTTP-plus-browser gate | **Exit 124 at 1500.005 s**; partial browser log has 56 passes / 27 truthful skips / 2 uncompleted mandatory pairs; native browser JSON never finalizes |

All DPR-1 cases complete. DPR-2 camera, stale-response fencing, inspection during
DPR change and both inventory captures also pass. The deadline interrupts
`polish magnifier follows edges after scrolling resize DPR and source replacement`
at DPR 2; the following `integrated two-card embeddings have real scientific
parity with the same checkpoint matrix` at DPR 2 has not run. This is missing
completed evidence, not a failed magnifier assertion. Both cases pass at DPR 1.
The partial DPR-2 trace remains outside Git for investigation.

Cleanup terminates only the three processes verified as belonging to this isolated
snapshot. All six owned ports (30100–30105) are closed, with no owned backend/static
survivors. Cleanup takes 0.305 s, for 1500.310 s including cleanup. Earlier failed
attempts remain intact. The prior observed 35 s of CI setup/report work is additional
headroom; this local gate already exhausts its own envelope.

The new selection and focused proof establish ownership consolidation, not complete
timing acceptance or a causal whole-pipeline speedup. The remaining bounded problem
is complete retained HTTP/scientific-browser cost within the unchanged workflow
budget and its setup/report headroom. The changed-code complete-attempt authority
is exhausted; no unchanged retry, case skip, larger timeout or new repair campaign
is inferred from this failure. Production backend/API/UI source, accepted specs,
locks and the reusable `test-quality/SKILL.md` remain byte-identical. Report-only
publication edits are reconciled against this exercised source and selection.

## Revised risk-based portfolio — implementation plan for #277

The 2026-10-02 replacement issue contract authorizes deliberate low-value coverage
retirement and routine/targeted-extended/full portfolios. It supersedes the preceding
fixed-selection and exhausted-attempt restrictions. The three historical failures
remain evidence of their revisions. Before removal, the following bounded decisions
record the proposed retained core and plausible lost detection; final collection,
validation and timing follow implementation. Additional savings are unmeasured
unless a comparable execution is explicitly recorded.

| Worklist / exact candidate family | Planned disposition, retained proof or accepted risk | Cost evidence / focused target |
| --- | --- | --- |
| 1. Both Playwright configurations, `retain-on-failure` | Routine traces off; explicit native `--trace on` diagnostic mode. Keep JSON, failure screenshots/logs and compact diagnostics. HTML optional. | Compare unchanged 1000×700 production pane with/without trace; no assertion change. |
| 2. `matrix-zoom.spec.ts` shape×DPR×responsive, selection/pixel/centering/magnifier matrices | Retire interchangeable combinations; explicit representatives retain all four shapes, a fractional DPR, DPR 2 and narrow layout. Cheap camera/geometry/history owners retain arithmetic. Keep native wheel/pinch/exact selection and one resource churn bridge. | Existing native lists; focused affected component files. Lost detection: a combination-specific layout defect outside the chosen representatives. |
| 3. `product.spec.ts` progressive, camera, magnifier and scientific parity | Short real producer-barrier/source-value/cleanup core, one camera bridge, compact exact three-request two-card proof. Retire repeated luminance/neighborhood/gesture/screenshot sweeps owned by renderer/component tests. Split by responsibility for selection. | Historical complete case durations; focused production core and retained renderer/native owners. |
| 4. Product prompt/inventory/viewport/density and reference permutations | Logical cases one density. Keep one native DPR bridge and event-order transition. Retire repeated inventory captures/presentation narratives; references and genuinely large traversals targeted extended at one density. | Native before/after pairs; no savings assigned to absent-reference skips. Lost detection: full presentation on every width/model/DPR combination. |
| 5. `test_tensor_analysis.py::test_endpoints_and_disk_reuse`, 19 inputs | Keep independent 19-input numerics; endpoint representatives cover asymmetric/constant/mixed/nonfinite/empty layouts. Persistent cold/warm exact metadata+bytes once. Retire weak DONE-only dtype wrappers: exact F16/BF16 bytes remain in `test_tensor_data.py::test_exact_shapes_and_native_source_unchanged`. | Focused analysis/data tests. Lost detection: artifact-count-only cross-dtype analysis check, not exact conversion proof. |
| 6. Selector broad test/directory routes | Finite actual-file responsibility groups, direct changed-test routes and explicit helper consumers. Shared/unknown/new inputs remain conservative; ordinary untagged tests remain routine. Preserve exact Git diff and empty/failure guards. | Native collection, existing selector/entrypoint behavioral controls. |
| 7. Fresh services per independent case | First remove duplicate journeys. Keep fresh services for cold/cache/barrier/cancel/error/model mutation. Reject service pooling if it requires a new state-reset/isolation framework; read-only candidates must show benefit and safety before adoption. | Historical eight-case startup 63.149 s; no fixture-copy campaign. |
| 8. Architecture 8/16-cycle lifetime/GC, full-size sweeps; real backend percentile/allocator thresholds | Keep a compact routine return/remap/cleanup/scale bridge; long soak/full traversal and real 2**24+1/allocator threshold retained as targeted extended, with original sizes. | Affected extended checks required on this exact refactor target; deferred work is not eliminated work. |
| 9. Backend Python 3.12/3.14 workflow | Primary optimized suite on 3.12; version-independent checks once. 3.14 installed-wheel/CLI plus representative real runtime ordinarily; full compatibility for relevant runtime/dependency/packaging changes and manual mode. Isolated wheel installs. | Native Python collections and wheel/runtime proof; reduced default 3.14 confidence explicit. |
| 10. API/UI generated checks repeated by integration | Dedicated exact-target API/UI workflows own drift and validation. Preserve full local path; CI routine integration avoids duplicate validation/regeneration only with that ownership explicit. | Inspect `--write` behavior, API validator/drift, entrypoint error propagation. No cross-workflow build cache for a ~0.4 s build. |
| 11. Headed acceptance for logical transport cases | Evaluate headless for non-native state/transport specs; preserve headed native/focus/window cases and SwiftShader. No extra workers merely for elapsed speed. | Focused real transport and native rendering mode checks; no assumed pixel equivalence. |
| 12. Observer/polling/snapshot work | Keep prior fast path and batching; remove repeated full-state reads/pixel sweeps with retired narratives. Further observer complexity only after measured evidence. Keep independent scalar/count/resource/error and real pixel observers. | Focused probe/retained scientific core; no instrumentation-only campaign. |
| 13. Remaining backend/UI/API/examples/infrastructure tests | Audit costly repetition and weak wrappers; keep cheap independent numerical/security/protocol/exporter and real lease/Git/command controls. Remove only support with no consumers. No mass mutation or unit-test count quota. | Existing owner suites, import/consumer searches; specific additional retirements recorded before edit. |
| 14. Component preview+dev and acceptance density servers | Resolve consumers; start only selected density servers where native selection can express this simply. Preserve separate dev and production build evidence. Reject caching/pooling machinery for negligible static startup. | Native config/CLI collection, selected-server startup and cleanup; additional savings unmeasured. |
| 15. Workflow planning/event duplication | Preserve existing check identities and same-repository/least-privilege gates. Narrow conditional product work from exact paths; no status fabrication, branch-rule change, controller or unrelated runner edits. | Actual workflow shell/selected-failure controls; current event/base/merge identity. |
| 16. Thread/host contention | Heavy measurements serialized. Evaluate bounded test-only native threads if needed; preserve production defaults and report host work separately. No additional parallel headed workers or another issue's cleanup. | Scoped identical scenario if adopted; otherwise no-change decision with unmeasured opportunity. |

Routine mandatory retains independent float32/statistics/distribution/protocol
values, a real producer-before-first-pixels bridge, exact source/parameter dispatch,
shared-consumer survival/cancellation, pre-/midstream errors, stale responses,
resource teardown, meaningful native/fractional/DPR and graph scale/culling proof.
Full/manual retains all non-retired extended/reference/CUDA checks. The new routine
combined gate has a 1200 s envelope (900 s engineering target); the workflow remains
25 minutes with explicit setup/report headroom. Two changed-target complete routine
attempts at most are authorized; focused checks precede them.

The first reduced complete component run at `98e5299` finished 450/451 cases.
The unchanged `architecture-camera.spec.ts` connection-center setup sampled the
camera after manual Fit returned from its click but before React Flow's queued
frame completed. Its final exact Back comparison therefore used the preceding
camera. React Flow's native `fitView` promise resolves after the queued fit;
`aria-busy` covers initialization, not that manual command. Repair only the test
setup to await the probe's actual Fit completion before taking its snapshot.
Retain the pending-generation and exact Back assertions. Preserve the failed run,
exercise a controlled delayed real Fit as a red reproducer, and run the camera
family and complete affected components after the repair. No production change,
retry, tolerance or retired test is involved.

Final routing inspection exposed missing extended backend selection for shared
`backend/tests/conftest.py`, deleted tests and model-file dependencies. A focused
negative control fails on the preceding selector for that omission. Shared,
deleted and nested backend test support now falls back conservatively; production
backend/support inputs include the real threshold/allocator checks. Direct
existing backend/browser tests retain their finite file route. This changes
selection only; the complete refactor plan already selected those same owners.

The first full installed-wheel 3.14 run exposed a harness import-mode defect:
`importlib` named the unchanged spawned helper `tests.test_artifacts`, but the
child's test-helper-only path could not import that namespace. The real cache
competition failed with `ModuleNotFoundError: tests`; 594 cases passed before
the owned run was interrupted at 236.600 s for diagnosis. Preserve that failure,
retain every cache/process assertion and restore native pytest imports. The
private wheel/import-location/CPU guards remain. Validate the actual spawned
competition first, then complete all 1,950 cases from a new immutable target.

## Audit-return routing repair plan

The exact-head audit of `79f3f3ae7f0831cb8b87de89d7fb87b348378698`
found two omitted owners. Shared backend test modules must select their backend
consumers and genuine extended thresholds; shared `acceptance/test_network.py`
must select all HTTP consumers, including the real `Service(model_root=...)`
branches. The LoRA browser route must include `test_lora_reference.py`, whose
independent source-byte/shape/algebra assertions own the retired browser matrix.

Use the existing finite owner map to classify known shared test modules before
isolated-file rules. Backend fallback stays within backend; HTTP fallback stays
within HTTP and keeps browser/build setup non-applicable. Regression assertions
use independently inspected current imports/owners, fail on the audited selector,
and pass after repair. Exercise the real selector/entrypoint commands and their
failure propagation, then run the native LoRA TCP owner with absent and invalid
supplied reference controls. Preserve truthful capability skips. Reconcile the
final complete-diff plan to prior passing executable proof; this routing repair
does not require another unchanged broad timing sample.

The real shared-HTTP route also exposes a prerequisite: the independent TCP
architecture semantic oracle needs Node 24's native TypeScript support, even
when no browser is selected. Keep Node setup for that owner, while npm/Chromium,
build and browser-server setup remain non-applicable. Add a native output/workflow
regression covering report-only, architecture, shared HTTP and browser routes.

## Revised portfolio — adopted decisions and final validation

Python 3.14 validation uses 3.14.7 and the same locked PyTorch 2.14.0+cpu
(the CPU-package guard passes); dependencies are installed in an owned temporary
environment, never reinstalled into another checkout's environment.

The executable target is `95059bbaac6f101f0c5baf9ed23e4bb7dd6cefd0`, based on
`25a8d7c3be08e4e74383a988cef74feafeef8414`. The complete component target is
`8378490db4105a7fff173646dd0786bbdf1f1821`; subsequent selector/CI edits preserve
its exact UI/config/dependency contents and its complete 451-case selection.
Routine acceptance was measured at `03884a3ef7a622ffd55effb03952fc2f62f4da13`;
later executable changes are the backend CI manual Python matrix and the
wheel harness import mode. Neither changes its selected files, command,
UI/backend/API dependencies or routine execution. The complete primary backend
target remains `799f703bcd4d4537eca20b312d46c581c4c260ca`: backend code/tests/config
are byte-identical after the wheel-only repair. Report-only publication changes are reconciled by exact Git diff.
Backend/API/UI production source and `docs/spec/**` are unchanged. The reusable
`skills/test-quality/SKILL.md`, runner, dispatcher and canonical epic context are
unchanged. The repository-specific testing reference records the new commands.

| Worklist | Adopted change or no-change decision |
| --- | --- |
| 1 | Traces off in both routine configurations; native `--trace on` remains usable. The unchanged 1000×700 headed case passes with the same assertions: native wall 51.764 s before / 13.009 s after. This single scoped sample does not establish a global percentage. Invalid-reference diagnostics also produce a real trace ZIP with zero retries. |
| 2 | Seven component files use explicit shape/overflow/fractional/DPR 2/narrow representatives. Complete native collection falls 520→451 (370/138/12→336/106/9 by desktop/narrow/native project); four churn transitions retain the real wheel/pinch/cell/lifetime bridge. Combination-specific and longer-churn defects outside those representatives are accepted coverage loss. |
| 3 | Split product checks into five responsibility files with isolated per-test closures. Keep short real first-pixels and three-request two-card scientific proofs, one camera bridge, real network/state failure journeys and teardown. Retire overlapping integrated navigation/magnifier/shell/capture narratives; their exact risks and remaining owners are recorded in the ownership map. |
| 4 | Routine production is 38 pairs (34 DPR 1 / 4 DPR 2); full is 53 (49/4), against 85 (59/26). Thus 32 full invocations retire and 15 remain extended. References lose redundant density replay; absent-reference skips are not counted as runnable savings. Invalid supplied and required-reference controls still fail with their original diagnostic. |
| 5 | Keep nineteen native numerical inputs, seven independent cold endpoint representatives and exact cold/warm metadata+payload comparison once with recomputation forbidden. Retire twelve endpoint repetitions and two DONE-only dtype wrappers; exact F16/BF16 conversion bytes remain independently tested. Focused routine analysis/data: 90 passed, 19 CUDA skips, two extended deselections in 166.72 s. |
| 6 | Actual test files and finite consumer groups select their native owners. Shared test modules reach their consumers before isolated-file rules; LoRA includes its independent TCP reference owner. HTTP-only edits avoid npm/browser/build/Chromium; the TCP architecture semantic oracle retains Node 24. Unknown/shared/deleted support is conservative, including real extended backend dependencies. Native orchestration/diff/empty/failure controls are updated by the audit-return evidence below. Ordinary new untagged cases stay routine. |
| 7 | No service pooling: retained cold/cache/barrier/cancel/error/model-mutation cases need independent state. Historical eight-case startup is 63.149 s; a new reset framework has no demonstrated benefit after retiring duplicate journeys. No repeated fixture-copy campaign. |
| 8 | Routine nested returns/remaps use three adversarial cycles; sixteen-cycle lifetime, large native pane/prompt, references and actual 2**24+1/Linux allocator proof remain extended at their original sizes. Three available production cases pass / twelve missing-local-reference skips in 69.198 s; sixteen cycles and original 1000×700 pane / forty-line 1178px prompt remain. Actual backend thresholds are included in the full primary and compatibility checks below. Deferred work is not eliminated work. |
| 9 | Ruff/format/mypy and the optimized primary suite belong to 3.12. Ordinary main 3.14 runs eighteen isolated installed-wheel CLI/server/session/dtype-shape cases; runtime/dependency/packaging/unknown/manual scope selects full compatibility. Manual CI includes both versions. Python 3.12 isolated smoke: eighteen passes, 48.69 s native / 52.918 s envelope. Python 3.14 complete isolated wheel: 1,930 passes / twenty CUDA skips, 1,950 collected, 1022.353 s native / 1029.816 s envelope; default eighteen-case mode: eighteen passes, 56.12 s native / 63.254 s envelope. Imported package paths remain inside the private wheel environment and the CPU guard passes. Default exhaustive cross-version compatibility confidence is deliberately reduced. |
| 10 | Independent API exact fixture drift and UI binding drift have one dedicated CI owner each. Local full commands retain both; integration `--main-ci` delegates their duplicated execution. Existing validator and generated-binding checks pass. Reject cross-workflow build caching for a build measured around 0.4 s. |
| 11 | Transport-only production cases use headless Chromium with real TCP/WebGL. Native scrolling, camera, pointer/window, prompt/IME/focus and pixel cases stay headed. One SwiftShader production worker and two component workers remain; no renderer or GPU provisioning change. |
| 12 | Retain the prior probe fast path and settled-frame batching; retired narratives remove repeated full observations. Keep the real populated-pixel barrier, exact scalar/upload/resource counters and compact timing attachments. No observer framework. |
| 13 | Keep cheap independent numerical/protocol/security/API/exporter and real Git/lease/command checks after inspecting the remaining families. No deletion quota or mass mutation campaign. UI units remain 1,043; protocol fixtures remain 118 instance / 262 architecture / 132 wire, with 365 references resolved. |
| 14 | Keep separate production static and development consumers and both configured density servers. Their independent purpose and small startup cost do not justify pooling/caching machinery. HTTP-only routes now omit all browser infrastructure. Additional static-server savings are unmeasured. |
| 15 | Keep workflow identities, 25-minute limit, exact diff/head/base semantics, same-repository/least-privilege restrictions, epic-child local deferral and final aggregate gates. Use conditional setup for irrelevant owners; no passing-status fabrication, branch-rule/controller or runner change. |
| 16 | Serialize heavy measurements on isolated snapshots/ports. Keep worker and production thread defaults; no speculative threading change or another issue's process cleanup. Native attempt-duration sums are host-work observations, not CPU seconds or elapsed pipeline time. |

The compact [native case/project sets](issue-277-case-sets.json) record complete
old/new production identities, the seven changed component sets and backend
parameter/version dispositions. Native backend collection is 1,964→1,950, with
1,948 routine and two actual extended tests. The 3.14 default has eighteen runtime
cases; fuller affected/manual validation retains all 1,950.

| Personally executed proof | Completed result / measured clock |
| --- | --- |
| Complete routine HTTP + production browser, one envelope | 66 logical HTTP passes / 60 subtest passes / 10 optional skips; 33 browser passes / five existing unsupported-fixture skips. Exit 0 in **561.084 s**, including selection/setup/build/discovery/teardown, within 900 s target and 1200 s limit. Native HTTP 189.277 s; browser 369.900 s / 366.281 s attempt sum. One complete changed-target attempt, no retries. |
| Complete component browser | 451 passed, zero skips/failures/flakes/retries; 583.171 s native / 583.704 s envelope. Native attempt-duration sum 1,134.831 s across two workers; JSON omits hook breakdown, which is unavailable rather than zero. |
| Complete primary backend, including real extended thresholds | 1,930 passed / twenty CUDA skips, 1,950 collected; both real extended tests pass. Native 974.20 s / measured envelope 975.073 s. |
| Isolated installed-wheel runtime/full compatibility | Python 3.12 isolated smoke: eighteen passes, 48.69 s native / 52.918 s envelope. Python 3.14 complete isolated wheel: 1,930 passes / twenty CUDA skips, 1,950 collected, 1022.353 s native / 1029.816 s envelope; default eighteen-case mode: eighteen passes, 56.12 s native / 63.254 s envelope. Imported package paths remain inside the private wheel environment and the CPU guard passes. |
| Affected extended production | Three available production cases pass / twelve missing-local-reference skips in 69.198 s; sixteen cycles and original 1000×700 pane / forty-line 1178px prompt remain. Actual backend thresholds are included in the full primary and compatibility checks below. |
| UI/API fast checks | API binding drift, TypeScript, ESLint, all 1,043 units/58 files and build pass; independent API validation and exact fixture comparison pass. Backend lint/format (136 files) and mypy (111 files) pass. Unchanged checked contents/dependencies are reconciled to the final target. |
| Selection and command controls | 32 methods / 60 subtests pass, including real native discovery, exact Git rename/deletion inputs, selected command failures and missing required groups. Their command stubs prove orchestration, not application behavior. |

Across the 33 completed production timing attachments, native phases sum to
39.920 s fixture generation, 207.045 s backend spawn-to-ready, 3.664 s browser
setup, 107.924 s bodies and 5.894 s teardown. Generation may overlap startup;
these are not additive elapsed phases or CPU seconds. Vite reports 227 ms compile
time. The 1.907 s combined remainder outside native HTTP/browser clocks includes
selection, lint, fresh npm/build startup and wrapper overhead.

Routine timing includes selection, HTTP lint/checks, fresh build, discovery,
server/browser setup, bodies and normal teardown. Dependency installation and
checkout are outside that local clock; workflow setup/report headroom is separate. Routine plus affected production
extended commands total 630.282 s. The earlier observed 35 s workflow overhead
is historical, not a fresh end-to-end Actions measurement; cold dependency
installation is unmeasured. The unchanged 25-minute ceiling has ample measured
command headroom, but no product Actions success is inferred from local clocks.
No complete baseline exists for a whole-pipeline percentage. The old
component observation was 521 cases / 742.602 s native / 1,424.791 s attempt sum;
its scope differs, so it cannot isolate the effects of trace removal and changed
matrices. The three 1,500-second complete failures above remain failed evidence.

Native JSON/JUnit, negative-control failures, logs/screenshots/trace and detailed
phase attachments remain outside Git in `/tmp/issue-277-evidence/portfolio/`.
No model files, tensor caches, large logs or traces are committed. All final passing validations finish normally and close only their owned processes/ports. CPU fixture
success does not certify unavailable CUDA or complete local reference models.

The initial 450/451 component run failed an unchanged camera setup race. A
controlled delayed real Fit reproduces the same exact Back mismatch; awaiting
actual Fit completion before the snapshot preserves every product assertion and
passes all eighteen camera cases and the complete 451-case suite. An initial
all-Fit diagnostic gate blocked a later command and is not used as regression
proof. A Vite worker URL setup failure with symlinked dependencies is retained;
private copied locked dependencies pass all 1,043 units without changing Vite.
Reference setup failures initially lost their diagnostic during fixture teardown;
the fixture now preserves the original exception while still cleaning resources.
The extended-helper routing negative control fails before its correction and
passes afterward. These repairs do not change production behavior or weaken
expectations, deadlines, tolerances or retry policy.

The first full installed-wheel 3.14 run failed cache competition before the
production helper executed: pytest importlib naming made `tests.test_artifacts`
unimportable to spawn. Its 594 passes / one failure and deliberate interruption
at 236.600 s are retained, with remaining cases unvalidated on that attempt.
Restoring native pytest imports preserves private wheel/location/CPU checks and
all cache assertions. The real four-process competition passes (1 selected /
1,949 deselected, 13.14 s); the repaired complete 1,950-case wheel run supplies
final proof, with finalized JUnit. The two real extended tests pass on both
Python versions.

## Audit-return completed evidence

### Cross-boundary and phase continuation plan

The next contract starts at `83843e38e666151aeaa9837b839cfaf1ba1108a3`.
An ad-hoc read-only inventory inspected 372 repository source/fixture files,
ordinary and function-local Python imports, embedded preparation commands,
dynamic CLM/Kev exporters, TypeScript imports and native JSON/HTML/oracle inputs.
The static graph is an overapproximation: manually inspected CLI branches keep
the ordinary tensor/tokenizer harness separate from architecture/polish fixtures.
No test module was executed for discovery. Inventory artifacts stay outside Git.

Before changing routing, retain the known leaf controls (session operations,
matrix zoom and HTTP report) and record the direct LoRA, Kimi and multi-hop
dense/quantized/model/stream/helper closure in the routing guide. Positively
reviewed current leaves may narrow; unclassified existing files cannot infer
isolation from a filename. Shared fixtures conservatively select their relevant
native family and threshold owners, without adding full Python compatibility for
every test-only edit.

Focused regressions cover the issue's direct/transitive/embedded inputs,
deterministic unions, new-existing/nested/deleted paths, shared HTTP plus reference
browser consumers, architecture UI plus LoRA, effective reference/full promotion,
extended-only routine non-applicability, distinct phase evidence, prerequisites and
failure propagation. Expected targets come from inspected consumer imports and
commands, not the selector's own map. Prove omissions on the starting selector,
then use native discovery/prerequisite/report/trace controls and deduplicated real
architecture/native-package/polish TCP plus architecture browser routine/extended
proof. Keep prior timings attached to their tested targets; the remaining complete
routine attempt is reserved for a material need after scoped reconciliation.

The bounded routing repairs are validated at
`1610d796297771b5a44ff628852bd11bf443d40a`, against unchanged integration
`25a8d7c3be08e4e74383a988cef74feafeef8414`:

- Six new regressions cover sixteen shared backend helper modules, actual shared
  HTTP consumers, LoRA's independent TCP owner, command failure propagation and
  Node-versus-browser prerequisites. The first five fail on the audited selector
  (twenty assertion failures, including sixteen subtests); the Node control fails
  before its repair (five assertions). All **38 selector/entrypoint methods** pass
  after correction, including the existing real Git, discovery and failure guards.
- The real shared-HTTP plan completes normally: **72 logical passes, 80 subtest
  passes and ten unchanged capability skips**, 82 collected logical cases and
  162 JUnit suite entries. Native JUnit wall is **191.031 s**; the complete selected
  HTTP/lint/entrypoint envelope is **191.807 s**. Browser/build setup is explicitly
  non-applicable. All owned TCP service processes have exited.
- The native LoRA TCP owner reports one truthful skip with no reference supplied.
  Supplying a nonexistent model root instead produces one failure, zero skips and
  the original `FileNotFoundError`. This verifies prerequisite truth, not actual
  full-model LoRA/QLoRA acceptance. Ruff/format and scoped diff checks pass.

The first shared-HTTP execution at `ef4ac78eb50033753f9a9fb01e2fd91bd262d33a`
is retained: one missing-Node setup failure, seventy logical passes and ten
capability skips, normal exit 1 at **191.595 s**. The isolated shell lacked Node;
inspection also found CI omitted Node setup for the selected independent TCP
architecture semantic oracle. The corrected workflow provisions the existing
Node 24 version for that owner without installing npm/Chromium or building UI.
The repaired exact-target run above passes; no assertion or timeout changed.

Complete final-diff selection resolves the same backend, HTTP, component and
production-browser files/portfolios as the audited head. The redundant raw backend
file plus `tests` entry now normalizes to `tests`; effective native targets are
identical. Backend source/tests/config/dependencies, API, UI source/tests/config,
the integration/wheel entrypoints and normative specifications are byte-identical
to the audited head. Its passing primary/wheel, component and extended evidence
therefore remains applicable; only changed routing/HTTP controls are freshly run.
The **561.084 s** complete routine measurement remains explicitly tied to its
earlier target, not a new timing sample of this repair. Six logical HTTP tests and
twenty subtest invocations are added; no further coverage retirement or performance
percentage is claimed. Final report-only publication is reconciled separately.

Native reports, both negative-control logs, the missing-Node failure, final plans,
source reconciliation and cleanup evidence remain outside Git under
`/tmp/issue-277-evidence/audit-return/`. The prior failures and all unavailable
reference/CUDA qualifications remain visible.
