# Test cost and coverage baseline — issue #272

## Pinned scope and evidence

Activation baseline: **`f014a3447d9f86da7707417a1bd34347ad1cf27a`**, 2026-10-01.
Both `main` and `codex/epic-issue-278` contained this exact revision at inventory
time. The [parent activation decision](https://github.com/murillo128/llm-model-explorer/issues/278#issuecomment-5938413270)
released the earlier hold regardless of CI outcome. This report changes no
product behavior, test selection, fixture sizes, workers, retries or workflow gates.

Incoming work is integrated, not merely completed on another branch:

| Delivery | Integrated source |
| --- | --- |
| Native CLM #266 / Kev #267 / integration #268 | [PR #271](https://github.com/murillo128/llm-model-explorer/pull/271), [#281](https://github.com/murillo128/llm-model-explorer/pull/281), [#282](https://github.com/murillo128/llm-model-explorer/pull/282); merge commits `06b69be0`, `679094ac`, `f014a344` |
| Adapted projection hierarchy and formula consistency #269 | [PR #280](https://github.com/murillo128/llm-model-explorer/pull/280), merge `8934df7c` |
| Routed boundary order and regular terminal pitch #270 | [PR #279](https://github.com/murillo128/llm-model-explorer/pull/279), merge `15d8ca43` |

The activation search found no open PRs or newer overlapping implementation;
the other cost-reduction children #273–#277 were queued. Relevant baseline coverage
includes `test_clm_export.py` (52 parameterized cases), `test_kev_export.py` (69),
`test_lora_architecture.py` (11), `test_architecture_grouping.py` (8), and
`test_operation_cards.py` (9). UI owners include `CardSummary.test.tsx`,
`auto-layout.test.ts` (routed order, nested boundaries, fan-out and equal pitch),
`architecture-interfaces.spec.ts`, and the new production `clm.spec.ts`,
`kev.spec.ts`, `lora-hierarchy.spec.ts` (one case each, two DPR invocations each).
Their collection is included below; an incomplete production run does not prove
all of them executed.

Inspected exact-baseline Actions evidence, all attempt 1:

- [UI checks](https://github.com/murillo128/llm-model-explorer/actions/runs/36906375279), success;
  [native HTML report and per-test attachments](https://github.com/murillo128/llm-model-explorer/actions/runs/36906375279/artifacts/11185597924).
- [Backend checks](https://github.com/murillo128/llm-model-explorer/actions/runs/36906374974), both Python jobs successful; logs retain the slowest 25 phases.
- [Application acceptance](https://github.com/murillo128/llm-model-explorer/actions/runs/36906375007), **cancelled**;
  [HTTP JUnit, failure traces and screenshots](https://github.com/murillo128/llm-model-explorer/actions/runs/36906375007/artifacts/11186078553).

The design-time `98a279fd` and historical `8653837c` are not this baseline.
Historical [UI run](https://github.com/murillo128/llm-model-explorer/actions/runs/36866492723)
reported 24 s fast checks and 19m27s browser; historical
[Application run](https://github.com/murillo128/llm-model-explorer/actions/runs/36866492565)
reported HTTP 35 passed/10 skipped in 212.65 s and an unfinished browser run.
Neither cancelled run establishes end-to-end coverage or a speedup denominator.

## Environment and reproducible inventory

Lock/input SHA-256 at the baseline:

| Input | SHA-256 |
| --- | --- |
| `backend/uv.lock` | `f5f2bc667e0f431bf761ffe743bcc4b94e0115ca364e111c7dcb5391b6fc9fd9` |
| `ui/package-lock.json` | `f35b169f20ad45cf2d27f92f2863cfd757a1f6128303a574025c98f7105ef44a` |
| `api/requirements.txt` | `d410e21212d4f226bc26434bead7b3cdedcc6191fd95e1504493ec5df4f08fb1` |

The self-hosted Linux host has 16 logical CPUs and 62 GiB RAM. CI used Node
24.14.0, npm 11.9.0, uv 0.12.13, Python 3.12.14/3.14.7, pytest 9.1.1,
PyTorch 2.14.0+cpu, FastAPI 0.141.1, Starlette 1.6.0, Transformers 4.57.6,
Playwright 1.63.0 and locked Chromium 153.0.8010.12 (revision 1243).
The focused local check used Python 3.12.14 and the same locked application/Node
packages; local uv was 0.12.16. These are distinct setup environments.

CI component browsers used **two workers**, desktop 1440×900 and narrow 390×844,
plus the headed native-scrollbars project. Acceptance used **one headed worker**,
1440×1000, DPR 1/2, with per-case viewport overrides. Retries are zero.
Both browser configs request SwiftShader. The focused acceptance attachment
actually reports `ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)
(0x0000C0DE)), SwiftShader driver)`, texture/renderbuffer/viewport limits 8192.
CI component attachments observe those limits but do not record the renderer
string: the configured backend alone is not physical-GPU evidence.

Warm persistent Python environments, locked `node_modules` cache hits, installed
Chromium, failure-only traces/screenshots and opt-in framebuffer capture already
exist. They are not new proposed savings. Main CI ran component/browser and
acceptance concurrently; both backend Python jobs overlapped from 18:29–18:45 UTC.
Consequently these timings are a **contended CI baseline**, not an isolated quiet
performance sample. The local focused run had no active repository Actions jobs;
an unrelated long-running headless Chrome and a LAN backend remained on the host
(observed host load 1.68/1.97/5.12). They were not terminated or reset. It is not
a controlled quiet benchmark either.

Native collection commands, from the unmodified baseline (dependencies as in
[README](README.md)); keep their large outputs outside Git:

```sh
# repository root
backend/.venv/bin/python -m pytest backend/tests --collect-only -q
backend/.venv/bin/python -m pytest acceptance --collect-only -q
"$LMEX_CONTRACT_PYTHON" api/validate_contract.py
# ui/
npx vitest list --no-static-parse --json
npx playwright test --list --reporter=json
npx playwright test --config acceptance/playwright.config.ts --list --reporter=json
```

Vitest 5 defaults to static list parsing: it returned 323 templates here and did
not expand the parameterized runtime cases. The runtime list above returned the
1,038 cases actually executed by baseline CI. Counts below come from collection
and native artifacts, never source-text `test(` counts.

| Boundary | Collected cases / expansion | Observed baseline result |
| --- | --- | --- |
| Backend pytest | 456 distinct test functions → 1,964 parameterized cases, 41 files; each Python job repeats the same set | Each job: 1,944 passed, 20 CUDA skips; main uses 3.12/3.14, PR uses 3.12 |
| Independent API | 118 schema-instance cases + 262 architecture cases; 132 reproducible wire fixtures; 365 references resolved | Validator personally passed at the baseline; this is a custom contract checker, not 512 pytest tests |
| UI Vitest | 1,038 runtime cases, 57 files; one configured jsdom project | 1,038 passed |
| UI component Playwright | 394 distinct collected title paths: 382 common cases × desktop/narrow + 12 native-scrollbar cases = **776 invocations**, 42 files | 775 passed, one intentional narrow-only pane-loading skip, no failures/flaky outcomes/retries |
| `acceptance/test_*.py` | 32 distinct functions → **45 cases**, eight files; includes nine cheap CI-entrypoint orchestration checks | 35 passed, ten explicit skips; JUnit has zero errors/failures |
| Production Playwright | **64 cases × DPR 1/2 = 128 invocations**; product 35, architecture 24, LoRA reference 2, CLM 1, Kev 1, LoRA hierarchy 1 | Cancelled: list log records 36 passes, 14 skips, two failures through invocation 52; the other 76 lack terminal results |

Backend/HTTP function counts strip pytest parameter suffixes; browser logical
counts deduplicate only project names, retaining intrinsic DPR/shape/viewport
parameters as collected cases. Production PR/epic CI selects DPR 1 (64);
main/local full runs select both (128). Component PR CI selects desktop plus
native-scrollbars (394); main/local full runs select all projects (776).
This PR adds five cheap parser tests, so post-change HTTP collection is 50;
the baseline remains 45. No baseline case is removed.

No reference variables were supplied to these runs. HTTP skips comprise the
explicit issue-178 reference gate, four complete architecture references, the
LoRA pair, CUDA, two packed Qwen references and SmolLM2 Base. Browser skips
observed before cancellation include unsupported deterministic families and
missing actual references. CLM/Kev/LoRA hierarchy have deterministic fallbacks;
fixture success must not be called reference validation. Local model directories
may exist on the host but were neither selected nor downloaded for this audit.

## Measured cost and limitations

All times below distinguish elapsed wall time from sums across tests. No figure
is measured CPU usage. Actions step timestamps have one-second granularity.

| Gate | Setup/build/teardown | Test wall time | Sum of test/phase durations |
| --- | --- | --- | --- |
| UI checks | 43 s checkout/setup/cache/fast-check/install before browser; fast aggregate itself 33 s, including Vitest 13.18 s and build; 16 s after browser through job completion | Browser step 1,318 s; native report **1,316.178 s**; job start-to-completion 1,377 s | **2,571.637 s** test sum across two workers: Before Hooks 151.594 s, After Hooks 72.937 s, body/unattributed remainder 2,347.106 s |
| Backend 3.12 | 62 s job setup/sync/lint/mypy before pytest (mypy 56 s); 6 s after pytest, including wheel smoke | Pytest **1,448.67 s**; job 1,517 s | Only slowest 25 phases retained; no full per-test JUnit or phase sum available |
| Backend 3.14 | 66 s before pytest (mypy 61 s); 9 s after, including wheel smoke | Pytest **1,465.03 s**; job 1,541 s | Same evidence limitation; Python jobs overlap, do not sum as pipeline wall |
| Real HTTP + production build | HTTP JUnit 268.755 s (pytest 268.76 s); production build about 1.77 s | Application step 1,503 s until cancellation; job 1,538 s | HTTP testcase sum **265.314 s** includes setup/body/teardown. Top-25 console entries alone: setup 114.75 s, call 138.23 s; these are partial sums, teardown is unmeasured, not zero |
| Production browser | Started 18:27:32.854 UTC; interrupted around 18:48:03; startup/teardown not fully separated | About **1,230 s partial elapsed**, not complete suite wall | No finalized `acceptance.json` in the cancelled artifact; full test/harness phase sums are unavailable |

The UI fast aggregate did not emit independent elapsed times for every
type/lint/build subcommand. Do not invent them from its 33 s envelope. UI browser
hook sums do not include all worker/server initialization; body is an explicitly
unattributed remainder, not a sum of nested overlapping Playwright steps.

Highest measured component families, summed test seconds including hooks:

| Family | Expanded invocations | Sum (s) |
| --- | ---: | ---: |
| `architecture-controls.spec.ts` | 30 | 264.310 |
| `architecture-connections.spec.ts` | 20 | 212.205 |
| `architecture-isolation.spec.ts` | 20 | 188.900 |
| `tokenizer-embeddings.spec.ts` | 52 | 180.257 |
| `architecture-card-actions.spec.ts` | 16 | 154.679 |
| `architecture-interfaces.spec.ts` | 26 | 142.869 |
| `architecture-browser.spec.ts` | 24 | 136.048 |

All `architecture*.spec.ts` families together sum 1,583.519 s (61.6% of UI
test sum). This is a ranking, not predicted parallel-wall savings. The 66
`matrix-zoom.spec.ts` invocations sum 88.118 s; its timing-boundary subset (18)
sums 22.050 s. Eight `shell-viewport.spec.ts` invocations sum 15.794 s.

Backend hotspots retain distinct numerical/structural risk: complete expert
reconstruction in `test_compact_architecture.py` costs 72–93 s per top 3.12
case; NF4 requested-row order/duplicates/unaligned chunks costs 88.51 s;
complete Kimi/DeepSeek/GLM graphs cost 43.20/37.90/32.43 s. These are not deletion
candidates merely because slow. HTTP families sum architecture 85.426 s,
network 101.474 s, embeddings 31.859 s, polish 32.726 s, distribution 12.388 s,
entrypoints 1.439 s. The Kimi complete-expert TCP case alone is 61.433 s.

## Failures and focused evidence

Both current-baseline failures exhausted the unchanged **90,000 ms** test
deadline. Trace errors for `product.spec.ts:577` (1000×700 geometry) end in
`page.evaluate: Target page, context or browser has been closed`; the camera
case at `:848` ends in `locator.boundingBox` at `:922`, while inspecting profile
overlap. There is no recorded failing numerical/equality assertion in those
terminal errors. Readiness polling assertions (`200` versus temporary `0`) in
trace history are retried setup observations, not their terminal defect.

Focused reproduction used an unchanged `git archive` of the baseline in
`/tmp/issue-272-baseline`, its own built UI, ports 29400–29405 and fresh backends,
with source resolution pinned to that snapshot. No other checkout was reset:

```sh
# /tmp/issue-272-baseline/ui; use the locked Node PATH
PYTHONPATH=/tmp/issue-272-baseline/backend/src:/tmp/issue-272-baseline \
  UI_TEST_PORT=29400 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  xvfb-run -a npm run test:acceptance -- --project=dpr1 \
  --grep 'integrated camera gestures|compact shell, navigation and four overflow modes at 1000'
```

Both passed without changed assertions, timeout increases or retries: geometry
61.655 s, camera 67.803 s; native wall **130.744 s**, sum **129.458 s**.
Two existing `harness-timing` samples give fixture generation 97.832 ms,
spawn-to-ready 15,798.977 ms, browser setup 288.667 ms, body 112,902.950 ms,
teardown 305.661 ms. Product fixture generation is inside spawn-to-ready;
do not add it again. Architecture fixture generation uses a separate subprocess.

The observed CI failure cause is deadline exhaustion during real browser work;
concurrent heavy suites plausibly contributed, but the causal contribution is
not proven by a two-case run. **Retain/investigate** both regressions. Focused
passing evidence does not repair the cancelled complete acceptance gate.
No extra complete expensive suite was started: exact-target backend/UI/HTTP
evidence already exists, production completion is missing, and the host was not
quiet. The permitted reporting outcome records this gap instead of repeatedly
running the repository. Final comparative measurement still needs a complete,
equivalent-scope sample; no end-to-end speedup is claimed.

## Cost/coverage owners and proposed child actions

The owning contracts are [Tensor Explorer](../docs/spec/ui/tensor-explorer.md),
[Architecture Explorer](../docs/spec/ui/architecture-explorer.md),
[backend analysis](../docs/spec/backend/architecture-analysis.md),
[API semantics](../docs/spec/api/contract.md) and
[binary streaming](../docs/spec/api/binary-streaming.md).
Rows authorize no deletion here. A missing independent proof means retain until
the named child supplies it, with no coverage quota.

| Exact candidate / classification | Contract and independent expected result | Current matrix / owner | Retained proof, distinct risk and exact proposed action |
| --- | --- | --- | --- |
| `ui/tests/shell-viewport.spec.ts` four `fixed shell contains normal states…` cases; **consolidate in #273** | Fixed document, reachable shell controls and 52/28 px chrome; explicit DOM bounds and accessible actions from shell visual-language/Tensor layout | Four self-set viewports (1440×900, 1000×840, 390×640, 280×400) × desktop/narrow = 8; each overrides project viewport, same DPR/browser | Keep all four responsive cases under one project: proposed 8→4 invocations. Remove only the duplicate project invocations; retain native-scrollbar project and other tests that actually use project viewport. `architecture-camera.spec.ts` explicitly depends on the narrow project's geometry and is **retain**, not part of this deduplication. |
| `ui/tests/matrix-zoom.spec.ts:208` three `wheel gesture timing…` schedules; **migrate conditionally in #274** | Normative ≤180 ms consecutive-event coalescing and >180 ms split; literal schedules `[0,179,358]`, `[0,180,360]`, `[0,40,221]` and explicit undo states | Three schedules × DPR 1/1.25/2 × desktop/narrow = 18; whole file 33 cases/project = 66 | Current `camera-history.test.ts` proves token grouping, **not** elapsed-gap boundaries; `tensor-viewport.ts` owns the threshold. Establish all three independent timing oracles at that owner before reducing boundary repetition. Retain real wheel/trackpad/touch, focal coordinates, local history, current DPR bounds and scalar lifetime in `matrix-zoom.spec.ts:111/:151/:188/:226/:254/:269`, pixel/selection sibling specs. Exact removal set is the 18 boundary invocations only after equivalent faithful cheap proof exists; otherwise retain. |
| `architecture-browser.spec.ts:91/:151/:293/:346`, `architecture-controls.spec.ts:162`, `architecture-isolation.spec.ts:144`, `components-large` (48 instances); **split/investigate in #274** | Ordered source hierarchy, concrete instance identity, nested Back state and readable native navigation; authored exact IDs/state snapshots, source-conservation unit oracles | Seven collected cases × desktop/narrow = 14 invocations; these call the 48-instance fixture, unlike ordinary 4-instance connections/mixed-stack fixtures | Use a compact adversarial fixture for generic first/nonzero/last instance, mixed branches, distinct bindings and nested restoration. Keep long-list scrolling/virtualization, offscreen selection/Center and scale-sensitive navigation on 48 instances. In particular, retain `unrelated model growth leaves a small isolated component readable and its bounds unchanged` (`isolation:133`), whose large-versus-small comparison is the contract. Proposed replacements require assertion-by-assertion review of the other exact locations; no whole-file deletion. Cheap `projection.test.ts`, `invariants.test.ts`, `ArchitectureExplorer.test.tsx` cannot replace actual pointer/DOM risks. |
| `architecture.spec.ts` four `full-size … all instances, bounded document, layout and memory evidence` cases; **retain** | Full configured layer counts, exact far-instance reachability, virtualized DOM, bounded document/worker resources; fixture source identities plus explicit bounds/heap observations | Four synthetic reference geometries × desktop/narrow = 8 | Keep genuine large-graph proof even if small navigation cases migrate. These are UI stress fixtures, not evidence for checkpoint analysis. No proposed removal; no smaller fixture can establish the same scale risk. |
| Entire `ui/acceptance` second DPR repetition; **selective retention plan in #276** | Real production bundle + TCP/CORS/session/LMEX/stream lifecycle; seed-17 scalar values, explicit token IDs, exact bindings and independent numerical references | 64 functional cases × DPR 1/2 = 128; main/full both, PR/epic DPR 1. Product 35, architecture 24, references 2, three new native/hierarchy cases | Keep all functional cases at DPR 1 initially. DPR 2 candidates for reduction: non-geometric tokenizer response ordering (`product:275/:512`), transport lifecycle (`:324/:346/:553/:1547/:1563`), numerical scale input matrix (`:940`), and binding-only repetition. Retain DPR 2 pixel/physical-cell/scrollbar/hover/profile/magnifier/source-linkage risks (`:143/:577/:641/:761/:848/:1048/:1317/:1446`), and constrained graph/port hit targets. Backend unit owners and `test_network.py`/`test_embeddings.py` protect computation/sharing; real TCP and production event fences remain distinct. No blanket 128→64 claim; #276 must enumerate selected titles after #274/#275, preserving failures and references. |
| Repeated immutable inputs in `acceptance/test_network.py::Service`, `fixtures.py`, `architecture_fixtures.py`, browser product/architecture hooks; **amortize selectively in #275** | Model files read-only, content-derived cache identity, cold publication, per-consumer/session/process isolation; deterministic source bytes and before/after filesystem checks | Fresh model/cache/process per case; product repeats across 35 cases/project, architecture across 24 including skips/references; native CLM/Kev/LoRA hooks add separate generation | Prepare immutable seed/config variants once, then isolated per-test copies; keep fresh backend/cache/session/barriers. Product focused generation was only ~49 ms/case versus ~7.9 s spawn-to-ready, so do not promise startup savings from copying. Architecture generation includes process cost but full phase data is missing. Preserve mutation/invalidation/export-atomicity tests with writable private copies. Never share live backends or mutable cache/session state. CLM/Kev/LoRA hooks lack `harness-timing` and remain separate evidence gaps, not zero cost. |
| CLM/Kev/LoRA/formula/port-order regressions; **retain** | Native admission/configured heads and hybrid pointer semantics, A/B exact parameter identity, source-backed formulas, regular routed terminal order without rebinding | Backend 52/69/11/8/9 cases above; UI 35 auto-layout and 22 CardSummary cases; new production cases × both DPRs plus interface browser tests | Keep backend `test_clm_export.py`, `test_kev_export.py`, `test_lora_architecture.py`, semantic baseline/grouping and API graph checks; UI `auto-layout.test.ts`, `CardSummary.test.tsx`, `architecture-interfaces.spec.ts`, `clm.spec.ts`, `kev.spec.ts`, `lora-hierarchy.spec.ts`. Explicit source-backed formula strings, configured weights, independently authored ports/pitch expectations detect different defects. No proposed deletion; production transport/weight-modal behavior remains necessary. |
| Quantized expert/numeric/LMEX backend and API families; **retain** | Every expert/parameter identity, requested row order/duplicates, numerical decode and stream framing/terminal rules | Backend version matrix; independent API schema/wire fixtures; UI decoder split/chunk tests; selected real TCP/browser weights | Keep native storage/numerical/API/generator/wheel-smoke owners. Transport and installed-wheel tests exercise boundaries mocked unit tests cannot. Do not replay every numerical domain input through every browser, but no removal is established by these timings. |
| Normative API/example workflow routing; **repair in #277** | Independent contract + both consumers must validate applicable changes, even when no implementation file changes | `api-contract.yml` covers `docs/spec/api/**`, `api/**`; UI covers only `openapi.yaml`, `api/fixtures/**`; application covers backend/UI/API/acceptance paths; backend covers backend paths | A `docs/spec/api/contract.md` or `binary-streaming.md`-only change misses UI/application/backend consumer gates. `examples/clm/**`/`examples/kev/**` exporter-only changes also miss them despite runtime admission coupling. #277 should add conservative explicit consumer routing for these paths; protect full API fixtures/generated bindings and real integration. Epic children intentionally ignore `codex/epic-issue-*` PR bases, requiring scoped local proof; final aggregate PR must run applicable gates. No gate is changed here. |

The candidate table distinguishes supported responsive deduplication from work
that still requires new cheap proof or measured setup evidence. In particular,
port-pitch/formula changes are not license to recalculate expected graphs from
production, and a mocked fixture response cannot replace production TCP evidence.

## Audit implementation and handoff evidence

`report.py` now summarizes existing native browser attempt/status counts, retries,
unrun/interrupted tests, global errors, wall time, per-file test sums and the
existing `harness-timing` phases. Null/missing phases remain unknown; missing
artifacts/statistics raise instead of producing an empty successful suite.
It introduces no dashboard, persisted result cache, metadata requirement or
timing threshold. Five tiny independent artifacts test pass/fail/skip/retry,
overlapping phases, interrupted/unrun data and missing inputs.

Scoped validation: native collection and independent API validation above;
`pytest acceptance/test_report.py -q` (five passed); Ruff lint/format for the
acceptance files; the two-case baseline reproduction described above. No browser
run was added to validate Markdown. Large reports/lists/traces remain in the
linked CI artifacts and `/tmp/issue-272-evidence`; focused JSON is
`/tmp/issue-272-baseline/ui/test-results/acceptance.json`.

Follow-up #277 must update this same report with exact final revision, selected
invocation counts, newly added cheap proof, complete applicable gate results and
equivalent environment/setup/body/teardown comparisons. Do not sum overlapping
child savings or derive a percentage from either cancelled acceptance run.
