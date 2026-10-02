# Production acceptance ownership and density selection

Issue #276 compares the incoming integration snapshot
`648e457c76ffa7633deedacb15b66d466e0c0b47` with the scoped changes below.
This is test evidence, not a replacement product specification. #274's
[retained domain owners](../ui/evidence/domain-test-ownership.md) and #275's
[fixture preparation no-op](test-cost-audit.md#fixture-preparation-decision--issue-275)
are inputs; their savings are not counted again.

DPR 1 selects every case, including new untagged cases. The native `@density`
tag selects additional DPR 2 invocations. There are no new runtime skips,
title-derived selection rules, changed worker/retry settings or timeout increases.
Use `npm run test:acceptance -- --list` from `ui/` to inspect the expanded matrix.
Absent references retain their explicit optional status; supplied invalid inputs
and required-reference mode still execute their existing validation in DPR 1.

| File | Before DPR 1 / DPR 2 | After DPR 1 / DPR 2 |
| --- | ---: | ---: |
| `product.spec.ts` | 35 / 35 | 32 / 14 |
| `architecture.spec.ts` | 24 / 24 | 24 / 18 |
| `clm.spec.ts` | 1 / 1 | 1 / 0 |
| `kev.spec.ts` | 1 / 1 | 1 / 0 |
| `lora-hierarchy.spec.ts` | 1 / 1 | 1 / 0 |
| `lora-reference.spec.ts` | 2 / 2 | 2 / 0 |
| **Total** | **64 / 64** | **61 / 32** |

The 35 removed invocations comprise six domain-input repetitions (three inputs
at two densities) and 29 second-density repetitions. Two of those 29 are optional
LoRA/QLoRA reference invocations: without references, 33 runnable fixture
invocations are eliminated. None of the existing unavailable deterministic
MoE/reference skips is presented as a saving. The three removed functional
inputs retain their exact TCP owner in the default full local path.

## Distribution inputs: exact retained proofs

The reviewed requirement is [API row/column distributions](../docs/spec/api/contract.md#row-and-column-distributions):
100 full-finite-range bins, exact uint32 counts, bin 50 for a constant, null
endpoints/zero counts without finite values. [Tensor Explorer](../docs/spec/ui/tensor-explorer.md#progressive-composition-and-density-baseline)
requires independently arriving results and fixed density normalization;
[rendering](../docs/spec/ui/rendering.md) keeps robust luminosity anchors separate
from the authoritative distribution domain.

| Old browser case | Exact retained owner / independent oracle | Remaining production bridge |
| --- | --- | --- |
| `production distribution scale is truthful for scale.concentrated` | `acceptance/test_distribution_scales.py::test_scale_domains_and_exact_counts`, `concentrated` input: explicit float32 samples, independent NumPy float64 endpoints and manually accumulated counts. `ui/tests/distribution-scale.spec.ts` asymmetric case, DPR-2 `subnormal`/`signed asymmetric` decimal cases and `ui/src/rendering/distribution-scale.test.ts` small nonzero formatting case retain zero placement and small-number display. | `…for science` carries real asymmetric `[-2,6]` metadata through TCP, decoder, controller, renderer and React header, including 25% zero. `ui/tests/renderer.spec.ts` `robust contrast: tightly centered` independently checks real luminance separation; small magnitude does not choose a separate binding/transport branch. |
| `…for scale.outliers` | Same TCP owner, `outliers` input: true `[-1000,3000]` endpoints despite p01/p99 both zero, exact counts. `backend/tests/test_tensor_analysis.py::test_native_numerics` outlier inputs. Component-browser `outliers: authoritative domain, bin placement and independent density` checks every rendered bin and late robust statistics without changing the full domain. Renderer `robust contrast: outlier-heavy` and `late robust statistics, constant/fallback/extreme values and transfer-only redraws` independently retain luminosity/fallback pixels without new scalar uploads. | `…for science` retains finite scale/header binding; the real progressive pixel case and two-card scientific parity retain statistics/transfer wiring. Outlier magnitude changes values, not endpoint selection or UI ownership. |
| `…for scale.constant` | Same TCP owner, `constant` input: `[2]*400`, exact bin-50 counts/domain. Backend `test_native_numerics` constants/signed zero. Component-browser `constant zero` and `constant positive` cases assert actual pixels, count storage and stable scale after statistics. `TensorHeader.test.tsx` full-range metadata case explicitly rerenders constant then nonfinite; `distribution-scale.test.ts` verifies bin-50 caption/no zero guide. The renderer late-statistics case independently asserts constant midtone pixels and transfer-only redraws. | Finite `science` and null `scale.nonfinite` retain the shared metadata callback and distinct fallback branch. Constant wording/zero-span behavior stays at the real header/scale owners rather than repeating it through a process launch. |
| `…for science` | Kept at DPR 1. Independent literal endpoints; exact scientific float32/statistics/counts also remain in `test_network.py::test_ten_operations_scientific_values_and_unicode`. | Added real distribution-producer barrier: DATA/status/domain/header arrive while the producer is held and no distribution artifact is valid; completion preserves header metadata. |
| `…for scale.nonfinite` | Kept at DPR 1; same complete TCP matrix and component-browser `nonfinite` case remain. | Real null-domain metadata produces no endpoints and unavailable min/max, before and after successful stream completion. This exceptional UI fallback remains a separate small scenario. |

The stream controller passes all domains through the same `distributionDomain`
callback. Matrix Explorer forwards it to the actual row/column renderer scales
and its React header. Neither dispatches by tensor name, magnitude or model
family. Constant handling occurs in the scale/header owners cited above; the
null-domain branch remains integrated. Expected endpoints/counts are never
computed by the implementation under test. Numerical fixtures/generators and
all lower-layer assertions remain unchanged.

## Every product family and its integration risk

Titles below identify the current cases; parameterized rows include every listed
input. All rows retain DPR 1 unless the three input relocations above say otherwise.
“Both” means explicit `@density`, never inference from a title.

| `product.spec.ts` family / inputs | Production boundary risk | Density |
| --- | --- | --- |
| `production UI renders before producer completes; native geometry, inspection and cleanup` | First actual populated pixels before publication; seed-17 float32 values, physical extents, all 81 magnifier pixels, linked profile pixels, bounded scalar uploads and navigation/session cleanup | Both: one physical pixel per cell, DPR-to-scroll/hover conversion |
| `live tokenizer uses real Unicode IDs/spans and suppresses delayed old responses` | Actual Unicode tokenizer echo/overlapping spans, delayed real-response fence and reconnect | 1: text/session ordering has no density-dependent oracle |
| `cancel and network disconnect…`; `real producer errors…` | Browser abort/offline handling, pre-META auxiliary failure versus midstream tensor failure, incomplete state/no partial cache/resource release | 1: operation ownership and terminal states |
| `local reference Base opens normalization, both MLP orientations and embedding with exact samples` | Actual checkpoint source-to-HTTP-to-native matrix inspection | Both: native strip height and pointer at half a device pixel |
| `real ordered embeddings render progressively…` | Ordered duplicate IDs, real first-row barrier, exact uploaded values/requested-row reads, annotation linkage and cleanup | 1: full logical row matrix remains; physical scroll/selection bridge below runs both |
| `real qwen3` / `real qwen3_5 input embeddings…` | Each admitted input-source resolver, unsupported-model isolation/recovery and exact row linkage | 1: model/source capabilities |
| `real A→B→A response reordering…`; `repeated prompt edits and explorer unmounts…` | Delayed actual responses, same-ID/model/session replacement, cancellation and CPU/GPU lifetime | 1: generation and consumer ownership |
| `production native pane geometry`: 1000×700 / 390×640 | All four native overflow modes/four corners, independent inventory scroll, fixed document and exact linked origins | Both: device-coordinate origins and density-dependent extents; historical deadline failure retained |
| `production prompt pixels, selection, history and composition…` | Real embedding completion preserves exact prompt screenshot/selection/undo/IME; actual request delivery | Both: rasterized prompt equality |
| `integrated inventory preferences and metadata…` | Progressive header stability, mounted scientific canvas through inventory resize/reload | Both: pixel-snapped scientific origin/header alignment |
| `production matrix navigation centers underfilled data…` | Native underfill, region previews on three tracks, local history and unchanged scalar storage | Both: physical cell centering, 100-device-pixel tracks and exact selection |
| `integrated camera gestures, exact selection…` | Native wheel/touch/range wiring, independent logical bounds, profile scale and adaptive magnifier | Both: physical tracks, coordinate conversion and DPR-dependent inspection; historical failure retained |
| `production distribution scale…`: science / nonfinite | Real progressive domain/header delivery and exceptional fallback, as above | 1: numerical/metadata branches; component browser owns density/domain matrix |
| `production stale results remain visible…` | Real tokenizer/embedding replacement without missing frames; independent embedding camera | Both: region selection/profile previews composed with prompt layout |
| `production inspection tolerates DPR change before viewport resize notification` | Inspection event ordering while backing viewport still has its prior density | Both: explicit 1→2→1 and 2→1→2 stale-viewport transitions |
| `architecture safety baseline…`: 390 / 1178 / 1440 px | Exterior shell geometry/style and tensor/tokenizer continuity through graph navigation (existing structural graph fixture) | 1: self-set CSS widths; actual semantic graph bridge remains in architecture suite |
| `polish inventory captures…`: 1178 / 1440 px | Mounted scientific surface/profile alignment through collapse/restore and native focus | Both: snapped canvas edges and fixed physical profile depth |
| `polish real tokenizer auto sizing…`: 1178 / 1440 px | Real tokenization/embedding completion, content allocation and native divider/focus/scroll; large prompt unchanged | 1: panel CSS sizing; progressive/native scientific density is separately covered |
| `polish magnifier follows edges after scrolling resize DPR and source replacement` | Native camera/inspection placement through scroll, resize and source replacement | Both: real 1→2 and 2→1 transitions have different scrollbar bounds and origin clamping |
| `expanded model coverage links native embeddings in GPTQ and NVFP4 checkpoints` | Real native input-table selection alongside packed checkpoints, exact token-row values and non-text model isolation | 1: storage/model selection; no additional density-specific source resolver |
| `integrated two-card embeddings have real scientific parity…` | Independent seed/count oracle, three real requests, checkpoint/embedding scientific parity, full card wiring and resource stability | Both: native canvas dimensions, zoom/scroll and independent prompt geometry |
| `real embedding distribution cancellation…`; `real embedding statistics failure…` | Independent auxiliary consumers preserve values, successful tokenization and surviving analysis | 1: cancellation/failure semantics |

## Architecture and package families

These continue to use the production backend, real semantic graph/API decoder,
layout worker and projection path. No native package is replaced by static JSON.
No long scenario is combined or reduced in size/cycles: its retained production
risk is not established by generic widget tests with simulated transport.

| Family | Retained owner and integration risk | Density |
| --- | --- | --- |
| Precise model-owned load failure | Same `architecture.spec.ts` case; real duplicate-key sidecar becomes typed capability finding and exact safe API diagnostic in UI | 1: diagnostic/provenance binding |
| Port labels/cable clearance/terminal emphasis | Same production case; generated geometry, real port/line pointer targets, exact emphasis, no numeric request/layout/camera change, resize/isolation | Both: constrained rasterized label and port/line hit targets; #270 regression retained |
| Full graph/concrete bindings/logical modal: seven deterministic and seven actual-reference families | Same complete cases; actual graph/compact decoder, source conservation, layout/projection, native/decoded later-instance weights, unavailable binding and explorer restoration | Both: real port/line hits and native scientific inspection. All actual-reference/large MoE shapes and memory evidence remain unchanged |
| Kimi complete repeated experts; actual Kimi issue-178 expansion/router/expert; actual Qwen exhaustive detail | Same cases; full configured expert/graph size, far-instance culling/reachability and actual router/expert binding at representative widths | Both: readable rendered-scale targeting/culling and native reference weight inspection; existing prerequisite skips remain explicit |
| Close during progressive data | Same three-cycle case; exact prefix, actual first render, modal cancellation and zero operations/readers/tasks/cache temporaries | 1: consumer/modal lifetime, while density/rendering is owned above |
| Repeated nested return; extended mounted lifetime/teardown | Same eight-/sixteen-cycle cases and graph sizes; exact Back cameras/source records, layout-worker/WeakRef retention, GC and complete teardown | 1: same semantic layouts and cycle count; no device-pixel oracle. Size/lifetime proof is retained, not moved to a smaller fixture or optional job |
| Isolated session replacement rejects late response | Same case delays actual backend graph; model/session replacement rejects stale graph/scope/selection | 1: generation fence |
| Shared templates/distinct instance weights/cancellation | Same complete scenario; neutral structure cannot fetch values, concrete instances use distinct exact float32 values, cancel/remap maintains one canvas, eight switch cycles stay bounded | 1: binding and layout lifetime; numeric density/selection bridge remains above |
| CLM configured head; Kev adapted hybrid projection/factor/pointer | Same `clm.spec.ts` / `kev.spec.ts` cases with fresh real native packages/backend and actual weight HTTP requests | 1: native export/admission/semantic binding; no density-specific assertion. Supplied reference roots still validate rather than silently fall back |
| Adapted projections/hierarchy/formulas/A/B | Same `lora-hierarchy.spec.ts` case; exact children, ordinary expand/isolate/Back, source-backed formula/ports/scalars and A/B streams | 1: #269 hierarchy/formula/weight binding regression intact |
| Actual LoRA and QLoRA references | Same two `lora-reference.spec.ts` cases; complete native target set, NF4 base and actual A/B inspection | 1: package/provenance/weight binding; absent prerequisites remain optional, supplied invalid roots fail |

Generic gesture schedules/window navigation retain #274's literal oracles and
actual component-browser bridges. Exhaustive production graph scenarios retain
their complete assertions because model family, layout and concrete binding are
independent integration risks; none is replaced with a generic widget success.

## Real transport contracts kept independently

`acceptance/test_network.py::test_progressive_shared_late_and_slow_consumers`
keeps first DATA before completion, late join from byte zero, independent exact
bytes and warm-cache identity. `test_cancel_and_disconnect_release_owned_work`
keeps all four actions (`one`, `all`, `session`, `disconnect`), including cancellation
or session deletion without stopping the surviving shared consumer.
`test_real_stream_failures_never_publish_partial_artifacts` retains both failure
stages. Restart/source invalidation, CORS and exact scientific/tokenizer results
remain separate tests. `test_distribution_bytes_arrive_before_publication` retains
independent TCP progressive counts.

`acceptance/test_embeddings.py` retains ordered duplicate-row progressive bytes
and operation/session/socket cancellation; `test_polish.py` retains native/packed
source-resolution families; `test_architecture.py` retains prepared graph
cold/warm identity, missing-cache/restart/partial capability and complete expert
TCP tests. These default HTTP gates and the exact numerical/component owners
above stay in `acceptance/check.sh`. No workflow, command orchestration, fixture
hook, model download or shipped implementation changes belong to this issue.

## Validation and measured cost

Implementation revision: **`3f63b7de84fa88690a9404ac03242615d2b52c12`**;
its acceptance-source hashes match the tree used for validation. The following
commit only records this evidence. Baseline is the exact incoming revision above,
archived independently in `/tmp/issue-276-baseline`; Python source resolution was
pinned to each checkout and each had its own production build. Dependencies were
installed once before comparison. No other repository Actions runs were active
at the start; the shared host was not controlled as a quiet benchmark.

Environment: Linux 6.8.0-139 x86_64/glibc 2.39, Python 3.12.14,
PyTorch 2.14.0+cpu, Transformers 4.57.6, pytest 9.1.1,
FastAPI 0.141.1/Starlette 1.6.0, Node 24.14.0/npm 11.9.0,
locked Playwright 1.63.0/Chromium 153.0.8010.12. The actual probe reported
SwiftShader Vulkan through ANGLE, 8192 texture/renderbuffer/viewport limits.
Acceptance used one headed worker, zero retries, fresh backends/model/cache roots
and ports 29702–29705 (comparison), 29722–29725 (full run). Component owners used
one worker and ports 29740–29741. No references were supplied or downloaded.

One complete scoped before/after sample uses the five original distribution
inputs plus Unicode/late-response tokenization at both DPRs, versus the two
retained progressive distribution inputs and that tokenization case at DPR 1.
Both samples passed without skips or retries: **12 → 3 invocations**. The stronger
producer-barrier assertions are included in the after sample.

| Phase (seconds) | Before | After |
| --- | ---: | ---: |
| Production build wall, including npm | 0.83 | 0.97 |
| Browser command wall, including npm/Xvfb | 216.12 | 52.21 |
| Native browser wall | 215.673 | 51.762 |
| Sum of native attempt durations | 214.184 | 50.897 |
| Fixture generation sum (inside backend startup) | 0.458 | 0.107 |
| Backend spawn-to-ready sum | 93.618 | 21.654 |
| Browser setup sum | 1.778 | 0.433 |
| Body sum | 115.164 | 27.797 |
| Teardown sum | 2.819 | 0.798 |

These are selection savings in one focused sample, not lower latency per test
or a whole-pipeline percentage. Generation overlaps backend startup and must
not be added again. #273's component-project savings and #275's zero fixture
savings are not counted. The complete baseline application run remains cancelled
in #272's evidence; it is not a speedup denominator.

Validation on the implementation sources:

- Native collection reconciles all 64 old families: 61 remain at DPR 1, only the
  three documented domain inputs relocate, and all 32 DPR-2 cases carry the native
  density tag and also run at DPR 1. There are no new/missing unexplained families.
- Complete production selection: **63 passed / 30 explicit skips**, native wall
  **1353.341 s**, zero retries, failures, interruptions, global errors or unrun
  cases. Both historical deadline regressions passed at both densities with their
  unchanged assertions/deadlines. Missing actual references and unavailable tiny
  MoE fixtures account for the skips; these remain missing reference evidence.
- Exact component-browser owners: **37 passed**, 46.139 s, no skips/flakes/retries.
  This covers the complete distribution-scale file and the renderer's three
  tightly-centered/outlier/constant-fallback cases cited above.
- Exact TCP owners: **16 passed / one CUDA skip**, 86.88 s. All four independent
  distribution inputs, shared cancellation/session/socket/failure/no-partial-cache,
  progressive counts, restart/invalidation and ordered duplicate embeddings pass.
- Backend `test_native_numerics`: **19 passed**. Distribution scale, real header
  and Matrix Explorer unit owners: **30 passed** across three files.
- UI typecheck, lint, production build, source-hash reconciliation, relative
  documentation links and `git diff --check` pass. The existing bundle-size
  warning remains; it is not a new failure.

The first sandboxed baseline attempt could not connect Chromium to Xvfb and
failed at browser launch, before application assertions. It was excluded from
comparison; the same unchanged baseline then passed with host Xvfb access. This
is an execution-environment correction, not a configured test retry or a timeout
waiver. Raw native lists, the old-to-new collection map, JSON reports, JUnit,
harness phases and logs stay outside Git in `/tmp/issue-276-evidence`.

Reproduce from `ui/` with the locked Node environment and isolated ports:

```sh
npm run typecheck
npm run lint
npm run build
UI_TEST_PORT=29720 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  xvfb-run -a npm run test:acceptance
UI_TEST_PORT=29740 PLAYWRIGHT_WORKERS=1 xvfb-run -a npm run test:browser -- \
  distribution-scale.spec.ts renderer.spec.ts \
  --grep 'authoritative domain|readable labels preserve|rulers follow the histogram|late robust statistics|robust contrast'
npm test -- distribution-scale.test.ts TensorHeader.test.tsx MatrixExplorer.test.tsx
```

From the repository root, with `PYTHONPATH` selecting this checkout's
`backend/src` and root:

```sh
backend/.venv/bin/python -m pytest acceptance/test_distribution_scales.py \
  acceptance/test_network.py acceptance/test_embeddings.py -q -ra
backend/.venv/bin/python -m pytest backend/tests/test_tensor_analysis.py \
  -k native_numerics -q
```

The existing application/UI/backend/API workflows explicitly defer epic-child
PRs targeting `codex/epic-issue-*`; the routing workflow has no changed applicable
paths. Local scoped evidence supplies this child gate; the final aggregate epic
PR still owes all applicable CI. No workflow or command orchestration was changed.
