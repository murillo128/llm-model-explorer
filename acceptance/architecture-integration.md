# Integrated Architecture navigation acceptance — issue 123

**Acceptance is incomplete: the repeated-return resource gate fails.** The
integrated canvas retains obsolete layouts while it remains mounted. This is a
product correction outside this evidence child's permitted test/harness scope;
#123 returns to design with a draft regression PR, not a `review-ready` handoff.

## Scope and provenance

This acceptance joins #119's preservation oracles, #120's contextual controls,
#121's source-owned Attention/MLP groups and #122's optional isolation. The
activation revision is `f7764fd69fffde644dd484ba2f7ca851123f2206` on
`codex/epic-issue-124`. All four contributing issues are completed. The exterior
and scientific-view baseline remains the integrated #114 revision
`c71f1b81bedfb534a6147f5c119ab0cfbf2cfeee`, captured by
[issue 119](architecture-safety.md).

This child changes acceptance observers/tests and corrects the API document's
obsolete implementation-status sentence. It changes no application, producer,
projection, layout, renderer, decoder or numerical expectation. Templates remain
outside this delivery. Fixture success and complete local checkpoint evidence
are reported separately.

## Required behavior and independent checks

| Requirement | Executable evidence |
| --- | --- |
| Conserved computation across grouping | `test_grouping_preserves_the_accepted_operation_level_contract` exports seven independently authored family cases and runs #119's directed multiset oracle against pre-grouping fingerprints. It contracts group forwarding only; operations, attributes, shapes, parameter identities, repetition order, fan-out, residuals and state ownership remain checked. |
| Changed counts, dimensions, orders and partial descriptions | Backend dense/Qwen3.5/V-JEPA suites and `architecture_grouping_cases.py`; UI projection, scope and automatic-layout suites. Both V-JEPA stacks and both hybrid attention variants remain distinct. |
| Contextual controls and exhaustive access | `architecture-controls.spec.ts` checks real keyboard search/popovers without layout or camera mutation. Production `architecture.spec.ts` checks received source IDs in search and all original nodes/edges in exhaustive projection. |
| Exact connection interaction | `architecture-connections.spec.ts` targets input/output dots, shared trunks, exclusive branches, nested forwarding, residuals and separate K/V state with native pointer/keyboard actions. Exact edge/endpoint sets and inspection labels are checked together with unchanged nodes, ports, routes, camera and layout count. |
| Reversible bounded isolation | `scope.test.ts` checks source paths, external re-entry and exclusion accounting. `architecture-isolation.spec.ts` checks unrelated-model growth, nested Back, View in model, cross-stack search, partial graphs, retry and obsolete worker callbacks. |
| Live session replacement and repeated return | New production tests repeat eight nested scope/Back cycles and count collectable layouts/source identities through native worker observations, and separately delay an actual HTTP response while replacing the scoped model/session. The resource check fails; see the finding below. |
| Scientific modal | Production tests inspect concrete native later-layer vectors and native matrices. GPTQ/NVFP4 fixtures now select isolated later-layer parameters and require their exact data request, plus independently calculated scalar values. Existing tests retain progressive cancellation, statistics, focus restoration and higher-rank/unavailable limitations. |
| No V-JEPA text requests | Both production family tests and the new delayed-session replacement test require no `/tokenize` request. |

The new observer uses weak references to objects already passing through native
worker calls. It stores no graph/layout copies and changes neither arguments nor
results. After explicit Chromium GC, the repeated-return test compares the same
returned view after two warm-up cycles and after eight cycles. Retained layouts
must not grow; one original source graph remains, and every completed layout
worker must terminate. The mounted canvas identity uses a weak reference, so the
test does not pin its detached DOM. Heap readings are
labelled main-thread JavaScript observations, not peak whole-browser RSS, GPU
memory, or a hardware-independent performance guarantee.

## Material finding: obsolete layouts retained by mounted canvas callbacks

On the built UI and real reduced SmolLM2 backend, every cycle enters the last
layer, enters its Attention component, returns to the layer and returns to the
original global context. Exact camera, selection, records and mounted-canvas
identity restore, and no graph/network request occurs. Nevertheless all completed
layout responses remain strongly reachable after explicit garbage collection:
**6, 11, 16, 21, 26, 31, 36, 41** retained layouts over eight cycles. The source
graph count stays one, workers return to zero, and visible geometry stays at
12 nodes / 14 routes in bounds 1442 × 636. Main-thread used heap grows from
8,990,424 to 12,670,108 bytes in the final regression run.

A separate control disables Playwright tracing and removes its persistent
element handle: **7 then 12** layouts remain after two cycles. After switching
to Tensor Explorer, **zero layouts and zero source graphs** remain. Thus whole
explorer disposal works, but repeated navigation on the mounted canvas does not
release prior layouts. This is not explained by active workers or by a per-view
copy in `GraphViews`.

The local Chromium heap snapshot identifies a strong retaining path from the
mounted **Model overview** button's React `onClick` property through successive
closure contexts to old `{ layout, options, invocation }` result objects.
`ArchitectureCanvas.tsx` creates the relevant navigation/inspection callbacks
and stores worker results. This establishes retained application state; the
exact callback-lifetime correction remains the owning UI task. Raw heap dumps,
traces and generated graphs stay outside Git. A compact receipt records the
observations and sanitized retainer description in
[checks.json](evidence/architecture-integration/checks.json).

The first test draft assumed at most two retained React layouts. Diagnosis
replaced that undocumented absolute bound with the actual issue requirement:
no growth across identical warmed return states. This still fails and is not a
tolerance increase to accept the defect. A diagnostic session-switch experiment
also exposed a Playwright/CDP stall when installing network interception after
many terminated workers; the separate session test installs interception before
layout begins. Neither observation changes product code.

## Executed checks and limits

The runtime remains exactly the activation revision; test-file hashes and source
tree identities are retained in the receipt. The local checks ran on Python
3.12.14, PyTorch 2.14.0+cpu, Node 24.14.0 and Chromium 153.0.8010.12.

| Check | Observed result |
| --- | --- |
| Full backend unit suite | 1,224 passed; 19 explicit CUDA-unavailable skips; lint, formatting and strict typing passed |
| Independent API validation | OpenAPI 3.1 valid; 283 reference resolutions, 88 instance cases, 144 Architecture cases, 76 reproducible wire fixtures |
| UI check | 731 unit tests in 33 files passed; generated bindings, TypeScript, ESLint and production build passed |
| Pre-grouping semantic comparison | All seven independent family cases preserve every fingerprinted semantic section |
| Unchanged global views | 24 fixture/view combinations exactly match pre-isolation revision `5d3522272cae1a7da8afcdf47017702b7698b36c`, including projection, boxes, ports, routes, source edge IDs and bounds; only elapsed time is excluded |
| Built UI / live backend, DPR 1 | Six passed, one failed, no skips/retries: four complete reduced-family cases, progressive modal cancellation and late actual session-response rejection passed; repeated-return resource retention failed |

The deterministic production cases include authored Attention/MLP groups,
compact/exhaustive views, isolated boundary fan-out, both hybrid variants, both
visual stacks, concrete native and fixture-decoded inspection, focus/camera
return and V-JEPA's absent tokenizer. Their graph bounds, layout durations and
labelled heap observations are retained. They do not establish actual-checkpoint
acceptance or a complete visual review.

Full component-browser, production DPR 2, all four complete local checkpoint,
remaining TCP, exterior screenshot and final performance comparisons were
deferred after the material failure. The existing operator manifest was found,
but no fresh actual-model PASS or missing-model SKIP is claimed by this child.
These are explicit release-review gaps, not waived requirements.

## Completion boundary

The controlling issue says: “A material product defect goes back to its owning
issue or design.” This lifetime defect violates the accepted repeated-return
invariant in the #122 integration. A corrective UI contract must break obsolete
layout retention while preserving the same canvas, exact Back state, source
identity, routing, hit testing and numeric inspection. Do not fix it by remounting
the canvas on each navigation, clearing valid history, dropping graph records,
raising timeouts or weakening the regression.

After correction, resume #123 on the reconciled integration revision and run
the full component browser, live TCP, production DPR 1/2, actual checkpoint,
visual exterior and performance comparisons. Prior sibling results remain
historical evidence; they do not replace those pending integrated gates.

## Reproduction

Use the repository's locked dependencies and the approved local manifest from
[Architecture acceptance](architecture.md). Keep offline model access and bound
CPU native threads. Use one heavy suite at a time; browser suites get separate
ports, output directories and an isolated Xvfb display.

```sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/backend/src"
# Set LMEX_ARCHITECTURE_REFERENCES to the operator-owned local manifest.
export LMEX_REQUIRE_ARCHITECTURE_REFERENCES=1
api/.venv/bin/python api/validate_contract.py
(cd backend && .venv/bin/ruff check . && .venv/bin/ruff format --check . &&
  .venv/bin/mypy && .venv/bin/python -m pytest)
backend/.venv/bin/python -m pytest acceptance -ra
(cd ui && npm run check)
(cd ui && UI_TEST_PORT=22620 xvfb-run -a \
  --server-args='-screen 0 1600x1200x24 -extension GLX' npm run test:browser)
(cd ui && UI_TEST_PORT=23620 xvfb-run -a \
  --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm run test:acceptance -- architecture.spec.ts)
(cd ui && UI_TEST_PORT=24620 xvfb-run -a \
  --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm run test:acceptance -- product.spec.ts --grep 'architecture safety baseline')
```

Disabling Xvfb GLX avoids this host's previously recorded NVIDIA EGL startup
failure; Chromium still uses the suite's SwiftShader WebGL2 path. CPU backend
and software WebGL results are not CUDA evidence. Missing optional local assets
must remain explicit SKIP; the strict Architecture reference gate rejects them.
