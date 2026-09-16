# Resumed integrated Architecture acceptance — issue 123

Validation status: **PASS** for the bounded integrated acceptance. This report records the resumed
integration after the separately accepted [layout-lifetime correction](architecture-lifetime.md).
The [original failure report](architecture-integration.md) and its receipt remain
byte-for-byte unchanged. The original eight-cycle regression is also unchanged;
its block hash is recorded with the resumed evidence.

## Source and scope

The scheduler's integration revision is
`02ab3bd334381525ec3fab72e8187329b0a4cf56`. The full acceptance source is
`4e876ed6d67ec9b0b2703f9607d66f65e5121603`; the additional exhaustive-detail
capture is at `463b23a` (only a new test case is added). It combines #119's preservation
oracles, #120's controls, #121's source-owned groups, #122's isolation and #149's
callback-lifetime correction. This child changes only acceptance code/evidence
and narrow documentation consistency fixes.

The resumed harness adds overview/exhaustive captures and actual later-layer
GPTQ/NVFP4 inspection. The latter selects the concrete operation in isolation,
requires exactly its tensor-data request, checks adjacent keyboard-selected
values, then closes and releases the shared Matrix Explorer. Its bounded
physical reads reuse `scalar_reference` from
`backend/scripts/check_quantized_reference.py`, without calling the production
decoder. Ten cross-shard adapter samples match the existing independent fixture
oracle byte-for-byte. Native checkpoint samples still use independent Safetensors
slices; synthetic packed cases retain their separate authored formulas.

Compact records: [checks and source identities](evidence/architecture-integration-completion/checks.json)
and [graph observations](evidence/architecture-integration-completion/graph-observations.json).

## Validation

| Gate | Result |
| --- | --- |
| UI bindings, TypeScript, lint, unit tests, build | PASS: 734 tests / 34 files at `5a71df5`; the application source tree is identical at the final acceptance source. Final harness type/lint checks also pass. |
| Full component-browser suite | PASS: 504 cases, all projects, no skips/retries |
| Independent API validator | PASS: OpenAPI 3.1, 283 references, 88 instance cases, 144 Architecture cases, 76 reproducible wire fixtures |
| Backend suite | Retained first-activation result: 1,224 passed / 19 CUDA-unavailable skips, lint/format/type checks passed. Backend source, tests, config, lockfile and API Git objects are identical; this is not represented as a fresh run. |
| Live TCP acceptance | PASS: 32 cases; one explicit CUDA-unavailable skip; all actual checkpoints present |
| Production Architecture / all actual references, DPR 1 and 2 | PASS: 24 full cases plus two separate global-detail cases, no skips/retries; all four complete checkpoints and all reduced families |
| Complete production scientific-view suite, DPR 1 and 2 | PASS: 64 cases, no skips/retries, including actual SmolLM2 Base |
| Pre-grouping directed-multiset conservation | PASS: all seven independent reviewed cases, run through the production producer export |
| Exact unchanged global layouts | PASS: 24 fixture/view combinations match `5d3522272cae1a7da8afcdf47017702b7698b36c` exactly, excluding elapsed milliseconds only |
| Exterior and scientific-view pixels | PASS: all six comparisons have zero changed pixels and identical shell geometry |

`ui/tests/architecture-connections.spec.ts` checks exact input/output port dots,
shared trunks, exclusive branches, MLP/nested boundary forwarding, residuals
and separate K/V state under native pointer and keyboard interactions. Its
independently authored expected source-edge sets and `stableState`/`unchanged`
checks retain labels, camera, node/port/route geometry and layout counts.
`architecture-isolation.spec.ts` checks exact nested Back, View in model,
exhaustive reachability and late worker/error recovery. Controls, shell and
inspection suites cover partial/unavailable graphs and modal failures.
These detailed component-browser oracles passed; the built HTTP/reference
suite separately verifies real Qwen boundary-port/trunk emphasis and complete
source reachability. Full-size, changed-order/dimension/count, partial/unavailable, retry, cancellation and stale
worker/session cases remain in the passed component suite. Actual checkpoint
receipts identify the approved revisions, content fingerprints and admitted
capabilities separately from reduced fixtures; no model download occurs.

## Resource and performance observations

The separate trace-disabled controls pass all four cases. At both DPRs the
original eight-cycle and extended sixteen-cycle tests retain two completed
layouts and one source graph at every post-GC sample, with zero active workers.
The mounted canvas, camera, selection, records and 1442×636 geometry (12 nodes,
14 routes) remain exact. Teardown retains zero graphs/layouts. The test compares
warmed growth; two is an observation, not a new fixed acceptance threshold.

Post-GC used JS heap goes from 8.36 to 9.00 MB (DPR 1, eight cycles), 8.34 to
9.00 MB (DPR 2), 8.32 to 9.31 MB (DPR 1, sixteen cycles), and 8.32 to 9.30 MB
(DPR 2). After teardown it is 7.99 MB at each DPR. These whole-heap observations
include runtime/test bookkeeping and do not prove zero allocation growth;
weak observers independently establish bounded graph/layout lifetime.

Actual-reference DPR 1 measurements are below. Counts are nodes/routes; source
counts retain all source edges, while visible routes contract transparent group
forwarding. The receipt preserves both DPRs, per-state heap fields and exact
scope identities.

| Model | Source nodes/edges | Compact nodes/routes; bounds | Exhaustive nodes/routes; bounds | Compact / exhaustive layout ms |
| --- | --- | --- | --- | --- |
| SmolLM2 | 1031 / 1596 | 12 / 10; 1228×740 | 1031 / 1236; 140270×7685 | 147 / 1647 |
| Qwen3 | 1019 / 1546 | 12 / 10; 1228×740 | 1019 / 1210; 136302×7239 | 175 / 1321 |
| Qwen3.5 | 972 / 1377 | 13 / 8; 1276×636 | 972 / 1168; 116518×4826 | 132 / 995 |
| V-JEPA2 | 854 / 1259 | 28 / 33; 3526×977 | 854 / 965; 118322×7441 | 261 / 1693 |

Isolated Attention remains 28/30 at 3224×577 (SmolLM2), 30/32 at 3414×577
(Qwen3), 32/35 at 3430×734 (Qwen3.5), and 17/18 at 2020×532 (V-JEPA2).
Language MLPs remain 8/7 at 1228×380; the visual MLP remains 6/4 at 1022×252.
All 36 matching isolation observations preserve #122's counts, camera,
component box and boundary fan-out exactly. Fixture-vs-complete-model and
altered-size component tests establish isolation from unrelated model size.

Main-thread CDP heap observations are not browser
RSS or GPU memory. Uncollected graph observations and post-GC lifetime
observations remain distinct.

The resumed full-Architecture launcher's SIGTERM initially appeared to terminate
its suite, but the Playwright child continued. Redundant reference runs overlapped
temporarily before their verified process tree was stopped. The final result
count uses the original complete suite once; redundant runs are excluded.
Timings during this overlap are explicitly observational, not a controlled
performance comparison. Resource controls and the exact global comparison run
separately afterward. Raw logs, traces, temporary caches and checkpoints stay
outside Git.

## Presentation

Reviewed captures retain two-row contextual controls, concrete instance/variant
identity, explicit isolation breadcrumbs and normal canvas panning. Dense Qwen
MLP shows gate/up branches and residual context; hybrid captures distinguish
linear and full attention; isolated Qwen shared-trunk emphasis follows the
correct branches; V-JEPA retains its encoder context and no tokenizer capability.
The nested-return capture at DPR 2 retains the same selected concrete layer.

Wide components deliberately extend beyond the initial readable camera. Explicit
Fit shows all boundary context at a smaller scale. The initial exhaustive capture
retained a prior off-content camera, so a separate capture test uses Find component
and Center selected to show readable global detail while requiring every source
node/edge to remain represented and no HTTP refetch. This corrects the evidence
capture, not application camera behavior.

At 1178×900 and 1440×900, full Tensor and Tokenizer captures are byte-identical
to the integrated #114 baseline `c71f1b81bedfb534a6147f5c119ab0cfbf2cfeee`.
Architecture exterior differences are zero across 151,932 and 179,704 pixels,
respectively; all six shell-geometry comparisons are exact. The 390 px capture
keeps navigation and actions accessible with the accepted horizontal control
scroll and canvas pan, without document overflow. Both DPRs passed the safety
cases at 390, 1178 and 1440 px.
The comparison retains the original rounded Architecture-interior mask, including
the one-pixel panel border and corner pixels. Tensor and Tokenizer images are
compared in full at the same Chromium revision, viewport and DPR.

Selected original-resolution captures:

| View | Capture |
| --- | --- |
| Model overview | [Qwen3, 1178 px](evidence/architecture-integration-completion/overview-qwen3-1178.png) |
| Dense MLP and residual | [Qwen3, 1440 px](evidence/architecture-integration-completion/dense-mlp-1440.png) |
| Hybrid variants | [Linear, 1178 px](evidence/architecture-integration-completion/hybrid-linear-1178.png) · [Full, 1440 px](evidence/architecture-integration-completion/hybrid-full-1440.png) |
| Isolated shared-trunk emphasis | [Qwen3, 1178 px](evidence/architecture-integration-completion/isolated-qwen3-trunk-1178.png) |
| Isolated visual component | [V-JEPA2, 1178 px](evidence/architecture-integration-completion/isolated-visual-1178.png) |
| Exact nested return | [SmolLM2 fixture, DPR 2](evidence/architecture-integration-completion/nested-return-dpr2.png) |
| Exhaustive detail | [Qwen3, 1440 px](evidence/architecture-integration-completion/exhaustive-global-detail-1440.png) |
| Constrained shell | [390 px](evidence/architecture-integration-completion/constrained-390.png) |

## Reproduction and limits

Use the locked backend/API environments and Node 24.14.0 dependencies in an
isolated checkout. Set `PYTHONPATH` to that checkout's `backend/src` when reusing
an editable Python environment. Use the operator-owned manifest described in
[Architecture acceptance](architecture.md), `HF_HUB_OFFLINE=1`,
`TOKENIZERS_PARALLELISM=false`, and `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

```sh
export LMEX_REQUIRE_ARCHITECTURE_REFERENCES=1
# Set LMEX_ARCHITECTURE_REFERENCES and LMEX_REFERENCE_MODEL_DIR to approved local assets.
npm --prefix ui run check
backend/.venv/bin/python -m pytest acceptance -ra
cd ui
UI_TEST_PORT=27620 xvfb-run -a \
  --server-args='-screen 0 1600x1200x24 -extension GLX' npm run test:browser
UI_TEST_PORT=28620 xvfb-run -a \
  --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm run test:acceptance -- architecture.spec.ts
UI_TEST_PORT=29620 xvfb-run -a \
  --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm run test:acceptance -- product.spec.ts
UI_TEST_PORT=31620 xvfb-run -a \
  --server-args='-screen 0 1600x1200x24 -extension GLX' \
  npm run test:acceptance -- architecture.spec.ts --trace off \
  --grep 'repeated nested return releases obsolete layouts|extended nested returns stay bounded'
```

Run one heavy suite at a time, with separate output directories. Confirm child
process/listener teardown as well as launcher exit before starting another.
The environment uses CPU PyTorch and headed Chromium/SwiftShader under Xvfb
without GLX. These results do not establish CUDA, hardware throughput, inference
correctness or support for unselected checkpoint variants. Templates remain
outside this acceptance.
