# Architecture safety baseline — issue 119

## Baseline and scope

The execution baseline is `c71f1b81bedfb534a6147f5c119ab0cfbf2cfeee`, the
integration of explorer-polish epic #114 (PR #134), pinned by epic #124.
The older graph milestone PR #115 / `3be018e1` and shared-trunk correction
`bd699369` are historical inputs. In particular, the current numeric contract
includes admitted GPTQ/NVFP4 logical matrices; their old metadata-only status is
not restored by these tests.

This increment changes tests and evidence only. It characterizes the accepted
source graph, recursive horizontal layout, native-pointer connection inspection,
and post-polish shell/matrix appearance. It does not establish that the finite
suite proves zero defects or that UI stress fixtures prove checkpoint support.
The [specification](../docs/spec/ui/architecture-explorer.md) owns requirements.

## Reusable checks and independence

[`ui/tests/architecture-invariants.ts`](../ui/tests/architecture-invariants.ts)
contains two narrowly scoped oracles, usable by Vitest and Playwright's Node
test process. It imports only API/graph **types**, never runtime projection,
grouping, connection-hit or layout code.

- `assertTraceability(graph, projection, exhaustive?)` extracts the existing
  projection suite's common checks. It verifies source-record identity, exact
  directed endpoints, original path continuity, group-only forwarding, source
  edge accounting and exclusive omission reasons. Exhaustive mode additionally
  checks all source nodes and the complete directed **multiset** of signals.
  A set of edge IDs alone would miss a lost path through two parallel forwarding
  branches; the controlled negative retains every original edge ID and fails.
- `semanticSnapshot` / `assertSemanticEquivalent` compare revisions by
  contracting only group ports. The independently written implementation
  eliminates boundary endpoints by composing incoming and outgoing relations;
  it does not reuse production path traversal. It preserves duplicate paths,
  input/output roles, state/context edge-kind transitions, operation types,
  attributes, formulas, port shapes, repetition order/index/variant and instance
  ownership, parameter bindings, physical/logical descriptors, aliases and
  actionable tensor identities. Open interfaces and partial diagnostics remain
  explicit. Identity/reshape/normalization operators are never transparent.

For an unchanged source graph, source IDs and records remain exact. For a
producer revision, supply a reviewed `SemanticNames` correspondence for changed
node, parameter, repetition and port IDs. Unlisted IDs retain their names;
many-to-one names are rejected. This is an explicit mapping, **not** a guessed
match by equal shape, label, node count or arbitrary reachability. Graph/edge
hashes, wrapper count and provenance revision strings are intentionally excluded.
Tensor inventory IDs are never remapped. Keep API conformance validation as a
separate prerequisite; these helpers are not a second graph schema validator.

Example for a later grouping child:

```ts
assertSemanticEquivalent(acceptedGraph, regroupedGraph, acceptedNames, revisedNames);
assertTraceability(regroupedGraph, projectGraph(regroupedGraph, {
  expanded: [], exhaustive: true,
}), true);
```

The positive control wraps a real Q projection in a new group, adds two boundary
segments, changes every graph-local ID and reverses serialized record order.
It remains equivalent under explicit correspondence. The semantic negatives
drop a residual, swap equal-shaped Q/K endpoints, use another instance's weight,
merge prior or next K/V states, cross instance ownership, duplicate a signal,
change data/state/context kinds, alter an operation/attribute/shape, reorder
variants/indices, change alias/tensor/storage/fused bindings, or remove partial
diagnostics. All expectations are authored independently of production grouping.

## Existing coverage retained

| Existing suite | Retained responsibility |
| --- | --- |
| `projection.test.ts` | Compact, first/last, contiguous windows, nonperiodic hybrid order, boundary forwarding, residual bypass, unconsumed interfaces, state/MLP filters, exhaustive restoration, partial/no-repetition graphs |
| `auto-layout.test.ts` | Generated recursive left-to-right serial geometry, independent branch rows, real endpoints, unrelated body/header/label avoidance, distinct signals, changed dimensions/count/order, both stacks, stable round trips |
| `connection-hit.test.ts` and `architecture-connections.spec.ts` | Native source/destination dots, shared trunks versus exclusive branches, equal-shaped ports, residual and K/V routes, focus/pin/clear restoration, unchanged camera/node/port/route coordinates and layout count |
| `graph.test.ts`, `ArchitectureExplorer.test.tsx`, connection and shell browser suites | Layout abort/timeout/late replies, explicit recoverable worker failure/retry, model/session replacement, local partial/unavailable errors |
| `architecture-inspection.spec.ts` and live architecture acceptance | Concrete native/packed numeric bindings, progressive bytes, modal close/cancellation, stale callbacks, focus return/trap, resource release and unsupported-rank metadata |
| Backend dense/Qwen3.5/V-JEPA architecture tests | Independent actual family topology expectations, dense Qwen-versus-Llama normalization/aliases, hybrid variant and state details, encoder/predictor ownership, changed dimensions and incomplete coverage |

The new common projection oracle also runs on each production architecture
received by `ui/acceptance/architecture.spec.ts`: dense SmolLM2/Llama and Qwen3,
mixed Qwen3.5, and V-JEPA encoder/predictor without tokenizer. It checks first
and last instances of every stack as well as exhaustive projection. Full-size
`referenceFixture` inputs remain explicitly synthetic capacity tests; their
generic interiors are not used as a Qwen/Llama semantic oracle.

## Visual preservation evidence

The `architecture safety baseline` cases in `ui/acceptance/product.spec.ts`
capture 1178×900 and 1440×900 built UI views: populated Tensor Explorer with
profiles, populated Tokenizer Explorer with embeddings, and Architecture.
Numeric/tokenizer results come from the production HTTP fixture service;
only the four-layer architecture graph is authored. The test checks unchanged
fixed shell geometry/styles when switching explorers, exact navigation geometry
on same-view return, and fixed document bounds. The existing active-tab font
weight changes the navigation content width (403.09375 px in Tensor versus
405.171875 px in Tokenizer on this host); each active view is recorded separately.
It records toolbar/repetition/focus-control rectangles and graph work area.

These captures are a comparison baseline for subsequent children. Current
Architecture control placement and the absent attention metablock are not
golden UX requirements. Scientific invariants, shell appearance and shared
Matrix Explorer behavior remain preservation requirements. No image tolerance
or expectation update may be used to silently accept a material baseline defect.

All six captures were opened and visually inspected:

| View | 1178 px | 1440 px |
| --- | --- | --- |
| Architecture | [capture](evidence/architecture-safety/safety-architecture-1178.png) | [capture](evidence/architecture-safety/safety-architecture-1440.png) |
| Tensor / Matrix | [capture](evidence/architecture-safety/safety-tensor-1178.png) | [capture](evidence/architecture-safety/safety-tensor-1440.png) |
| Tokenizer / embeddings | [capture](evidence/architecture-safety/safety-tokenizer-1178.png) | [capture](evidence/architecture-safety/safety-tokenizer-1440.png) |

[Visual measurements and image hashes](evidence/architecture-safety/visual-baseline.json)
retain both viewports and per-view shell observations. In this compact graph
state the navigation/repetition rows are each 39 px high, focus controls 27 px,
and the graph work area begins at y=201.5 with height 653.5 px. Its width is
1136 / 1398 px respectively. These are baseline observations, not new limits.

## Executed controls and validation

The owner-prepared [shared-trunk negative](evidence/architecture-safety/trunk-negative.json)
ran in a separate archive with only `connection-hit.ts` changed to return
`[edgeId]`. The existing native-pointer test failed at its exact emphasized-set
assertion: expected Q/K/V (`edge-112`, `edge-113`, `edge-114` paths), observed only
Q (`edge-112`). The unchanged resolver passes that test in both browser projects.
No production mutation or weakened expectation is included in this PR.

Local validation on 2026-09-15:

- UI gate: **710 unit tests**, TypeScript, ESLint, generated-binding freshness
  and production build pass. The new invariant suite contributes 29 cases;
  the existing projection suite contributes 23 reused cases.
- Architecture component browser suites: **46 passed**, desktop and narrow.
- Visual baseline: **2 passed**, headed Chromium at DPR 1, 1178 and 1440 px.
- Relevant backend dense/Qwen3.5/V-JEPA tests: **109 passed**.
- Independent API validator: OpenAPI valid, 283 references resolved, 88 instance
  cases, 144 architecture cases and 76 reproducible wire fixtures checked.
- Built-UI/live-backend Architecture acceptance: **9 passed in 8.9 minutes**
  at DPR 1: four deterministic checkpoints, all four complete local references
  and progressive modal cancellation. The common oracle passed on every graph;
  native matrix/vector samples and fixture packed values retained exact results.
  DPR 2 reference browser cases were not rerun for this issue.
- Architecture HTTP acceptance: **7 passed**, including all four required
  complete local checkpoints with cold/warm startup and immutable retrieval.

The [check receipt](evidence/architecture-safety/checks.json) records commands,
source-file hashes, browser case outcomes and layout/heap observations.
The [fresh reference receipt](evidence/architecture-safety/references.json)
records repositories/revisions, content fingerprints, selected file names/bytes,
quantization metadata, producer/source revisions, graph identities and cold/warm
observations. Startup scans the common approved model root; these timings are
not isolated per-model benchmarks. SmolLM2 has 971 nodes / 1386 edges; Qwen3 963 / 1350;
Qwen3.5 948 / 1329; V-JEPA 782 / 1079. All retain complete declared coverage.
Source-review/component fixtures, synthetic live integration and actual
checkpoint observations remain separate evidence.

The initial isolated UI run hit Vite's filesystem restriction on symlinked ELK
dependencies; copying the locked dependency tree into the archive resolved it.
Headed Chromium required access to its isolated Xvfb display. Baseline test
development corrected two test assumptions: active-tab geometry is view-specific,
and navigation must not use a setup helper that creates a replacement session.
The final tests retain exact assertions without pixel tolerances or production
changes. Existing third-party Python deprecation warnings remain non-failing.

## Reproduction

Use the repository's locked Node, UI, Python and Chromium environments. Run one
heavy suite at a time, with a separate `UI_TEST_PORT` and output directory.

```sh
npm --prefix ui run check
npm --prefix ui run test:browser -- architecture.spec.ts architecture-connections.spec.ts architecture-inspection.spec.ts architecture-shell.spec.ts
backend/.venv/bin/python -m pytest backend/tests/test_dense_architecture.py backend/tests/test_qwen35_architecture.py backend/tests/test_vjepa2_architecture.py -ra
api/.venv/bin/python api/validate_contract.py
backend/.venv/bin/python -m pytest acceptance/test_architecture.py -ra
# From ui/; built UI and real backend, headed Chromium:
xvfb-run -a npm run test:acceptance -- product.spec.ts --grep 'architecture safety baseline' --project=dpr1
xvfb-run -a npm run test:acceptance -- architecture.spec.ts
```

For actual checkpoints use the existing approved local manifest through
`LMEX_ARCHITECTURE_REFERENCES`, with `LMEX_REQUIRE_ARCHITECTURE_REFERENCES=1`
for the strict no-skip gate. [Reference reproduction](architecture.md) owns
provisioning and identity evidence. Missing models must remain explicit SKIP;
fixture success never substitutes for reference runs. Weights, caches, full graph
responses and bulky traces remain outside Git.
