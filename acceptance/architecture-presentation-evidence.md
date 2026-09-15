# Architecture presentation acceptance — issue 106

The Architecture panel now derives a compact, source-preserving view and lays out
visible dependencies horizontally at every expanded scope. Source graph/API,
application shell, native Matrix Explorer and backend behavior are unchanged.
The [UI specification](../docs/spec/ui/architecture-explorer.md) owns intended
behavior; this document records observed implementation evidence.

## Reference and intermediate checkpoint

The approved HTML was downloaded and opened in Chromium before editing. Its
SHA-256 matched `45bdc40c74192de94be1f73df5f1d1bf3150da0fa65b3bf2a4a6a493cae85e54`.
Both binding issue comments were applied. The original local graph is
`7a11207126ee52d510e406921fa8d0890fd7aba9bd4e6e1b7b5449cb2de14379`, descriptor
`transformers-qwen35-nvfp4` revision 1, reviewed Transformers revision
`2cba19507be799b7bef247ca6c1c4708bf881b5b`. Its artifact digest matched
`3969cbe6065a19ec74e0b538d34d89b646d754ae7bac5494224433970c6dbc3c`.
The artifact alone does not identify a commercial checkpoint.

A fresh independent review of projection commit `a7367ce` returned
**PASS_WITH_NOTES**, final-capable **no**, before UI integration. It checked
source identity, forwarding continuity, fan-out, exact instance ownership and
50 mixed view states. Its note about open-group input-as-source and
output-as-destination roles is covered by layout/port regression tests.

HTML, full original graph, response replay, model files and traces remain local.
CI fixtures are independently authored, including 24 hybrid layers, changed
count/order/size, two independent encoder/predictor stacks, no tokenizer,
nonperiodic variants and partial/no-repetition graphs.

## Generated geometry

ELK 0.12.0 supplies compound layered RIGHT placement and orthogonal routes with
explicit ports and label rectangles. This dependency meets the recursive
placement, containment, routing and size requirements; it is isolated behind the
existing asynchronous layout boundary. It adds no backend/API dependency.
License: EPL-2.0 OR GPL-3.0-or-later. See the
[ELK layered documentation](https://eclipse.dev/elk/reference/algorithms/org-eclipse-elk-layered.html)
and [elkjs implementation](https://github.com/kieler/elkjs).

Only projected records enter ELK. No cache-specific coordinate table, manual
node dragging, alphabetical layout or vertical fallback is used. A real worker
runs ELK; abort rejects obsolete work immediately, terminates its nested ELK
worker through an acknowledgement, and bounds unresponsive owner cleanup to one
second. A ten-second timeout exposes Retry/collapse instead of hiding records.

[geometry.json](evidence/architecture-presentation/geometry.json) records nine
local source scenarios and bounded generated coordinate samples. All pass:
exact route endpoints, no unrelated body/header intersections, no distinct
signals sharing a segment, no label collisions and horizontal serial-sibling
progression. Exhaustive mode checks **1045** serial sibling dependencies.

Examples from the generated nested linear layer (absolute graph coordinates):

| Scope | Stages (`x`, `y`, `width`, `height`) |
| --- | --- |
| Model | Embedding `(286,583,150,100)` → Layer 0 `(492,263,4674,811)` → compressed 1–23 `(5254,476,150,172)` → final norm `(5444,476,150,100)` |
| Layer 0 | Input norm `(516,724,150,100)` → attention `(722,364,2978,686)` → residual add `(3756,572,150,124)` → norm `(3954,572,150,100)` → derived MLP `(4160,492,776,332)` |
| MLP | Gate `(4184,572,150,100)` → SiLU `(4374,572,150,100)` → multiply `(4572,572,150,124)` → down `(4762,572,150,100)`; parallel up `(4374,700,150,100)` feeds multiply |

The JSON includes attention scope coordinates and exact source IDs. Dimensions
add measured label space without changing source connectivity. Tests repeat
geometric checks with dimensions, expansion/collapse and changed fixtures.
Browser resizing 1178→1440→1178 preserves node/port/route coordinates and layout
invocation count; a newly requested stack window uses the current panel width.

## Built browser reproduction and visual comparison

Environment: Linux x86-64, Intel i7-11700K (16 logical CPUs), 62.7 GiB RAM,
Node 24.14.0, Chromium 153.0.8010.12, DPR 1, height 900, widths 1178 and 1440.
Captures use the production build and deterministic replay of immutable real
local HTTP responses, including the complete original structural graph.
They are separate from the live-backend acceptance below.

42 navigation captures and eight additional dimension/attention captures were made for overview, window, linear/full layer, MLP, linear/full
states, ports, line interiors, dimensions, last layer, exhaustive context/global
fit/detail, resize and return. The principal views at both widths were opened
and visually inspected. The reference's organization is retained as focus states
of one canvas, with dimensions initially hidden and transient inspection.
Tokenizer/context nodes keep their actual source meaning without invented edges.

- Before: [1178](evidence/architecture-presentation/before-1178.png), [1440](evidence/architecture-presentation/before-1440.png).
- Compact ×24 and verified L–L–L–F pattern: [1178](evidence/architecture-presentation/overview-1178.png), [1440](evidence/architecture-presentation/overview-1440.png).
- Automatic serial window: [four instances at 1178](evidence/architecture-presentation/stack-1178.png), [five at 1440](evidence/architecture-presentation/stack-1440.png).
- Concrete [linear layer](evidence/architecture-presentation/linear-layer-1440.png) and [full layer](evidence/architecture-presentation/full-layer-1440.png): different actual consumed auxiliaries and bindings.
- [Linear states](evidence/architecture-presentation/linear-states-1178.png) and [separate K/V state routes](evidence/architecture-presentation/full-states-1440.png).
- Full MLP connection from [output-port hover](evidence/architecture-presentation/port-hover-1440.png) and [line-interior hover](evidence/architecture-presentation/line-hover-1178.png): warm amber route, arrowhead and exact endpoint labels/markers.
- Additional [readable dimensions](evidence/architecture-presentation/dimensions-readable-1178.png) and [nested attention detail](evidence/architecture-presentation/attention-detail-1440.png).
- [Readable operation reached in exhaustive mode](evidence/architecture-presentation/exhaustive-detail-1440.png), with all 948 source nodes retained.

All positioning comes from the worker. Captures use existing navigation, Fit and
Center selected controls. Global exhaustive Fit is a footprint of a very large
graph, not a readable all-labels image; selecting a source component restores a
readable detail view. Expansion itself never automatically fits the whole model.

[comparison.json](evidence/architecture-presentation/comparison.json) records
**zero changed pixels outside the rounded Architecture panel interior** across
all 50 graph captures, against baseline `0e8baebb`. The original panel hairline,
application bars and rounded outer corners remain in the comparison mask.
Viewport/document/body/scroll geometry is equal. Ten additional before/after
captures compare the initial Architecture shell and whole Tensor/Tokenizer views,
including switching through Architecture: **zero changed pixels**. Image SHA-256
values are retained; matching Tensor/Tokenizer images need not be duplicated.
The baseline Tokenizer prompt clears on returning from another explorer; the
populated and return states are compared separately. Native modal styling and
resource/focus behavior retain their existing baseline.

## Size, duration and memory observations

The original source has 948 nodes, 1329 edges, 333 parameters and 49 groups;
24 instances comprise 18 linear and six full-attention layers. Composed routes
retain every original segment, so visible connection counts differ from source
edge counts. [browser.json](evidence/architecture-presentation/browser.json)
records all states, bounds, layout invocation count and browser observations.

| 1440 px state | Visible nodes / connections | Original edges represented | Bounds | Worker layout | Main JS heap after GC |
| --- | ---: | ---: | --- | ---: | ---: |
| Initial compact | 13 / 8 | 40 | 1276 × 636 | 168.5 ms | 9.77 MiB |
| Selected linear layer | 20 / 17 | 52 | 2520 × 652 | 180.1 ms | 11.29 MiB |
| Exhaustive, retained camera | 948 / 1168 | 1329 | 114902 × 8661 | 1144.7 ms | 21.16 MiB |
| Exhaustive global Fit | 948 / 1168 | 1329 | same | same layout | 58.02 MiB |
| Exhaustive selected detail | 948 / 1168 | 1329 | same | 1200.1 ms | 26.50 MiB |
| Return to compact | 13 / 8 | 40 | 1276 × 636 | 128.1 ms | 23.06 MiB |

Collapsed geometry excludes hidden descendants; exhaustive restores all source
nodes and edge references. GC observations use CDP `HeapProfiler.collectGarbage`
then `Performance.getMetrics`, with DOM counters, after each action settles.
They are main-thread point observations, not process RSS, worker/GPU peaks, a
zero-retention proof or a universal performance guarantee. Other acceptance work
shared the host. Global Fit renders more DOM than a culled detail viewport.

## Validation and reproduction

- Repository UI gate: API binding freshness, TypeScript, ESLint, **595 unit tests** and production build pass.
- 44 architecture browser cases pass across desktop/narrow projects, plus a four-case evidence rerun. Native pointer hits test exact source/destination dots and independently targetable interiors; no forced events. Assertions cover fan-out, boundary chains, residuals, K/V, same-shaped ports, stop-at-computation, pin/clear, Tab/Escape/focus return and unchanged geometry/camera/layout count. Existing native streaming, disposal, unavailable representations and shell isolation pass.
- Production HTTP architecture acceptance: **7 passed**, requiring all four complete local references. Built-browser/live-backend architecture acceptance: **18 passed in 24.8 minutes**. All four synthetic checkpoints and all four complete local references passed at DPR 1 and 2, together with progressive cancellation. Actual references were SmolLM2 Base, Qwen3 GPTQ, Qwen3.5 NVFP4 and V-JEPA 2; no reference was skipped. Native vectors/matrices were compared with the local value oracle, and unavailable packed/higher-rank representations remained metadata.
- The [live acceptance receipt](evidence/architecture-presentation/live-acceptance.json) retains case names, projects, outcomes and durations without model/cache data. A fresh-cache development run also passed **12/12 browser cases with two workers** after preoptimizing the two worker-loaded ELK modules; no dependency-triggered page reload remained.

```sh
npm --prefix ui ci
npm --prefix ui run check
npm --prefix ui run test:browser -- architecture.spec.ts architecture-connections.spec.ts architecture-inspection.spec.ts architecture-shell.spec.ts
node ui/scripts/architecture-layout-evidence.mjs --graph /local/source-graph.json --out /local/geometry.json
LMEX_ARCHITECTURE_REFERENCES=/local/references.json LMEX_REQUIRE_ARCHITECTURE_REFERENCES=1 backend/.venv/bin/python -m pytest acceptance/test_architecture.py -ra
# From ui/, with the same reference environment:
xvfb-run -a npm run test:acceptance -- architecture.spec.ts
```

Reference provisioning/reproduction remains governed by [architecture.md](architecture.md).
The browser connection suite writes real screenshot/coordinate files to its
Playwright output directory. Those fixtures do not establish model support;
the original cache reproduction is also distinct from fresh checkpoint acceptance.
