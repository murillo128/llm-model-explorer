# Local canvas evidence

Measured 2026-09-14 on Linux 6.8.0 x86-64, Intel Core i7-11700K (16 logical
CPUs), Node 24.14.0, Playwright 1.63.0 / Chromium 153.0.8010.12, one browser
worker. The desktop viewport was 1440 × 900; the tests also exercise 390 × 844
and the shell at 280 × 400. Chromium uses the repository's SwiftShader flags.

These are configuration-count-derived **synthetic UI stress fixtures**, not
checkpoint graphs. Their generation, pinned model/configuration revisions and
limitations are documented in [README.md](README.md) and
`ui/tests/architecture-fixtures.ts`. Actual checkpoint/analyzer acceptance is not
claimed by this evidence.

| Fixture | Instances | Nodes | Edges | Worker packing (ms) | Expand-all observation (ms) | JS heap used (decimal MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 configuration | 28 | 928 | 1,069 | 2.7 | 532 | 97.97 |
| Qwen3.5 configuration | 24 | 940 | 1,079 | 1.3 | 475 | 98.50 |
| V-JEPA 2 configuration | 24 + 12 | 1,193 | 1,375 | 1.8 | 589 | 66.46 |
| SmolLM2 configuration | 30 | 994 | 1,145 | 1.5 | 566 | 105.62 |

Worker timing excludes clone/transport and DOM work. Expand-all observation is
wall-clock time from clicking to the complete expanded layout becoming observable
in the component. Heap is Chromium `Runtime.getHeapUsage.usedSize` after centering
on the final instance; it includes the harness/application and is not peak memory.
The test attachment also records total heap, embedder heap, backing storage,
viewport, fixture revision and DOM count. Values are observations, not guarantees.

At that centered viewport, six nodes were mounted while all semantic records and
the complete expanded layout remained retained. Tests reveal the last operation,
collapse every group, and recover the exact selected instance with Center selected.
The independent contract fixture verifies boundary edge identity, grouping,
dimensions, safe text, keyboard expansion and inspection callback identity.
Worker tests check timeout and abort termination; browser tests exercise repeated
creation/disposal, and component tests reject obsolete session/model responses.

Reproduction commands are in [README.md](README.md). Browser runs attach the raw
compact JSON measurements and expanded-graph screenshots to the Playwright report;
bulky traces, reports, model files and generated graph copies are not committed.

## Regression-helper correction

The first CI candidate passed 327 browser cases but timed out in the existing
1440 × 900, DPR-2 native-scrollbar case. Its native-camera setup dispatched 20
zoom-out events per tensor even after reaching the native lower bound. Each event
still invoked synchronous scientific redraws. The helper now stops when the
display extent no longer shrinks, retaining the same bounded maximum and every
downstream pixel/geometry assertion. No production camera behavior was changed.
All ten native-scrollbar cases then passed locally in 34.5 seconds; the previously
timed-out case completed in 7.5 seconds. Typecheck and lint also passed.

## Issue #242: repeated GLM expert layout

Measured 2026-09-25 on the #178 local reference host with Node 24.14.0, Vitest
5.0.0 and elkjs 0.12.0. The prepared graph came from the pinned
`cyankiwi/GLM-4.7-Flash-AWQ-4bit` revision
`25624b53414e585bcf7dcb9584667c3106c6089b`; no model files are in Git.
The 28 MB compact response has 2,486 node records, 19,082 edges and 46 compact
expert families. Exact materialization creates 5,430 nodes; full projection has
5,430 visible nodes and 17,956 routes, representing all 19,082 source edges.

The same local graph and exhaustive projection options were profiled before and
after the change. Stage timestamps were taken around projection, ELK input
construction, ELK calls and output conversion. These are Node geometry timings,
excluding browser worker transport and paint. `/usr/bin/time -v` measured the
peak resident memory of the benchmark process, including Vitest.

| Measurement | Before | After |
| --- | ---: | ---: |
| Projection | 92 ms | 99 ms |
| ELK input ready, cumulative | 140 ms | 240 ms |
| Repeated interiors ready, cumulative | n/a | 996 ms |
| Outer ELK ready, cumulative | 18,329 ms | 5,598 ms |
| Routes converted, cumulative | 18,347 ms | 5,646 ms |
| Complete geometry | 18,350 ms | 5,650 ms |
| Peak benchmark RSS | 1,718,520 KiB | 850,668 KiB |

Both runs produced 5,430 boxes, 20,491 real ports and 17,956 routes. The
optimized run retained all 19,082 represented source edge IDs and every route
met its exact source and target port (zero endpoint failures). The deterministic
46-layer/64-expert UI fixture exercises the same repeated fan-out without a
checkpoint in CI; one final expert has a deliberately different port and edge
to verify that geometry reuse follows structure rather than names. The fixture
has more than 18,000 explicit source edges, matching the reference fan-out
scale.

The production browser check used the pinned local GLM files from #178 (a
hard-linked single-model root), this branch's backend, a production Vite build,
headless Chromium 153 with SwiftShader, and a 1178 × 900 viewport at DPR 1.
Playwright forwarded the backend's actual HTTP responses through Node because
the sandboxed Chromium process could not connect to the local backend port;
responses and graph records were not mocked. The first successful browser run
completed the full expansion with a 4,523.5 ms worker layout and 11,590 ms
from click to visible operations, without a layout error. It retained 5,430
logical nodes, 17,956 visible routes and all 19,082 represented source edge
IDs. Eight nodes were mounted in the viewport, including five operation cards.
The final concrete expert was selected from the source browser, centered, and
inspected successfully.

After indexing ports and routes for React rendering, the final candidate's
browser run completed worker layout in 4,605.6 ms and showed operation cards
9,821 ms after the Expand all click. It again retained the same node, route and
source-edge counts, with five operation cards mounted. CDP sampled a peak
main-renderer JS heap of 511,756,404 bytes in that run; it did not provide a
separate peak for the layout worker. The screenshot and raw browser observation
remain outside Git under `/tmp/lmex-issue-242-*`.
