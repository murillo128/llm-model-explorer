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
