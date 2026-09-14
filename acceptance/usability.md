# Navigable matrix and stable tokenizer acceptance

Issue #67 validates the integrated usability children #59–#66 on epic #68's
integration branch. Changes add deterministic acceptance assets/checks and fix
one inspection redraw race during DPR changes. API/specification contracts are
unchanged.

## Reproduction

Run `acceptance/check.sh` using the locked dependencies and optional reference
configuration in [README.md](README.md). Production browser tests run the built
Vite bundle against a separate real backend origin at DPR 1 and 2. The injected
probe observes native uploads, allocations, readers and display pixels; it is not
part of the shipped bundle. Camera helpers observe DOM extents/origins and drive
normal wheel/pointer events without accessing renderer internals.

Focused additions:

```sh
backend/.venv/bin/python -m pytest acceptance/test_distribution_scales.py
(cd ui && npm run build && xvfb-run -a npm run test:acceptance -- \
  --grep 'integrated inventory|integrated camera|production distribution|production stale')
```

## Integrated coverage

| Contract | Production evidence | Complementary full-stack evidence |
| --- | --- | --- |
| Inventory/header | Fresh deep branches closed; resize and hidden-state reload; complete width reclamation; unchanged canvas/upload owners; fixed 40px header and matrix geometry across held streaming/completion; anchored ephemeral/pinned metadata and exact fields | Component keyboard/touch, clamp, malformed storage, cancellation/error geometry |
| Camera/selection | Underfilled fit width, native floor, wheel focal coordinates, real touch pinch, square cells, exact amber matrix/row/column bounds, selected-range fit, orthogonal center, exact scalar click readout | Fractional DPR/raster-edge/band pixel oracles, reversed/degenerate/cancelled drag, live DPR change, two independent instances |
| Distributions | Shared matrix/profile origins and axis extents; fixed 100-device-pixel bin thickness; stable domain labels through camera/inspection; five numerical shapes | Independent TCP float32 values, full finite endpoints, exact uint32 counts; component density/guide pixel oracle |
| Magnifier | 7→9→10.1→9→7 CSS-pixel hysteresis, retained exact numeric readout, pane containment/distribution exclusion, thin centered guides | Four-corner geometry, exact threshold boundaries and full 9×9 scalar pixels |
| Tokenizer | Embedding zoom/rectangle/scroll/reset preserves prompt rectangle, scroll and caret; delayed real tokenizer response retains mapped brackets/IDs and old canvas; animation-frame sampling detects blank annotations/matrix; matching result promotion and disabled/restored generation linkage | Real duplicate IDs/inserted BOS row uploads, A→B→A response reordering, Unicode/IME/history, unsupported embeddings and repeated resource cleanup |
| Lifecycle | Existing progressive DATA/publication barriers, cancellation, failures, source replacements, reader/GPU/CPU cleanup, fixed document geometry | Explicit WebGL2 unsupported/context-loss/reconstruction checks and scalar ownership tests |

The streaming header check uses the existing 32×32 fixture so chrome validation
does not depend on large-tensor software-rendering throughput. Gesture helpers
wait for two layout frames before beginning a subsequent drag. The local reference
smoke explicitly selects native scale for its existing scalar oracle; dedicated
camera scenarios separately validate fit-width and enlarged cells.

The extended fixture adds four 20×20 float32 matrices: near-zero concentration,
398 zeros between −1000/+3000 outliers, constant 2, and all-nonfinite values. The
existing 3×4 scientific tensor supplies the asymmetric −2/+6 domain with zero at
25%. TCP counts use an independent NumPy/float64 oracle. The outlier check proves
p01/p99 both equal zero while the displayed full-range domain remains −1000/+3000.
Generated tensors, model assets, cache files, raw logs and traces stay outside Git.

## Specification consistency

The integrated `AGENTS.md` scalar invariant and `docs/spec/ui/rendering.md` agree:
one authoritative logical scalar per square cell, native 1:1 minimum, indexed
sampling, and a local uniform camera. `tensor-explorer.md` owns fit/wheel/pinch,
selection and magnifier behavior. `tokenizer-explorer.md` explicitly retains mapped
stale annotations and complete previous embeddings, with generation-safe linkage.
UI architecture and visual-language documents refer to these owning rules. No
contradictory exact-one-pixel/no-zoom or hide-on-pending instruction remains in
these normative sources. Historical evidence reports describe their original
validation snapshots and do not override the updated specification.

## DPR event-order defect

The full browser run exposed an unhandled exception when inspection cleared
between a browser DPR change and viewport resize notification. A deterministic
production-browser regression forces that event order and reproduced
`Device pixel ratio changed. Call setView before drawing again.` before the fix.
Inspection now updates selection state but skips redraws whose viewport DPR is
stale. The ordinary resize path owns the next valid draw. The regression checks
cleared readout, recovered exact keyboard inspection, and unchanged texture/upload
counts at DPR 1 and 2. This is a bounded local defect correction permitted by #67.

## Evidence limits

Software WebGL2 establishes browser correctness, not physical GPU performance.
The default locked Linux PyTorch environment is CPU-only; CUDA results remain
explicit capability skips. Local SmolLM2-135M Base checks use existing operator
assets read-only, without download or cache/model publication. Exact candidate
heads, complete gate counts and CI links are retained in the delivery PR.
