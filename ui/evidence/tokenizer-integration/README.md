# Tokenizer cards and scientific integration

The captures use the production Vite bundle, Chromium/SwiftShader and a real
CPU backend with generated local Hugging Face fixtures. They are reproducible
from `ui/acceptance/product.spec.ts`, especially `integrated two-card embeddings
have real scientific parity with the same checkpoint matrix`. No histogram
response, scalar upload, bin domain or transfer is substituted in that case.

## Observed result

- Prompt / Tokens and Input Embeddings have separate full-width title bars and
  bordered bodies; their workspace has no enclosing third card. The empty prompt
  captures hold an actual tokenization response while the UI reports waiting.
- The real byte tokenizer produces ordered IDs `[1,35,175,256,249,225,35]` for
  `A😀A`. Its two positions with ID 35 remain distinct rows. The derived matrix is
  `[7,576]`, with 4,032 logical float32 values and a full-range bin domain `[-1,1]`.
- Native GPU uploads match independently enumerated fixture values and every
  row/column histogram count. Each count section sums to 4,032. A fixture checkpoint
  tensor containing these same rows produces identical scalar/count uploads,
  domains, native dimensions and transfer uniforms through Tensor Explorer.
- Luminosity uses percentile anchors `[-0.984375,0.984375]`, separately from
  full-range histogram endpoints. At native scale the scientific canvases are
  576×7 (matrix), 100×7 (rows), and 576×100 (columns) device pixels.
- Short data remains a thin strip with aligned blank margins; row and column
  tracks remain at the viewer edges. Manual zoom and both-axis scrolling keep
  square cells and leave prompt geometry unchanged. Title/popover/splitter changes
  preserve subscriptions and uploads.

## Captures

`two-cards-empty-{1178,1440}.png` and
`two-cards-populated-{1178,1440}.png` show the independent cards.
`two-cards-resize-focus.png` shows the keyboard resize target and adjacent card
boundary feedback. `two-cards-zoom-scroll.png` shows the complete viewer with
both real histograms and overlay scrollbars. `integrated-tensor-{1178,1440}.png`
shows the inherited title, visible Inventory restore tooltip and aligned rulers.
These images were visually inspected alongside exact geometry assertions.
`parity-dpr1.json` and `parity-dpr2.json` record the compact numerical/geometry/
resource observations at both device pixel ratios.

## Validation

- API binding drift check, TypeScript, ESLint and production build passed.
- All 804 UI unit tests passed, including immediate exact uploads, coalesced
  presentation, final draw before replacement promotion, and stale-frame disposal.
- Focused desktop/narrow browser coverage includes tokenizer editing,
  replacement and disposal, automatic/manual panel layout, shared matrix and
  overlay scrollbars, Tensor title/inventory, and Architecture weight inspection.
- Python acceptance lint and formatting passed.
- TCP acceptance passed: 24 tests, with eight explicit skips for unavailable
  CUDA or unconfigured local reference/quantized checkpoints.
- Final production UI/live-backend DPR 1 focused checks passed: exact parity,
  independent auxiliary failure/cancellation, and both long-prompt widths.
- Focused production DPR 2: all 18 cases passed, including exact scientific
  parity, auxiliary failure/cancellation, duplicate rows, stale generations,
  editor history/IME, camera/overflow behavior, both long-prompt widths,
  all four Architecture fixture families, and progressive modal closure.
- Complete desktop/native-scrollbar and production DPR 1 gates are recorded in
  the PR's UI and Application acceptance checks. Five production local-reference
  cases require configured checkpoint directories and are explicitly skipped
  when those are absent; fixture success does not establish reference acceptance.

The shared consumer uploads every scalar/count chunk immediately, presents the
first values progressively, and coalesces subsequent redraws per animation frame.
It retains no extra scalar payload. The controller flushes pending presentation
before promoting a completed replacement and publishes only status transitions.
Native browser checks verify exact uploads/inspection, one queued refresh, and
resource cleanup at DPR 1/2 and desktop/narrow widths. The four 630-row layout
cases passed in approximately 13–21 seconds per case, including resize and camera
interactions. They retain the inherited 45-second completion wait and 90-second
case budget with every value/analysis and geometry assertion. These observations
are correctness evidence, not a rendering throughput benchmark.

## Reproduction

Use the repository's Node/Python environments and installed Playwright Chromium.
Set `PYTHONPATH` to this checkout's `backend/src` and repository root if reusing
Python dependencies from another checkout. Run headed browser suites serially,
with unused ports and a separate Xvfb display:

```sh
cd ui
npm run check
UI_TEST_PORT=23190 xvfb-run -a npm run test:acceptance -- \
  --grep 'integrated two-card|real embedding distribution cancellation|real embedding statistics failure'
```

The acceptance fixture adds only a small checkpoint tensor made from the exact
ordered rows used by the parity case. The test-only native WebGL probe observes
uint32 uploads and transfer uniforms; production rendering has no inspection
backdoor or alternate computation.
