# Compact explorer integration acceptance — 2026-09-14

Issue #45 validates the integrated compact shell, Tensor Explorer, and live
input embeddings against the real production UI and backend on separate origins.
The changes are confined to acceptance instrumentation/tests, reproduction docs,
and compact evidence. No production algorithms or accepted contracts change.

## Reproduce and inspect

Run `acceptance/check.sh` with the locked dependencies described in
[README.md](README.md). The default remains self-contained CPU fixtures and
SwiftShader WebGL2. Both browser suites use Xvfb; the product suite now exercises
native scrollbar gutters rather than relying on overlay scrollbars.

The compact screenshots are [Tensor inspection](evidence/improvements/tensor.png)
(1000×500), [narrow Tensor workspace](evidence/improvements/tensor-narrow.png)
(390×640), and [Tokenizer with exact embedding rows](evidence/improvements/tokenizer.png)
(1440×1000), all at DPR 1. The scientific surfaces retain native device pixels;
the short seven-token embedding is deliberately only seven device pixels tall.

## Final local validation

`acceptance/check.sh` completed successfully. Machine-readable results and
geometry/resource measurements are in [cpu.json](evidence/improvements/cpu.json).

| Gate | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Backend unit/integration | 540 | 0 | 16 CUDA-only |
| Real TCP acceptance | 16 | 0 | 1 CUDA-only |
| UI unit | 253 | 0 | 0 |
| Existing component browser | 172 | 0 | 0 |
| Production browser, DPR 1/2 | 22 | 0 | 0 |

Contract validation passed with 137 resolved references, 88 instance cases and
76 reproducible wire fixtures. Fixture and UI binding regeneration produced no
diff. Backend Ruff/format/mypy and UI TypeScript/ESLint/build all passed. The
backend emitted two upstream Starlette TestClient deprecation warnings.

The production-browser count includes both optional local SmolLM2-135M Base
smokes, and TCP includes its local reference smoke. A separate bounded reference
embedding smoke passed for IDs `[49151,0,128,1,128]`: shape `[5,576]`, 11,520
bytes, SHA-256 `238e4ae3d863f067f2d276cf8995b621cda8f1aaf9dfe48ef4a37bbe58fd3eb3`,
zero cache entries. Run that optional check with
`backend/.venv/bin/python backend/scripts/smoke_input_embeddings.py /local/SmolLM2-135M`.
No weights were downloaded or modified. Without a configured local checkpoint,
reference checks report SKIP; the initial self-contained browser run recorded
20 passed and 2 reference skips.

Environment: Python 3.12.14, Node 24.14.0, Playwright 1.63.0 / Chromium
153.0.8010.12, PyTorch 2.14.0+cpu, Linux x86_64. Both DPRs reported SwiftShader
through ANGLE, 8192 texture/renderbuffer limits and an 8192×8192 viewport limit.
The CPU environment exposes no CUDA runtime/device.

For `A😀A`, the real IDs were `[1,35,175,256,249,225,35]`. The first 576 floats
were rendered while delivery was held; all 4,032 result floats matched the oracle.
Only 16,128 scalar GPU bytes were allocated for that matrix. Source observations
show 4,032 requested elements plus the preceding 576-element empty-prompt BOS
lookup, at most 576 elements per block, and no full-tensor iterator calls.

## Programmatic evidence

- One global top bar and bottom status bar, accessible model refresh/session
  close and tensor info, relative leaf labels with inline shape/dtype, keyboard
  selection, and no persistent tensor completion text.
- Four real fixture matrices cover no overflow, vertical only, horizontal only,
  and both axes at 1000×700 and 390×640, each at DPR 1 and 2. Positive native
  gutter assertions prevent false confidence from headless overlay scrollbars.
  Inventory scrolling leaves the matrix stationary; all four matrix scroll
  corners preserve profile origins, dimensions, and alignment. Document/body
  dimensions remain exactly the viewport with zero window scroll.
- Matrix scalar samples have stable monotonic luminosity and greater than 0.7
  decoded luminance contrast. Matrix and profile guides visibly change to amber;
  exact hover values and all 81 magnifier pixels remain correct. Guide changes
  allocate/upload no scalar textures.
- The real tokenizer's ordered IDs for `A😀A` are sent unchanged to the embedding
  endpoint. The probe compares every uploaded float against an independent
  arithmetic oracle over the deterministic embedding table. Duplicate ID rows
  remain separate sequence positions, including the overlapping emoji byte IDs.
  Keyboard cell inspection and annotation hover/focus link the correct rows.
- A cancellation-aware backend consumer barrier stops before the second block.
  The browser displays the first populated row and its exact readout while the
  response remains incomplete. Source observers count only requested row elements
  and no full-tensor iterator calls; the browser owns only the requested matrix.
  The independent TCP test also verifies exact bytes, CORS, bounded row blocks,
  operation/session/socket cancellation, and no persistent embedding artifacts.
- Delayed real tokenization and embedding responses, including A→B→A, cannot
  alter the newest annotations/readout or allocate/upload replacement textures.
  An in-flight embedding is invalidated on model/session replacement. A model
  without embedding capability retains an editable, functioning tokenizer.
- Repeated tensor/explorer switching, prompt edits, cancellation, and session
  teardown return operation/consumer/reader/flight/task/temp-file counts to zero.
  Browser readers/textures return to zero on unmount; explicit GC finds no retained
  scalar owners. While mounted, an empty prompt legitimately owns one BOS row.

## Frozen prompt baseline

The four PNGs under `ui/tests/tokenizer-prompt-regression.spec.ts-snapshots/`
are retained unchanged from issue #44. Their pre-embedding source pin is
`8166d94bfae6b0f51da571dff614d900cf5f4375`; its tokenizer source is identical to
planning baseline `a65025c0004fca179b874efe8a2353e91ae31f57`. The existing regression
compares empty and annotated prompt pixels after integration at 1440×900 and
390×844. This issue additionally compares exact real-product prompt screenshot
bytes and editor geometry before/after embedding completion. The 260px editor,
selection replacement, caret placement, undo/redo, Unicode spans, and composition
start/end suppression are exercised with real tokenizer/backend responses.
Composition events test the browser editor lifecycle, not an operating-system IME.

## Limits

Software WebGL2 is browser correctness evidence, not hardware performance.
Counters observe explicit application resources and iterator boundaries, not
browser/driver copies, total process memory, or VRAM. Test-only framebuffer
readbacks and barriers are excluded from production. Timings demonstrate ordering
around a deliberate barrier, not latency targets. Optional capability skips remain
separate from passes. The historical PoC/CUDA evidence remains in
[evidence.md](evidence.md).
