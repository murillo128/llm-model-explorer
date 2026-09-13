# Matrix inspection evidence

The source tests are the reproducible evidence; browser JSON attachments contain
matrix/profile RGBA pixels, all 81 magnifier pixels, exact value text, viewport
geometry and display allocation totals. Screenshots supplement those assertions.

- `npm run check`: API generation consistency, TypeScript, ESLint, 181 unit tests
  and production build pass. Numeric tests sweep 1,001 luminances at four chroma
  strengths, assert in-gamut RGB and unchanged prequantized Y, and round-trip
  float32 signed zero, subnormal/extreme finite and nonfinite values.
- `tests/renderer.spec.ts`: 32 desktop/narrow checks pass, including a full-range
  anchored luminance sweep with saturated selection, unchanged black/white,
  banded scalar resources, DPR geometry, prefix uploads and context lifecycle.
- `tests/matrix-inspection.spec.ts`: 16 desktop/narrow checks pass at DPR 1/2.
  First/last/interior cells and both sides of texture-band boundaries report the
  stored float32 value. Every main/profile pixel is checked against the selected
  logical row/column and linear luminance. All 81 neighborhood positions match
  original matrix pixels or explicit unavailable edge cells. Actual composited
  enlargement samples verify nearest-neighbor output without interpolation.
- Repeated hover leaves texture-creation count, scalar/integer allocations,
  Float32Array constructor count, all texture uploads, retained CPU sizes, fetch
  count, view geometry and native scroll offsets unchanged. Exactly one 9×9
  display renderbuffer is reused and released on disposal/context loss.
- Additional interaction cases cover native rounded scroll offsets, focus/arrows/
  Escape, all viewport corners, pending cells becoming available, signed zero,
  nonfinite/precise decimals, late transfer changes, leave and tensor replacement.
- The broader browser regression run passed the other 102 cases, including four
  headed native-scrollbar checks. Its four initial corner-fixture failures were
  corrected by fixing the test host's explicit width; all 16 inspection checks
  subsequently passed. CI runs the complete suite on the published head.

Local validation used Node 24.14.0, exact locked dependencies, and installed
Chromium revision 1200 via a temporary Playwright config. The disk was full, so
isolated dependencies, test scratch space and bulky browser artifacts lived in
shared memory outside Git. CI uses repository-native Chromium installation and
retains the full `ui-browser-evidence` artifact. No dependency/config workaround
is committed.

The magnifier reads 324 rendered RGBA bytes per update from the renderer-owned
inspection framebuffer; this synchronous, bounded display readback is a deliberate
tradeoff to reuse the original scalar textures without a second WebGL context or
weight upload. Tests do not claim real-model/backend integration or a performance
benchmark. See the owning UI specifications for coefficients, gamut rule and the
0.0045 maximum decoded-luminance display quantization allowance.
