# Shared MatrixViewport evidence

Baseline: `c71f1b81bedfb534a6147f5c119ab0cfbf2cfeee`. After: the implementation
in the commit containing this report. Captured and visually inspected on Linux
with headed Chromium `153.0.8010.12`, Playwright from `ui/package-lock.json`,
SwiftShader WebGL2, 900 CSS-pixel browser height, and an 800 CSS-pixel fixture
workspace. These are deterministic shared-component fixtures, not model captures.

| Case | Same input/settings for each pair | Before | After |
| --- | --- | --- | --- |
| 1178 CSS px, DPR 1 | `[6144,1024]`, manual scale 1, domain `[-1,1]` | [Before](before-1178.png) | [After](after-1178.png) |
| 1440 CSS px, DPR 1 | `[6144,1024]`, manual scale 2, domain `[-1,1]` | [Before](before-1440.png) | [After](after-1440.png) |
| 1178 CSS px, DPR 2 | `[4,128]`, Fit width, domain `±Math.fround(.124)` | [Before](before-1178-short-dpr2.png) | [After](after-1178-short-dpr2.png) |

All pairs use the same deterministic scalar formula, transfer settings and
histogram counts derived from those scalars. Removing native gutters deliberately
increases available client space. Thus the first case now underfills horizontally
by 12 CSS pixels, and Fit width in the short case uses the recovered width. The
1440 case shows both overlay controls. The short case shows the bottom track
remaining at the viewer edge while its four rows remain centered vertically.

## Captured geometry

Full small measurements are in [before-geometry.json](before-geometry.json) and
[after-geometry.json](after-geometry.json). Distances below are CSS pixels.

| Case | Native gutter X/Y before → after | Client W×H before → after | After matrix X/Y offset | After row/right and column/bottom track gaps |
| --- | --- | --- | --- | --- |
| 1178 | 15/15 → 0/0 | 1021×601 → 1036×616 | 6/0 | 10/10 |
| 1440 | 15/15 → 0/0 | 1283×601 → 1298×616 | 0/0 | 10/10 |
| Short DPR 2 | 15/0 → 0/0 | 1071×666 → 1086×666 | 0/316 | 10/10 |

In every after capture, row-data top equals matrix top, column-data left equals
matrix left, and each profile bin axis remains exactly 100 device pixels. The
row minimum/maximum caption top difference changes from −12 to 0. The short
case uses a shared `×10⁻¹` caption; exact endpoints remain in tooltips and ARIA.

## Reproduce

Prepare an isolated checkout of the baseline, install its locked UI dependencies,
and run its Vite server on 4318. Run this checkout's server on 4317. From this
checkout's `ui/`, run the same capture script against each server:

```sh
MATRIX_EVIDENCE_URL=http://127.0.0.1:4318 MATRIX_EVIDENCE_HEADED=1 \
  xvfb-run -a node scripts/matrix-viewport-evidence.mjs before /tmp/matrix-evidence
MATRIX_EVIDENCE_URL=http://127.0.0.1:4317 MATRIX_EVIDENCE_HEADED=1 \
  xvfb-run -a node scripts/matrix-viewport-evidence.mjs after /tmp/matrix-evidence
```

Headed checks here required execution outside the filesystem sandbox because its
X11 restriction prevented Chromium from connecting to Xvfb. These captures and
browser results establish SwiftShader behavior, not hardware GPU performance.
