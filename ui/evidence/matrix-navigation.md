# Matrix navigation and linked inspection

The reusable Matrix Explorer centers underfilled axes on device pixels and keeps
the row/column profiles attached to the visible matrix. Magnifier placement
follows the current pointer or focused cell, flips near boundaries, and excludes
distribution data surfaces. Transient ranges share exact logical bounds across
the matrix and profiles. Escape/right-click restore bounded local camera history;
drag cancellation and focused popover/modal dismissal retain priority.

## Built application captures

- [Strongly underfilled matrix](matrix-navigation-underfilled.png): the production
  application and local acceptance backend show a complete 32×32 tensor at native
  scale. At DPR 1 its matrix is 32×32 CSS pixels inside an 801×586 client viewport,
  centered at `(709, 428)` with 10 CSS-pixel profile gaps and 100-device-pixel bins.
- [Zoomed linked preview](matrix-navigation-zoomed.png): selecting columns `[4,24)`
  and rows `[8,24)` produces a uniform scale of 35.6875 device pixels per cell.
  A second transient rectangle projects into both distributions. The next Escape
  cancels that preview; another restores the native camera.

Both scenarios pass at DPR 1 and 2. Across navigation, the acceptance probe retains
3 textures, 3 uploads and 29,696 GPU bytes. Captures use deterministic fixtures;
they do not establish local reference-model acceptance.

## Validation and reproduction

Geometry/unit and browser checks cover forward/reverse matrix and one-axis drags,
one-cell indicators, preview cancellation, per-axis centering, attached profiles,
native overflow, focal preservation, source replacement, resize and DPR changes.
Magnifier checks cover pointer movement through centers/edges/corners, stationary
pointer scrolling/zooming, layout translation and compact fallback. Camera checks
cover region/range history, coalesced wheel/trackpad/touch pinch, fit reset, local
shortcut scope and dismissal priority. Existing exact-pixel, hit-test and resource
checks remain in force.

From `ui/` with Node 24.14 and locked dependencies:

```sh
npm run check
UI_TEST_PORT=18430 xvfb-run -a npm run test:browser -- \
  --project=desktop --project=native-scrollbars
CAPTURE_MATRIX_NAVIGATION_EVIDENCE=1 UI_TEST_PORT=18530 \
  xvfb-run -a npm run test:acceptance -- --grep 'production matrix navigation'
```

The production test requires the repository backend environment. Xvfb/WebGL
checks used host execution because sandboxed Xvfb could not allocate rendering
resources. Use separate output directories for concurrent browser runs.
