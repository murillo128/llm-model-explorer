# Adaptive Matrix Explorer magnifier

Reproduce with Node 24.14.0 and the locked UI dependencies:

```sh
npm run check --prefix ui
UI_TEST_PORT=22630 npm run test:browser --prefix ui -- matrix-inspection.spec.ts magnifier-adaptation.spec.ts matrix-explorer.spec.ts matrix-zoom.spec.ts matrix-zoom-pixels.spec.ts --project=desktop --project=narrow
```

`inspection-layout.test.ts` checks four corners, right/bottom panel adjacency,
source-neighborhood avoidance, empty scientific space, constrained-pane fallback,
and exact 8/10 CSS-pixel hysteresis boundaries. `MatrixExplorer.test.tsx` verifies
that suppressing/restoring the card retains the readout and logical callbacks.

The browser tests cover camera-driven transitions at DPR 1, 1.25 and 2, independent
instance histories, exact pending/nonfinite/float32 readouts, linked profiles,
texture-band boundaries, context loss and disposal. DOM geometry asserts the
center border and full centered 1 CSS-pixel guides; screenshots sample all 81
nearest-enlarged cells away from those overlays against untinted scalar output.
Resource counters detect scalar allocations, uploads or requests during inspection
and magnifier suppression/restoration. Existing camera and profile-pixel tests
retain the exact hit-test and aligned-navigation oracles.

Browser screenshots and pixel attachments remain generated test artifacts outside
Git. The neighborhood still reads only 324 display bytes from the existing scalar
bands; the warm-orange border/guides are separate DOM overlays. The full card may
fall back to its compact readout if protected geometry leaves insufficient space.
This evidence covers synthetic UI fixtures; the PR's application-acceptance check
owns integrated backend/product validation.
