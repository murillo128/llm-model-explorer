# Matrix Explorer camera evidence

The camera changes display geometry only. Scalar R32F and profile R32UI storage,
progressive prefixes and the inspection framebuffer retain their existing owners.
The default is fit width with a native 1:1 floor; Fit width resets local origins.

Reproduce with the locked Node/Chromium dependencies:

```sh
npm run check --prefix ui
UI_TEST_PORT=4200 xvfb-run -a npm run test:browser --prefix ui
```

Use a runtime that permits Chromium WebGL2 under Xvfb for the native-scrollbar
project. No models or backend service are needed for component browser evidence.
The complete application gate remains `acceptance/check.sh`.

- `geometry.test.ts` exercises fit/minimum scale, focal coordinates, bounds and
  independent profile-axis scales at DPR 1, 1.25, 2 and 3.
- `matrix-zoom.spec.ts` covers short/wide `[4,128]`, tall/narrow `[900,25]`, native
  overflow `[576,1536]` and ordinary `[120,100]` matrices at DPR 1, 1.25 and 2.
  Browser measurements verify fixed profile thickness, shared origins/scales,
  anchored panels, native scroll bounds, wheel/ctrl-wheel, touch pinch, reset,
  pending/click values, DPR changes and two simultaneous independent cameras.
  Repeated zoom/DPR changes do not upload or allocate scalar storage; disposal
  leaves zero live GPU resources and retained CPU bytes.
- `matrix-zoom-pixels.spec.ts` reads actual WebGL profile pixels at fractional
  scale and compares every sampled scanline's logical index with the matrix and
  its authoritative uint32 density. It runs at DPR 1/2 in wide/narrow browsers.
  Selection probes at scales 1.25, 1.99, 3.25 and 4.9 and scroll offsets 0, 1 and
  7 verify one-pixel guides stay in the inspected matrix cell and align with
  both distribution panels without scalar/count allocations or uploads.
- `renderer.spec.ts` checks every framebuffer pixel at scales 1, 2, 3.25, 4.9 and
  23.7 across 8-cell texture bands against exact scalar/hit-test values. The 9×9
  magnifier output remains identical across scales. JSON geometry is attached to
  browser results. Existing transfer/value/magnifier oracles remain intact;
  `native-camera.ts` explicitly selects native scale where those oracles depend
  on one device pixel per cell.
  A separate full-frame selection oracle covers scales 1, 1.01, 1.25, 1.5,
  1.99, 3.25 and 4.9 across texture bands and fractional logical origins. It
  checks exact cell membership, guide thickness and stronger intersection color.

The fractional-scale oracle exposed a division-rounding seam at a texture-band
boundary. Sampling, hit testing and band scissors now share float32 rasterized
cell edges, keeping the integer logical origin separate. This avoids both the
seam and loss of integer identity from converting a large scroll origin to a
float uniform. Zoom does not create a recolored or resized copy of tensor data.

Selection guides use the midpoint of those same rasterized pixel intervals.
Rounding a logical cell center could previously place a guide in its neighbor
near native scale. The new matrix/profile and scalar selection probes fail with
that previous shader and pass with the raster-interval midpoint correction.
