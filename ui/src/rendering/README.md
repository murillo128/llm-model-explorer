# Exact tensor renderer

`index.ts` exports the independent WebGL2 renderer and a small native-scroll DOM
adapter. No React, HTTP, model metadata interpretation, or backend computation is
required. The logical descriptor accepts the `shape`, `rank`, `numel`, and
`logical_dtype` subset of the API tensor descriptor. Stream adapters convert
little-endian C-layout tensor bytes to consecutive Float32Array chunks upstream.

```ts
import { TensorViewport } from './rendering';

const viewport = new TensorViewport(emptyHost, {
  shape: [2, 3], rank: 2, numel: 6, logical_dtype: 'float32',
});
viewport.renderer.upload(new Float32Array([-1, -0.5, 0]));
viewport.refresh(); // remaining row is explicitly pending
viewport.renderer.upload(new Float32Array([0.25, 0.5, 1]));
viewport.renderer.setTransfer({
  statistics: { minimum: -1, maximum: 1, percentiles: { p01: -1, p99: 1 } },
});
viewport.refresh();
// Later: viewport.dispose(); before closing/replacing this tensor.
```

For custom composition, construct `TensorRenderer(canvas, descriptor, options)`.
The renderer owns that canvas's exclusive WebGL2 context. `setView(cssWidth,
cssHeight, scrollLeft, scrollTop, dpr)` returns integer data origins, framebuffer
size, CSS size and the full logical scroll extent. The canvas uses integer CSS
dimensions plus a DPR-only transform, avoiding Chromium's fractional canvas-size
rounding. Custom hosts must constrain/clip its untransformed layout box to the
returned CSS extent, as `TensorViewport` does. Call it on viewport/DPR changes,
then `draw()`. Drawing with stale DPR geometry fails explicitly. `cellAt()` accepts
coordinates relative to the snapped canvas rectangle; `readCell()` distinguishes
finite, nonfinite, pending, out-of-bounds, and received-but-not-retained states.

The default retains one CPU array. `retainValues: false` eliminates it; callers
then need their own exact-readout source if needed. Inputs are copied directly to
scalar storage and optional CPU storage, never retained as chunk lists. Uploads
must start at `populatedPrefix`; the optional offset argument detects discontinuity.
`setTransfer()` replaces the small transfer configuration, with unspecified fields
returning to defaults. No texture upload occurs until more data or reconstruction.

`TensorViewport` owns an initially empty host's children and temporary layout
styles; give the host a width/height and optionally max-width/max-height. It uses
native overflow, respects those CSS ceilings and hardware framebuffer limits,
aligns the canvas to device pixels, and refreshes on scroll/resize/DPR changes.
Call `refresh()` after uploading or changing transfer parameters. It displays
empty/error/context states; application composition owns controls and retry actions.
Constructor errors leave a visible explanation; clear that failed host before
constructing a replacement. The renderer constructor cleans partial GPU allocations.

`onStateChange` reports `ready`, `empty`, `lost`, `needs-reconstruction`, `failed`,
and `disposed`. After actual restoration, call `reconstruct()` and refresh. With
retained CPU values it reuploads the received prefix; otherwise restart the stream
from zero. Dispose removes listeners/resources and is idempotent. `diagnostics`
provides scalar/CPU bytes, live texture count, upload/allocation counters,
reconstruction generations and queried hardware limits. Optional lower
`textureLimit`/`framebufferLimit` ceilings support budgets and deterministic tests.

Run `/renderer-demo.html` through Vite dev or the production preview for a minimal
standalone progressive example. Browser tests use a source-only harness at port
4174 and exercise the production demo at port 4173. The harness and its GL call
instrumentation are excluded from production build inputs.

The normative geometry, scalar storage, transfer and lifetime rules are in
[rendering.md](../../../docs/spec/ui/rendering.md). Transport and React composition remain outside the renderer.

`MatrixViewport` composes the scalar `TensorViewport` with a right-hand and lower
`DistributionRenderer` for nonempty rank-2 descriptors. It uses a single native
matrix scroller and passes its exact view origins/extents to both profiles.
`TensorViewport.onViewChange` is the small composition hook; ordinary callers
remain unchanged. Rank-1 creates only the common scalar strip.

`DistributionRenderer(canvas, [rows, columns], axisLength, options)` uses the same
band allocation, indexed draw, prefix tracking and resource lifetime as the scalar
renderer, with R32UI / RED_INTEGER / UNSIGNED_INT storage and a Uint32Array CPU
copy. Its density shader converts counts only at draw time using the fixed
`log1p(count)/log1p(axisLength)` transfer. Pending counts remain distinguishable
from received zeros. No client-side histogram computation is performed.

`setSelection({ row, column })` sets small chroma uniforms; `null` clears selection.
A negative coordinate disables that axis, allowing linked row/column profiles.
`MatrixViewport` accepts `onInspection` to enable its rank-2 interaction controller.
The callback supplies exact coordinates, round-trip value text, viewport placement,
and a guarded `draw(canvas)` function for an application-owned 9×9 2D canvas.
React owns the floating card/readout; the renderer owns the shared-texture display
pass and its small framebuffer. `drawNeighborhood()` never changes the main view,
allocates scalar storage, or uploads weights. See the owning specifications for
linear-sRGB chroma, display encoding, missing-data and keyboard semantics.
