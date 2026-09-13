# Exact tensor renderer validation

Captured 2026-09-13 with Node 24.14.0, npm 11.9.0, Playwright 1.63.0,
Chromium 153.0.8010.12 on Linux. WebGL reports `WebGL 2.0 (OpenGL ES 3.0
Chromium)` through ANGLE/Vulkan SwiftShader (Subzero). This is real browser
WebGL2 shader/texture/framebuffer execution using a software GPU; discrete GPU,
Firefox and Safari behavior have not been measured.

Reproduce from `ui/`:

```sh
npm ci
npm run check
npx playwright install chromium
npm run test:browser
```

The local run used the already-installed Chromium through
`PLAYWRIGHT_BROWSERS_PATH=/tmp/issue-3-playwright`. Typecheck, lint, production
build and 32 unit/component tests passed. All 44 browser cases passed across the
desktop and narrow projects (30 renderer cases plus 14 existing shell cases).
The renderer fixture uses port 4174; shell and standalone demo checks use the
production preview on 4173. No backend or downloaded model is required.

## Pixel and geometry observations

For shape `[2,3]`, row-major `[-1,-0.5,0,0.25,0.5,1]`, anchors `[-1,1]` and
default slope 8, `readPixels` returned neutral grayscale bytes:

```text
  0   27  128
189  228  255
```

At DPR 1 the canvas framebuffer and CSS extent were both `3 × 2`. At DPR 2 the
same framebuffer occupied `1.5 × 1` CSS pixels. Both retained the same matrix
orientation and exact cell hit/readout. Tiny deterministic screenshot fixtures
also assert the actual composited device pixels with zero pixel/color tolerance.
These caught Chromium rounding a fractional CSS canvas width before rasterizing;
integer dimensions plus a DPR-only presentation transform preserve the exact
3×2 output. A four-value prefix left the remaining
two pixels pending `[46,61,76,255]`, distinguishable from zero `[128,128,128,255]`.
Received NaN and either infinity produced `[178,51,140,255]`. The shared API
`tensor-special-floats` fixture also preserved float32 values and signed zero.

Rank-1 input rendered a horizontal strip. Shapes `[0]`, `[0,4]`, `[4,0]` displayed
“Empty tensor” and made zero scalar allocations. Forced 4-pixel texture/framebuffer
ceilings exercised scrolling across both band axes, first/last cells, fractional
host positioning and DPR 1/2. Separate chunks ending at elements 2, 11, 13, 24 and
35 of a `[5,7]` tensor verified every populated and pending pixel across band edges.
Native Chromium scrolling rounds requested CSS offsets; data origins derive from
the actual offsets, and canvas screen positions are snapped to device pixels.

## Limits and resource observations

| Queried capability | Actual value |
| --- | ---: |
| MAX_TEXTURE_SIZE | 8192 |
| MAX_RENDERBUFFER_SIZE | 8192 |
| MAX_VIEWPORT_DIMS | 8192 × 8192 |

Generated shapes `[2,8195]` and `[8195,2]` each used two R32F textures totaling
65,560 scalar bytes, two strided uploads, and a bounded 3×2 or 2×3 visible
framebuffer. Last-band pixels matched their logical values on both axes.
The artificially partitioned `[7,9]` case used six disjoint textures totaling
exactly `63 * 4 = 252` bytes, with one optional CPU array of the same size.

GL prototype instrumentation counted actual scalar allocations/uploads/deletes.
Late p01/p99 statistics, fallback min/max, constant values, opposite-sign float32
extremes, changed sigmoid slope, ordinary redraw and DPR changes caused no scalar
reallocation/reupload. Twelve create/upload/draw/dispose cycles and a partial
allocation failure deleted all 74 created textures; no live texture remained.

`WEBGL_lose_context` invalidated every live scalar texture and rejected drawing.
Restoration left the renderer awaiting explicit reconstruction. Retained mode
reuploaded only the four received values; nonretained mode reset the prefix to
zero. Both modes disposed to zero scalar/CPU bytes. An injected real GL allocation
error cleaned partial storage; unavailable WebGL2 produced a visible explanation.

The production non-React demo rendered after 98,304 of 786,432 values, accepted
another chunk and late statistics, and stayed within the narrow viewport.
Screenshots, exact pixel/resource JSON attachments and the HTML report are generated
under ignored `test-results/` and `playwright-report/`; CI retains these as the
`ui-browser-evidence` artifact. Large test tensors are generated at runtime.
