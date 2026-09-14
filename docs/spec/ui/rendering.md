# Tensor rendering

## Pixel mapping

The durable spatial invariant is **one authoritative logical scalar per matrix cell**.
A rank-2 `[rows, columns]` matrix uses columns on X and rows on Y, with row zero
at the top. Its camera applies one uniform scale `s >= 1` device pixels per cell
to both axes, so cells remain square. Enlarged cells use exact indexed/nearest
sampling; interpolation, aggregation and subpixel minification are forbidden.
Zoom never changes ordering, scalar storage or the logical identity of a complete
matrix. Rank-1 retains the horizontal `N × 1` device-pixel strip. Empty tensors
show an explicit empty state and allocate no scalar textures.

At devicePixelRatio `d`, the complete matrix extent is `W*s/d × H*s/d` CSS pixels.
Visible framebuffer dimensions and screen scroll offsets are device-pixel integers.
The logical camera origin is `round(nativeCSSScroll*d)/s`, which may be fractional.
Hit testing resolves the logical cell containing the device pixel on each axis
and excludes coordinates outside the visible surface. CPU hit testing and GPU
sampling use the same float32 rasterized cell edges (`ceil(cellOffset*s -
fractionalOriginPixels)`), with an integer logical origin kept separately.
This avoids division rounding choosing a neighboring cell at exact edges or
texture-band seams, including at large scroll origins. Native browsers may round requested CSS scroll offsets; focal
preservation is subject to that rounding and matrix bounds.

Snap the canvas screen position to device pixels. Integer canvas dimensions with
a DPR-only presentation transform and a constrained layout box avoid browser
compositing interpolation. Zoom is a shader camera transform, never CSS scaling
of a previously rasterized tensor. DPR changes recompute framebuffer/CSS extents
while preserving logical camera origins as far as native scroll bounds permit.

Oversized content uses normal scrollbars. Internal texture/band partitioning may
accommodate WebGL2 limits but must preserve exact scalar identity and sampling.

Query actual texture, renderbuffer and viewport limits. Use a bounded visible
framebuffer over the complete data extent rather than requiring a giant
canvas. Disjoint texture bands are needed only when a dimension exceeds the
effective texture ceiling (the hardware limit or an explicitly lower resource
ceiling). Band edges must neither omit nor duplicate cells. Report unsupported
WebGL2, allocation failure or an unrepresentable browser scroll extent explicitly;
never alter logical tensor values or ordering to fit a limit. This partitioning does not change
complete-tensor downloading or introduce API tiles/prefetch.

## Numeric representation

The renderer consumes the logical visualization representation supplied by the backend. The initial representation is one `float32` logical value per weight regardless of the model's physical storage representation.

The tensor value remains authoritative. The renderer must not rewrite tensor values in order to change color, brightness, selection state, zoom, or another visual property.

Store one authoritative scalar GPU representation using single-channel R32F
textures with exact indexed sampling (`texelFetch`, nearest filtering, one mip
level). Green scalar and semantic-color/inspection drawing reuse that storage.
No RGB/YUV weight encoding, interpolation, downsampling or quantization is allowed.
One optional CPU Float32Array may support exact readout and reconstruction. Do not
retain accumulated chunk lists or values belonging to disposed tensors.

Accept consecutive row-major float32 chunks and track their populated prefix.
Not-yet-received cells and received nonfinite values are distinct unavailable
states, visually separate from finite green data; neither represents a zero
weight. Rendering may start before the complete prefix or statistics arrive.

## Luminosity

Weight value is represented through luminosity rather than semantic color.

The renderer maps tensor values to a normalized `[0, 1]` visual intensity using a configurable nonlinear, sigmoid-like transfer function. The default transfer uses robust tensor statistics such as percentiles rather than relying only on raw minimum and maximum values, so a small number of extreme values does not collapse most weights around the same mid-level intensity.

Tensor statistics are supplied independently by the backend and may arrive after tensor data has already started rendering. The renderer may use a provisional mapping and update the visual transfer function when the statistics become available without retransmitting or rewriting tensor values.

The reproducible baseline uses `a=p01`, `b=p99`, `k=12`, and
`u=clamp((w-a)/(b-a), 0, 1)`. With `L(x)=1/(1+exp(-x))`, intensity is
`(L(k*(u-0.5))-L(-k/2))/(L(k/2)-L(-k/2))`. Normalize without overflowing
float32 subtraction, including opposite-sign finite extremes. If percentile
anchors coincide, use finite minimum/maximum when distinct; constant finite data
maps to 0.5. Before statistics arrive, use the provisional centered logistic
`L(k*w)`, clamping its exponent input for numeric stability. Changing slope or
anchors changes small draw uniforms, never scalar bytes. The renderer accepts
slopes from 0.01 through 80 to keep endpoint normalization well-conditioned.
The steeper default expands tonal separation around the robust range center.
The green display curve below keeps midrange data darker than an sRGB-encoded
neutral 0.5, while retaining ordered luminance across the range.

## Resource lifetime

Disposal releases scalar textures, shader/program/vertex resources, the optional
CPU array and the visible framebuffer. Context loss invalidates GPU storage and
stops drawing; context restoration alone does not mark old textures valid.
An explicit reconstruction/retry allocates new resources and reuploads the retained
prefix, or restarts the prefix at zero when no CPU copy exists. Ordinary drawing,
hover, scrolling, zooming and transfer changes do not take this reconstruction path.

## Color

The default scientific palette is monochromatic sequential green. Hue does not
encode a second variable: scalar intensity controls monotonically increasing
luminance from near-black green through green to pale green. Matrix values and
distribution density use the same family. Warm amber/orange is reserved for
interaction overlays; future semantic modes require their own accepted design.

Display overlays must preserve the underlying scalar transfer and stored values.
Thin inspection guides may change final composited luminance, as defined below.

## Interaction scope

Matrix Explorer fit-width defaults, focal-point wheel/pinch zoom and scroll navigation are owned by `tensor-explorer.md`. Each instance owns its camera; these transforms never affect other explorers.

### Concrete linear-sRGB scalar and guide transfer

For normalized scalar intensity `t`, the green linear-sRGB color is
`G(t) = (0.001 + 0.819*t^3, 0.006 + 0.994*t^1.5, 0.002 + 0.858*t^3)`.
All components and luminance increase monotonically, and green is the largest
component throughout. Luminance uses `Y = 0.2126 R + 0.7152 G + 0.0722 B`.
Distribution intensity uses its separately owned density normalization before
this same display curve. It changes density brightness only, never value-bin
coordinates. Distribution numeric rulers and neutral zero references are
display-only chrome owned by `tensor-explorer.md`; they do not update scalar or
count storage. Constant finite tensors use `t=0.5`; all-nonfinite and
pending regions retain their explicit status colors instead of this curve.

The active row and column each receive one device pixel of amber overlay,
centered on the active logical row/column at the current camera scale, including texture-band origins. The guide thickness stays one device pixel when cells enlarge.
Blend in linear sRGB: `C = (1-alpha)*G(t) + alpha*(1, 0.32, 0.015)`.
Default guide alpha is `0.65`; the intersection uses `0.9`. This makes guides
visible even at both scalar endpoints and keeps underlying variation visible.
The overlay changes final display luminance, never scalar intensity, authoritative
values/counts, or their storage. Selection and opacity are small uniforms.

Encode each resulting linear component once with the sRGB OETF:
`12.92*c` for `c <= 0.0031308`, otherwise `1.055*c^(1/2.4)-0.055`.
The canvas and inspection RGBA8 buffer store these display-encoded bytes; do not
apply a second gamma conversion. Pending/nonfinite colors remain explicit status
colors outside this scalar transfer. Pixel validation permits at most one 8-bit
code per channel relative to ideal display encoding; decoded luminance versus
prequantized color permits `0.0045` absolute error. CPU numeric validation before
display quantization uses floating-point tolerance, not that display allowance.

### Magnifier display overlays

The 9×9 inspection buffer uses the same scalar bands and green transfer with
selection overlays disabled; enlarging it with nearest sampling must not enlarge
the main-view guides into cell-wide bands. The composition adds a 1 CSS-pixel
warm-orange border around the center cell, plus one horizontal and one vertical
1 CSS-pixel guide through its center across the magnifier. The guides use lower
opacity (0.4) than the border and blend over the display without filling or
replacing neighboring cells. These overlays require no duplicate scalar texture.
Magnifier visibility and collision placement are owned by
[tensor-explorer.md](tensor-explorer.md#magnifier).
