# Tensor rendering

## Pixel mapping

The fundamental proof-of-concept rule is exact spatial mapping: **one tensor weight equals one rendered pixel**.

A 2D tensor of width `W` and height `H` therefore occupies `W × H` rendered pixels. The renderer must not fit the matrix to the viewport, resample it, aggregate weights, or use zoom as a substitute for the one-to-one mapping.

Here a rendered pixel means one framebuffer/device pixel, not one CSS pixel. At
devicePixelRatio `d`, the complete data extent is `W/d × H/d` CSS pixels. Rank-2
`[rows, columns]` uses columns on X and rows on Y, with row zero at the top. The
rank-1 baseline is a horizontal `N × 1` device-pixel strip. Empty tensors show an
explicit empty state and allocate no scalar textures.

Visible framebuffer dimensions and data scroll origins are integers. Convert the
browser's actual native scroll offsets to data origins with `round(offset * d)`;
hit testing uses `origin + floor(localCSSCoordinate * d)` and excludes coordinates
outside the visible surface. Native browsers may round requested scroll offsets
before this conversion. Snap the canvas's screen position to device pixels too.
Account for browser compositing: fractional canvas CSS dimensions must not cause
rounding/interpolation. Integer canvas dimensions with a DPR-only presentation
transform and a constrained layout box are one valid implementation.
A DPR change explicitly recomputes canvas dimensions, CSS extent and hit-test
geometry; it never enlarges a scalar into a multi-pixel block. A separate inspection
magnifier does not change the main surface's mapping.

If the rendered dimensions exceed the available viewport, the containing UI uses normal horizontal and/or vertical scrolling.

If a tensor dimension exceeds a WebGL2 texture or render-target limit, the renderer may partition the representation internally into multiple textures/bands. This is an implementation detail and must preserve the visible one-weight-to-one-pixel mapping and the logical identity of one complete matrix.

Query actual texture, renderbuffer and viewport limits. Use a bounded visible
framebuffer over the complete native scroll extent rather than requiring a giant
canvas. Disjoint texture bands are needed only when a dimension exceeds the
effective texture ceiling (the hardware limit or an explicitly lower resource
ceiling). Band edges must neither omit nor duplicate cells. Report unsupported
WebGL2, allocation failure or an unrepresentable browser scroll extent explicitly;
never silently scale the data to fit a limit. This partitioning does not change
complete-tensor downloading or introduce API tiles/prefetch.

## Numeric representation

The renderer consumes the logical visualization representation supplied by the backend. The initial representation is one `float32` logical value per weight regardless of the model's physical storage representation.

The tensor value remains authoritative. The renderer must not rewrite tensor values in order to change color, brightness, selection state, or another visual property.

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
hover, scrolling and transfer changes do not take this reconstruction path.

## Color

The default scientific palette is monochromatic sequential green. Hue does not
encode a second variable: scalar intensity controls monotonically increasing
luminance from near-black green through green to pale green. Matrix values and
distribution density use the same family. Warm amber/orange is reserved for
interaction overlays; future semantic modes require their own accepted design.

Display overlays must preserve the underlying scalar transfer and stored values.
Thin inspection guides may change final composited luminance, as defined below.

## Interaction scope

The proof of concept does not require zoom or pan. Detailed Tensor Explorer interaction behavior is owned by `tensor-explorer.md` and must not be inferred from this common rendering specification.

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
computed from exact integer logical coordinates, including texture-band origins.
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
