# Tensor Explorer

## Authority and design references

This document owns Tensor Explorer-specific behavior, layout, interaction, and scientific-view requirements. Cross-cutting rules remain owned by the documents linked from [`../README.md`](../README.md) and are referenced here rather than repeated.

In particular:

- exact scalar-cell rendering, scalar-value luminosity, semantic color, and viewport overflow behavior are defined by [`rendering.md`](rendering.md);
- browser/runtime and WebGL2 boundaries are defined by [`architecture.md`](architecture.md);
- tensor inventory, logical tensor materialization, and model-format handling are defined by [`../backend/models.md`](../backend/models.md);
- statistics and tensor transfer are API/backend concerns defined by [`../api/contract.md`](../api/contract.md) and [`../api/binary-streaming.md`](../api/binary-streaming.md).

Current visual design references:

- [Miro board — llm-model-explorer wireframes](https://miro.com/app/board/uXjVHnoDEYY=/)
- [Miro — Matrix Inspector light/hover prototype](https://miro.com/app/board/uXjVHnoDEYY=/?moveToWidget=3458764683523196190)
- [Miro — chroma-selection Matrix Inspector prototype](https://miro.com/app/board/uXjVHnoDEYY=/?moveToWidget=3458764683528973088)
- [Canva folder — LLM Model Explorer design explorations](https://www.canva.com/folder/FAHVARFRfSY)

The written specification is normative when a mockup and this document disagree. Mockups are visual references, not alternative contracts.

## Proof-of-concept scope

The Tensor Explorer opens tensors from the hierarchical inventory of the session's selected model. The proof of concept supports complete rank-1 and rank-2 logical tensors; the detailed Matrix Inspector interaction below applies to rank-2 tensors.

The initial reference model is the product-level reference model, `HuggingFaceTB/SmolLM2-135M` Base. The Tensor Explorer itself must remain model-generic and must not hard-code SmolLM2 layer names or dimensions.

Rank-1 tensors remain in scope, but the dedicated design work so far has not fixed a separate rank-1 interaction/layout beyond using the common exact renderer. Do not infer a specialized rank-1 histogram or magnifier layout from the rank-2 rules below.

## Tensor navigation and header

The inventory follows the public descriptor's logical path segments, never filesystem structure. Branches use folder icons and alone use chevron disclosure controls. Each leaf is one compact row with a tensor icon, its relative final path segment, and inline shape/storage dtype when width permits. Selection highlights the row rather than opening a card. Use shallow, capped indentation, truncate long labels, and retain full public identity in accessible names/tooltips so duplicate leaf names remain distinguishable. Native disclosure and selection work with Enter/Space and Tab; arrow keys navigate visible rows, open/close branches, or return to a parent, with Home/End reaching the first/last visible row.

The inventory is secondary navigation in a dedicated left track outside the
Matrix Explorer panel. The expanded drawer has a compact `Inventory` header and
an accessible collapse control. Collapsing replaces the pane and divider with a
40 CSS-pixel icon-only rail: only the Inventory restore icon is persistent, with
no text, disclosure chevron, or empty pane. The restore button has an accessible
name and a hover/focus tooltip. Focus moves to the corresponding control after
collapse/restore. Restoring recovers the prior expanded width and tree state.
The drawer never overlays the matrix; its collapse/restore resizes the scientific
workspace while keeping selection, camera, operations, and streaming consumers
mounted. There is no automatic hiding on selection or inventory search/filter.
Global explorer navigation remains the existing top tabs.

On a fresh browser preference state, open only first-level branches. Deeper
branches, including layer lists and their internals, start collapsed. Remember
explicit branch expansion/collapse choices by the array of public logical path
segments (shared across models with matching paths); never use filesystem identity.
Visibility, preferred desktop width, and branch choices persist in local browser
storage across explorer visits and reloads. If storage is unavailable or malformed,
use safe defaults and retain working interactions in memory.

The selected tensor has one persistent scientific-panel header, composed through the reusable Matrix Explorer header. Use a single compact row, approximately 36–44 CSS pixels high, with its own subtle panel background and separator. Keep it stationary while the scientific content scrolls. Emphasize its logical breadcrumb/path with the existing scientific green accent, keeping shape/storage dtype visually secondary on the same row and the accessible standard information icon adjacent to this identity metadata. The scientific surfaces begin immediately below the header without a reserved secondary metadata row. Do not repeat identity in global chrome, a large title, a permanent logical-path field, or a second metadata block. Use public descriptors supplied by the inventory/API, never inferred names or filesystem paths.

The information control opens a lightweight labelled popover directly below the icon, clamped within the scientific pane when necessary. Pointer hover and keyboard focus show an ephemeral preview without moving focus or showing a close control. Click, tap, Enter, or Space pins the popover and moves focus to its small accessible × close control. Pinned metadata survives pointer departure and focus navigation; the close control, outside pointer interaction, or Escape dismisses it. Close and Escape restore trigger focus without immediately reopening the preview. Touch activation requires no hover. The popover contains full logical path, rank, element count, storage dtype, storage format when supplied, and logical dtype. For rank-2 tensors it also contains the distribution-domain description and authoritative true finite minimum/maximum, reporting pending or unavailable metadata honestly. General pixel-orientation and keyboard help do not belong here. It floats over the content without reserving scientific width.

Pending/streaming feedback and cancellation occupy reserved space inside the header, with accessible compact text and a spinner. The header height and matrix origin remain fixed across loading, streaming, ready, cancelled, and successful completion. At constrained widths, the status slot may scroll horizontally while cancellation, information, and `Fit width` remain reachable. Ordinary success clears status without leaving Complete/Ready labels. Preserve independent tensor, statistics, and distribution errors/cancellation; auxiliary failures must not disable a usable matrix. Rendering resource failures remain explicit and actionable. Use the composable right-side action slot for operation actions followed by the compact `Fit width` camera reset at the far right.

The primary screen should remain visually sparse. The current design has three primary data rectangles for a 2D tensor and does not add a fourth legend/control block in the lower-right corner merely to fill space.

## Matrix orientation and geometry

For a logical rank-2 tensor with shape `[rows, columns]`:

- tensor rows map to screen Y;
- tensor columns map to screen X;
- the matrix viewport's native data geometry is therefore `columns × rows` logical cells.

The default view does not transpose a matrix for aesthetic reasons. If a future explicit transpose operation is added, it must be visible as a view transformation and must not be confused with native tensor order.

At native scale (`s=1`), examples for SmolLM2-style MLP matrices make the convention concrete:

- a tensor shaped `[576, 1536]` has native geometry `1536 × 576` cells;
- a tensor shaped `[1536, 576]` has native geometry `576 × 1536` cells.

Exact mapping and overflow/scroll behavior are inherited from [`rendering.md`](rendering.md). Matrix Explorer applies its shared square-cell camera to the matrix and aligned profile axes.

## Matrix Inspector layout

A rank-2 Tensor Explorer uses three aligned surfaces:

1. **Main matrix** — the exact tensor surface.
2. **Row distributions** — a narrow panel immediately to the right, aligned with matrix rows.
3. **Column distributions** — a short panel immediately below, aligned with matrix columns.

The current design uses a small gap between the main matrix and each distribution panel so the surfaces remain visually distinct while preserving obvious alignment.

Distribution thickness is fixed at 100 device pixels (100 bins), independent of
matrix zoom. CSS chrome/gaps remain fixed. Treat the matrix and its right/bottom
profiles as one aligned scientific object: center independently on each axis
where the data underfills the available matrix viewport at the current scale.
Attach each profile to the corresponding visible matrix bound, including when
the data is short or narrow; do not leave a gap to a viewport edge. Overflowing
axes retain normal native scrolling and scrollbar clearance. Their visible
data-axis extents match the matrix canvas, excluding native scrollbars.
Centering changes only device-snapped presentation offsets, never camera scale,
logical origins, square cells, sampling, or scalar/count storage.

### Row distributions

Each logical row owns one horizontal profile in the right panel. Its vertical
extent scales with that matrix row and shares the exact matrix Y camera origin.
The horizontal bin axis retains its fixed thickness; this is a density image,
not a separate bar chart for each row.

### Column distributions

Each logical column owns one vertical profile in the bottom panel. Its horizontal
extent scales with that matrix column and shares the exact matrix X camera origin.
The vertical bin axis retains its fixed thickness.

### Matrix camera

Each Matrix Explorer instance defaults to **fit width without minification**:
`s = max(1, floor(availableCSSWidth * DPR) / columns)`. Underfilled matrices expand
uniformly; wider matrices remain at native scale and use horizontal scrolling.
The `Fit width` header action restores this scale, resets scroll origins and clears zoom-back history.
While fit mode remains active, resizing refits the width. Manual zoom retains its
scale across layout changes. A new source starts a fresh local camera.

Wheel and trackpad/touch pinch zoom around their pointer/gesture focal point,
retaining its logical coordinate as far as native scroll rounding and bounds
permit. Zoom-out stops at native 1:1. Zoom-in supports substantial enlargement
(the implementation ceiling accommodates both 64 times fit width and a single
logical cell filling either viewport dimension). Scrollbars remain the panning
mechanism; primary pointer drag selects a region for direct zoom. No minimap is
present. Camera changes never move the document,
inventory, another Matrix Explorer or Tokenizer prompt/editor.

The transform applies equally to X and Y, preserving square cells. Hover, click
and keyboard inspection resolve exact logical indices, including pending cells.
DPR changes preserve logical camera origins as far as scroll bounds permit.

### Histogram consistency and numeric scale

All row profiles in one tensor use the same bin boundaries, and all column profiles in that tensor use the same bin boundaries, so navigation does not change the histogram coordinate system underneath the user.

The distribution surfaces continuously expose enough numeric scale information to interpret the value axis: the lower and upper endpoints of the active bin domain and the mathematically correct location of zero when zero lies inside that domain. The tensor's true finite minimum and maximum remain available in the information popover. If a future robust/clipped domain differs from the true extrema, its endpoints must not be mislabeled as `min`/`max`; the true extrema remain distinguishable from the displayed domain.

The density/intensity normalization may be chosen to keep profiles readable, but it must be stable for the open tensor and must not silently change on hover, scroll, or zoom. Camera navigation changes which rows/columns are visible, not the value-bin domain.

Both orientations expose a compact numeric ruler for the actual shared bin
domain: low to high runs left to right for rows and top to bottom for columns.
The right-hand row-profile ruler runs horizontally above the histogram width,
with low at its left edge, high at its right edge, and zero at its actual numeric
position when present. Align its ticks and labels to the histogram data rectangle,
including any device-pixel alignment offset; they must not drift from its edges.
Use the distribution metadata's endpoints, never the matrix's robust luminosity
anchors. The current API uses the full finite range; label it as full-range bins
and show the authoritative true finite `min` and `max` in the tensor information
popover alongside that domain description. Rounded display values retain exact endpoints in accessible ruler
names and endpoint tooltips. A narrow profile with extreme outliers can correctly
reflect concentration near zero; do not stretch its numeric coordinates to fill
the panel. If a later accepted bin domain is clipped, describe its endpoints as
bin/clipped bounds and keep true extrema separate.

For a distinct domain containing zero, place a quiet neutral zero reference at
`(0 - low) / (high - low)` of its bin depth, including at either boundary.
Graphite rulers/text and a subtle neutral guide remain materially quieter than
amber inspection guides. Keep rulers outside the data rectangles and retain the
100-device-pixel profile depth. Constant finite data reports both equal endpoints
and explicitly states that samples occupy bin 50 with no numeric span; do not
invent a zero location. All-nonfinite data reports no finite domain/extrema and
has no numeric ticks. Before distribution metadata arrives, the domain is
unavailable; valid metadata can label a progressive prefix without claiming that
the operation is complete. Statistics arrival/failure does not change this scale.
Keep these descriptive states in the information popover, with wrapped text and
keyboard/touch-accessible bounded overflow at constrained widths. Do not reserve
a permanent secondary metadata line; metadata arrival or failure never shifts
the matrix origin. The numeric rulers remain visible independently of the popover.

Hover, native scrolling, and any separately accepted camera navigation change
only inspected/visible matrix coordinates; they must not rescale value bins or
their labels. Auxiliary operation status remains independent from these labels.

The matrix must be usable before distribution/statistics results have completed. Distribution panels may progressively become available after the tensor itself; they must not block first render. The production and transport of these statistics remain owned by the backend/API specifications.

## Matrix viewport navigation

The Matrix Explorer owns one local camera with a uniform scale for both axes. Wheel and pinch gestures zoom around the pointer or gesture focal point so the logical cell under that point remains stable as far as matrix bounds permit. A fit-width reset returns the matrix to its default view. Camera changes are local to the current Matrix Explorer and never alter another matrix view or the Tensor inventory.

Primary pointer dragging directly navigates to a region, without a mode toggle or
confirmation. A matrix drag of at least 5 CSS pixels previews an arbitrary rectangle
with a thin warm amber border and an 8%-opacity amber fill. Snap both endpoints to
the nearest exact logical cell boundary, using the same raster edges as hit testing;
normalize forward and reverse drags into half-open integer row/column bounds. Clip
the preview to the data surface and clamp captured pointers outside it to its edge.
The preview is small transient DOM state, independent of scalar/count storage and
transfer settings; underlying variation remains readable.

During the preview, project the same exact snapped row range into the right
profile and column range into the bottom profile. All three surfaces update
synchronously for forward/reverse drags; one-cell ranges use thin exact
row/column indicators. Clearing or consuming the preview clears all projected
ranges; no persistent tensor-region selection is created.

On release, fit the entire rectangle with the largest uniform square-cell scale
allowed by the available matrix viewport dimensions, retaining the native 1:1
minimum. Center the selected region in any spare dimension and clamp origins to
matrix bounds. Use the full available viewport even when current data underfills
it; account for native scrollbar geometry. Zero-width/height selections are no-ops.
A movement below the threshold remains ordinary exact cell click/tap inspection.
During an active drag, suspend hover inspection and suppress the release click so
navigation does not become data selection. Resume normal inspection afterward.

The aligned distribution panels provide one-axis equivalents. Dragging at least
5 CSS pixels along the bottom panel selects a column boundary range; dragging
along the right panel selects a row boundary range. Use the same amber preview
across the fixed bin depth without changing counts. On release, fill the aligned
matrix viewport dimension with that range using the same uniform camera; retain
the previous logical center on the orthogonal axis as far as bounds allow.
Its preview projects the selected axis back across the matrix extent with the
same exact bounds; it does not invent a range on the unselected axis.

Escape, pointer cancellation/capture loss, window blur, camera/viewport changes,
source replacement and context loss remove a pending preview without applying it.
Wheel zoom cancels a pending selection before using the existing focal zoom.
One-finger dragging on a data surface selects a region/range; a second touch
cancels selection and allows the existing two-finger matrix pinch. Native
scrollbars continue to own panning. These gestures remain local to their Matrix
Explorer and never transform, export, or select tensor data or zoom prompt views.

### Zoom-back history and dismissal priority

Each Matrix Explorer source owns a bounded history of 32 committed logical
scale/origin states. A region/range zoom adds one prior state; wheel/trackpad
events within 180 ms of the previous event form one gesture, and a two-finger
pinch forms one gesture until its touches end. No-op zooms add no history.
`Fit width` and source replacement clear history. Resize/DPR changes retain
history; restoring it uses current viewport/native scroll bounds.

Escape first cancels an active drag/preview. Focused modal/info-popover dismissal
retains its local priority. Otherwise Escape restores the previous camera when
the scientific surfaces own keyboard focus, or are hovered with neutral document
focus. A hover-only magnifier never consumes Escape just to hide itself. Right
click on a matrix or distribution data surface restores the same previous camera
and suppresses that surface's browser context menu. Other surfaces and global
browser shortcuts remain unchanged. With no prior state, camera back is a no-op.

## Scroll synchronization

Tensor Explorer is a bounded workspace inside the fixed application shell. The
inventory owns its vertical overflow independently from the selected tensor's
scientific pane. Neither inventory navigation nor scientific scrolling moves the
document, contextual header, or application bars. The default rank-2 view has one
inventory scroller and one matrix scroller; the scientific pane and its wrappers
do not add nested scroll surfaces. Transient notices and on-demand metadata may
retain their own bounded overflow.

Above 760 CSS pixels, the visible inventory defaults to 280 CSS pixels. A
focusable vertical divider supports pointer dragging, Left/Right arrows in 16 px
steps, and Home/End to reach width bounds. Clamp inventory width to 200–480 CSS
pixels and to the available workspace width minus the 16 px divider and the
scientific pane's 360 CSS-pixel minimum. A temporarily constrained viewport must
not overwrite the user's preferred width.

At or below 760 CSS pixels the visible panes stack: the inventory receives 24%
of workspace height (at least 64 CSS pixels), and the scientific pane receives
the remainder with an 8 px gap. The desktop width divider is unavailable in this
layout; the saved desktop width is retained. Collapsing removes the inventory
row and gap, leaving the same 40 CSS-pixel left rail beside a full-height
scientific pane. The inventory's compact controls remain reachable
while its tree scrolls. Native data geometry never shrinks to fit.

The matrix viewport receives the scientific pane's remaining height after context,
transient feedback, distribution depth, and gaps. Reserve its vertical scrollbar
gutter deterministically, excluding that gutter from the available fit width.
The baseline keeps the native vertical scrollbar track present even for short
tensors, avoiding changes to content width when vertical overflow changes.
Use the viewport's actual client dimensions, excluding scrollbars, for rendering.
A tall tensor whose native width fits must not gain horizontal overflow merely
because a vertical scrollbar appears. Genuine excess width retains horizontal
scrolling. Short matrices retain their zoomed data height within the bounded viewport; rank-1 strips retain their full native data height plus horizontal scrollbar chrome.

Scrolling and zoom preserve distribution alignment:

- vertical navigation of the main matrix keeps the right-hand row-distribution panel on the same logical rows and at the same data-axis scale;
- horizontal navigation keeps the bottom column-distribution panel on the same logical columns and at the same data-axis scale;
- distribution thickness across the value-bin axis remains fixed while the shared row/column axis follows the matrix camera;
- labels and floating hover UI may remain viewport-relative, but data coordinates must remain exact.

## Hover inspection

Hover is the primary fine-grained inspection interaction for a 2D tensor.

When the pointer is over a populated matrix cell, Tensor Explorer resolves the exact tensor cell and exposes at least:

- row index;
- column index;
- logical tensor value.

The current visual treatment uses a compact monospace readout adjacent to the magnifier rather than permanently printing coordinates over the matrix.

Hover must resolve the exact cell at every camera scale, including native one-pixel resolution. Never average cells or report a neighbor’s value.

If a progressively streamed region has not arrived yet, the UI must not invent a value for that cell.

### Cursor

Inside the matrix, the hover pointer is represented as a small warm-orange/amber cross rather than a large arrow cursor or filled marker. The cross should remain visually small enough that the underlying cell and immediate neighborhood remain legible; a small center gap is preferred to covering the selected pixel with an opaque mark.

The cross is an interaction marker only. It never changes the tensor value or its luminosity mapping.

## Magnifier

At low and medium matrix zoom, hovering a matrix cell opens a local neighborhood magnifier near the pointer.

The magnifier:

- enlarges source cells with pixel-preserving nearest rendering; it does not interpolate between weights;
- preserves the same luminosity and semantic-color rules as the main matrix;
- identifies the center cell with a thin warm-orange border and uses thin centered horizontal/vertical guides as secondary orientation aids without replacing neighboring cell colors;
- repositions around the active cell when necessary to stay inside the usable scientific pane, avoid the right/bottom distribution panels, and avoid covering the inspected neighborhood when another valid position exists;
- is an inspection aid, not the Matrix Explorer camera itself.

The magnifier does not change matrix scale or scroll position. It is suppressed
when the effective on-screen logical cell size reaches 10 CSS pixels per side
(`camera scale / devicePixelRatio`). Once hidden, it returns only below 8 CSS
pixels per side. These per-instance hysteresis thresholds are named UI tokens in
`inspection-layout.ts`. Hiding or restoring it preserves the active logical cell,
linked inspection state, and compact row/column/value readout. When shown, it
continues to inspect the same 9×9 logical neighborhood.

Placement follows the current pointer/focused cell with a small offset inside
the visible scientific pane, rejects intersections with either distribution data
surface, and avoids the inspected neighborhood when a legal position permits.
Try pointer-relative placements first, flipping left/above near boundaries;
clamp only as a final fallback. Recompute on pointer movement, scroll, zoom,
resize, DPR changes, source replacement and layout shifts affecting usable bounds.
If the pane cannot accommodate the complete card/readout surface, retain the
compact readout instead.

## Selection encoding: green data and amber inspection guides

Unselected matrix and distribution pixels use the common renderer's sequential
green scalar family. The underlying scalar transfer remains independent of
selection. Hover/focus on a populated cell links exactly one active coordinate:

- a thin high-contrast amber horizontal guide through the matrix row;
- a thin amber vertical guide through the matrix column;
- a stronger amber intersection at the exact active cell;
- the same row guide across the right distribution scanline;
- the same column guide across the lower distribution scanline.

Guides normally occupy one device pixel. They blend over the scientific display
at draw time, retaining underlying variation; they are not opaque replacement
lines. The common renderer owns concrete palette, opacity and display encoding.
Pending/nonfinite pixels retain explicit status colors. Pending-cell inspection
may expose unavailable text but does not activate linked guides until populated.

The small amber cursor and magnifier center marker remain available. The magnifier
uses the same green transfer and active accent as the matrix. Exact coordinates,
value text and a visible keyboard focus outline communicate state without relying
on color alone. Arrow-key focus uses the same coordinate propagation as hover.

## GPU interaction-state constraint

Hover, selection, and zoom must not require a recolored copy of the tensor or a second copy of the weights in GPU memory.

For the standard single-cell hover state, the renderer should be able to express selection with small interaction state such as the active row index, active column index, hover flag, and semantic-color parameters. The shader derives row/column/intersection overlays from that state while reading the same immutable scalar tensor representation.

A per-pixel selection texture/mask is unnecessary for the normal one-cell hover case and should not be introduced merely to recolor a row and column.

## Visual style

The preferred Tensor Explorer shell is light and editorial rather than terminal-like: warm/off-white page background, white data panels, graphite text, subtle borders and shadows, restrained typography, and generous empty space around labels.

Warm orange/amber is the current interaction accent. It is deliberately distinct from the data luminosity channel.

The UI chrome must stay secondary to the tensor. Avoid heavy panels, decorative legends, or saturated background areas that compete with the matrix.

The unselected tensor color treatment remains governed by [`rendering.md`](rendering.md); the green-tinted design explorations in Canva/Miro are visual experiments and do not override the common rule that scalar value is encoded by luminosity and semantic color is independent.

## Reusable Matrix Explorer primitive

The Matrix Inspector must be implemented as a reusable matrix/tensor exploration primitive rather than as a weight-only screen. The same aligned matrix, zoom/navigation, hover, magnifier, and linked-selection concepts are intended to be reused later for:

- token × hidden activation matrices;
- attention/token × token matrices;
- intermediate inference tensors;
- matrix/vector-operation inputs and outputs;
- other rank-2 scientific views.

A future matrix-multiplication view may compose several Matrix Explorer instances side by side. Hovering an output cell can then highlight the corresponding input row and column without changing the underlying matrix values. That future composition is outside the proof-of-concept, but this component must not make it impossible.

## Scientific interpretation guardrails

A rendered weight matrix is a matrix coordinate system, not a natural 2D physical field. Native hidden-unit ordering has no guaranteed spatial meaning.

Consequently:

- apparent blobs, filaments, periodicity, or local orientation in native order are hypotheses, not evidence by themselves;
- any row/column reordering must be explicitly labeled as a transformed view;
- scientific analysis views should be compared with appropriate shuffled/random controls before visual structure is treated as meaningful;
- visual manageability of a matrix is not evidence that its information is compressible.

These guardrails are part of the explorer because it is intended as an analysis instrument, not merely an image viewer.

## Future analysis modes — captured design backlog, not proof-of-concept requirements

The following ideas have been discussed for later Tensor Explorer iterations. They are recorded here so the dedicated design work is not lost, but they do **not** extend the current proof-of-concept scope in `product.md`.

### Derived scalar and matrix views

Candidate views include:

- raw tensor values;
- absolute value/magnitude;
- block/local RMS or energy;
- robust outlier score/density;
- `W Wᵀ` and `Wᵀ W` correlation/Gram views;
- 2D FFT/power spectrum, including radial spectrum and angular anisotropy;
- autocorrelation;
- local structure-tensor anisotropy/orientation;
- wavelet or multiscale views;
- singular-value/spectral summaries.

Derived views must be visibly identified as derived data rather than silently replacing the native tensor.

### Controls and null models

Scientific controls should be first-class comparison targets where relevant, including:

- shuffled tensor values;
- matched-marginal random controls;
- Gaussian controls with matched simple statistics;
- independently row- or column-shuffled views;
- random orthogonal-coordinate controls where mathematically appropriate.

### Non-destructive reordering

Future views may reorder rows/columns for interpretation without mutating the source tensor. Candidate orderings include clustering, cosine/similarity ordering, activation-based ordering, spectral ordering, and SVD-related ordering.

The native order must remain directly available, and the UI must make the active permutation explicit. Conceptually, reordered views are view transforms such as `P_r W P_cᵀ`, not edits to `W`.

### Comparison mode

A later comparison mode may show two tensors and their delta using a shared visual scale, for example:

- model A vs model B;
- layer vs layer;
- Base vs Instruct/SFT;
- before/after fine-tuning, LoRA merge, pruning, distillation, or another transformation.

Shared scaling is important in comparisons: independent per-tensor autoscaling can fabricate apparent differences. Comparison views should therefore support a common/family scale as well as any per-tensor inspection scale.

### Alternative semantic color channels

The standard inspector uses chroma for interaction. Future analysis modes may use semantic chroma for additional derived information while keeping scalar luminosity independent. Candidate mappings discussed include:

- row-relative and column-relative scores;
- local anisotropy magnitude/orientation encoded as a two-component chroma vector;
- activation statistics;
- sensitivity/gradient information;
- cluster or categorical identity.

Any such mode must state what the chroma means and must compose selection in a way that keeps hover/selection clearly distinguishable.

### Region-focused analysis

A later analysis pane may operate on an explicitly selected tensor region such as `[rows a:b, cols c:d]` so expensive or detailed analyses can focus on a subset while the complete tensor remains visible in context.

## Acceptance summary for the current Matrix Inspector

A conforming current Matrix Inspector for rank-2 tensors has all of these properties:

- native `[rows, columns]` orientation with columns on X and rows on Y;
- exact logical cell/value identity with square-cell scaling;
- an underfilled-width default view plus local wheel/pinch and direct region/range zoom navigation;
- aligned right-hand row distributions and bottom column distributions, currently 100 bins deep, sharing the matrix camera on their data axes;
- synchronized navigation that preserves row/column alignment;
- distribution scale information for domain endpoints, zero, and true finite extrema;
- hover readout for the exact cell value and coordinates;
- small orange cross cursor;
- collision-aware pixel-preserving neighborhood magnifier at low/medium zoom, suppressed when the main view is already locally legible;
- green scalar transfer with thin amber row/column guides and a stronger intersection;
- linked selection state in both distribution panels;
- guides blend over data without changing scalar transfer or stored values;
- no recolored duplicate of the tensor in GPU memory;
- light, minimal, editor-like UI shell;
- local square-cell camera with fit-width reset, focal-point wheel/pinch zoom and native scroll navigation; no minimap or primary-drag pan.

## Progressive composition and density baseline

The scientific screen tracks tensor, statistics and distribution operations
independently. A received prefix remains explicitly incomplete until that
operation completes successfully, including validation through stream EOF.
Failure or cancellation preserves any received prefix for inspection; an
auxiliary failure does not invalidate successfully received tensor data.
Statistics update the scalar transfer only after a valid successful result,
without reuploading weights. Selection generations fence all callbacks, including
returning to the same tensor. Disposing the view cancels its own operation handles
and releases its rendering resources.

Distribution storage preserves the API's uint32 counts. The baseline density is
`log1p(count) / log1p(axis_length)`, evaluated only for drawing; `axis_length` is
columns for row profiles and rows for column profiles. A zero-length axis maps to
zero intensity. This normalization is fixed for the tensor, independent of hover,
arriving subsets or the maximum count observed so far. Unknown counts use the
same visibly unpopulated treatment as unknown tensor pixels. The common domain
and binning remain backend/API-owned.

The matrix and both profiles share the common renderer's logical coordinate
convention and camera on the row/column data axes. Distribution depth remains
100 device pixels across the value-bin axis, with CSS gaps for chrome. Matrix
navigation drives the right profile's Y origin/scale and the lower profile's X
origin/scale; both profiles cover the same visible logical data extent as the
matrix on their shared axis. Empty tensors show an explicit empty state without
starting tensor/statistics/distribution operations. Rank-1 uses the common strip
with statistics and no distributions.

### Inspection implementation and resource ownership

The renderer's snapped canvas rectangle, current logical camera origin and uniform
cell scale resolve pointer coordinates through DPR; texture-band origins affect sampling,
not logical selection identity. Readout uses the retained float32 scalar value's
shortest JavaScript round-trip decimal, with explicit `-0`, `NaN`, and signed
infinities. Pending cells report unavailable and no numeric value. Stream and
transfer updates refresh an active inspection; leaving, disposing, or losing the
matrix context clears it. Focus exposes the first visible coordinate, arrow keys
inspect adjacent cells with native scrolling at the current scale as needed, and
blur clears inspection. Escape follows the dismissal/zoom-back priority above.
The focus outline and coordinate/value text remain independent of hue.

The 170px magnifier card presents a 9×9 display canvas at 162×162 CSS pixels with
nearest-neighbor enlargement. It reuses the matrix shader and existing scalar
bands through one renderer-owned 9×9 RGBA8 renderbuffer/framebuffer. A 324-byte
readback transfers only the small rendered display image to the card's 2D canvas;
it is not scalar storage, a selection mask, or a weight upload. There is no new
WebGL context, weight texture, tensor array, or per-hover request. The offscreen
pass does not resize the main canvas or change its view/scroll/scale. Clear the
inspection framebuffer to unavailable before drawing intersecting source bands,
so out-of-bounds cells never repeat edge weights. Dispose the display resources
with the owning renderer, including context loss/reconstruction.

The card/readout chooses among four pointer-relative placements and clamps within
the viewport while excluding the inspected neighborhood where viewport dimensions
permit a 170×212 CSS-pixel inspection surface. The magnified display retains its thin center marker in addition to linked
guides. Concrete palette, compositing and display encoding are owned by
[`rendering.md`](rendering.md#concrete-linear-srgb-scalar-and-guide-transfer).
