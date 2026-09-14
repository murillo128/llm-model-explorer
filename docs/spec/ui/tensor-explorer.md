# Tensor Explorer

## Authority and design references

This document owns Tensor Explorer-specific behavior, layout, interaction, and scientific-view requirements. Cross-cutting rules remain owned by the documents linked from [`../README.md`](../README.md) and are referenced here rather than repeated.

In particular:

- exact logical-cell rendering, scalar-value luminosity, semantic color, and Matrix Explorer viewport behavior are defined by [`rendering.md`](rendering.md);
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

The inventory follows the public descriptor's logical path segments, never filesystem structure. Branches alone use chevron disclosure controls. Each leaf is one compact row with a tensor icon, its relative final path segment, and inline shape/storage dtype when width permits. Selection highlights the row rather than opening a card. Use shallow, capped indentation, truncate long labels, and retain full public identity in accessible names/tooltips so duplicate leaf names remain distinguishable. Native disclosure and selection work with Enter/Space and Tab; arrow keys navigate visible rows, open/close branches, or return to a parent, with Home/End reaching the first/last visible row.

The inventory is secondary navigation and may be hidden to reclaim its workspace width, restored explicitly, and resized while visible. Its initial state is progressively collapsed rather than recursively opening deep layer internals. User choices for inventory visibility, width, and branch expansion should survive revisiting the explorer in the same browser. These navigation preferences do not change tensor identity or model state.

The selected tensor has one compact contextual header inside the workspace: its breadcrumb/path plus immediately useful shape and storage dtype. Do not repeat it in global chrome, a large tensor-name title, or a permanent logical-path field. Use descriptors supplied by the inventory/API rather than inferring model metadata from names.

An information control beside the tensor identity exposes secondary metadata on demand: full logical path (including any truncated portion), rank, element count, storage dtype/format, and logical dtype. Hover or keyboard focus may expose this information ephemerally. Explicit activation by click, tap, or keyboard pins it for continued reading; a pinned state can be closed explicitly, by Escape, or by interaction outside it. Filesystem paths never appear. General matrix-orientation or keyboard instructions are not tensor metadata and do not belong in this information view.

Pending/streaming operation state is presented within the stable tensor/matrix header so the matrix does not move when loading starts or finishes. Cancellation remains available while work is active. Successful results return to a quiet ready state without permanent completion labels. Keep independent tensor, statistics, and distribution error/cancellation states visible; auxiliary failure must not disable a successful matrix. Rendering resource failures remain explicit and actionable.

The primary screen should remain visually sparse. The current design has three primary data rectangles for a 2D tensor and does not add a fourth legend/control block in the lower-right corner merely to fill space.

## Matrix orientation and geometry

For a logical rank-2 tensor with shape `[rows, columns]`:

- tensor rows map to screen Y;
- tensor columns map to screen X;
- the matrix viewport's native data geometry is therefore `columns × rows` logical cells.

The default view does not transpose a matrix for aesthetic reasons. If a future explicit transpose operation is added, it must be visible as a view transformation and must not be confused with native tensor order.

Examples for SmolLM2-style MLP matrices make the convention concrete:

- a tensor shaped `[576, 1536]` has native geometry `1536 × 576` cells;
- a tensor shaped `[1536, 576]` has native geometry `576 × 1536` cells.

Matrix cells remain square at every zoom level. The default view enlarges an underfilled matrix to use the available width, but does not shrink below the native one-device-pixel-per-cell scale. Wider matrices therefore retain native scale and horizontal navigation rather than aggregating values. Exact logical mapping and the common camera rules are inherited from [`rendering.md`](rendering.md).

## Matrix Inspector layout

A rank-2 Tensor Explorer uses three aligned surfaces:

1. **Main matrix** — the exact tensor surface.
2. **Row distributions** — a narrow panel immediately to the right, aligned with matrix rows.
3. **Column distributions** — a short panel immediately below, aligned with matrix columns.

The current design uses a small gap between the main matrix and each distribution panel so the surfaces remain visually distinct while preserving obvious alignment.

For the current reference layout, the distribution depth is `100 px`:

- for a `rows × columns` tensor, the row-distribution panel has 100 value bins across its horizontal depth;
- the column-distribution panel has 100 value bins across its vertical depth.

This means a `[576, 1536]` tensor has 576 row profiles and 1536 column profiles, each using the same 100-bin value domain. The `100 px` dimension corresponds to the current 100-bin visual design. It is a Tensor Explorer visualization parameter, not a tensor dimension.

### Row distributions

Each tensor row owns exactly one logical row profile in the right-hand panel. Along that profile, the panel represents the distribution of values in that tensor row across the shared histogram bins.

The profile's on-screen thickness along Y follows the same matrix camera scale as the corresponding logical row, so zooming the matrix preserves row alignment. The histogram-bin depth itself remains fixed and does not grow with matrix zoom.

### Column distributions

Each tensor column owns exactly one logical column profile in the bottom panel. Along that profile, the panel represents the distribution of values in that tensor column across the shared histogram bins.

The profile's on-screen thickness along X follows the same matrix camera scale as the corresponding logical column, while the histogram-bin depth remains fixed.

### Histogram consistency and numeric scale

All row profiles in one tensor use the same bin boundaries, and all column profiles in that tensor use the same bin boundaries, so navigation does not change the histogram coordinate system underneath the user.

The distribution surfaces expose enough numeric scale information to interpret the value axis: the lower and upper endpoints of the active bin domain, the mathematically correct location of zero when zero lies inside that domain, and the tensor's true finite minimum and maximum. If a future robust/clipped domain differs from the true extrema, its endpoints must not be mislabeled as `min`/`max`; the true extrema remain distinguishable from the displayed domain.

The density/intensity normalization may be chosen to keep profiles readable, but it must be stable for the open tensor and must not silently change on hover, scroll, or zoom. Camera navigation changes which rows/columns are visible, not the value-bin domain.

The matrix must be usable before distribution/statistics results have completed. Distribution panels may progressively become available after the tensor itself; they must not block first render. The production and transport of these statistics remain owned by the backend/API specifications.

## Matrix viewport navigation

The Matrix Explorer owns one local camera with a uniform scale for both axes. Wheel and pinch gestures zoom around the pointer or gesture focal point so the logical cell under that point remains stable as far as matrix bounds permit. A fit-width reset returns the matrix to its default view. Camera changes are local to the current Matrix Explorer and never alter another matrix view or the Tensor inventory.

The user may also navigate directly to a region. Dragging a nontrivial rectangle over the matrix selects exact logical row/column bounds and previews that pending zoom region with the interaction accent. Releasing the selection zooms to the largest uniform square-cell scale that fits the selected rectangle in the matrix viewport. Cancelling the gesture leaves the camera unchanged.

The aligned distribution panels provide one-axis equivalents. Selecting a start/end range in the bottom column-distribution panel zooms to that column range; selecting a range in the right row-distribution panel zooms to that row range. The same square-cell camera is used, and the orthogonal view position is preserved as far as bounds allow. These selections are navigation gestures only; they do not select, transform, or export tensor data.

## Scroll synchronization

Tensor Explorer is a bounded workspace inside the fixed application shell. The
inventory owns its vertical overflow independently from the selected tensor's
scientific pane. Neither inventory navigation nor scientific scrolling moves the
document, contextual header, or application bars. The default rank-2 view has one
inventory scroller and one matrix scroller; the scientific pane and its wrappers
do not add nested scroll surfaces. Transient notices and on-demand metadata may
retain their own bounded overflow.

At desktop widths the inventory uses 200–280 CSS pixels and the scientific pane
has a 360 CSS-pixel minimum. Below 760 CSS pixels the panes stack: the inventory
receives 24% of workspace height (at least 64 CSS pixels), and the scientific pane
receives the remainder. This preserves usable scientific chrome instead of
collapsing a side-by-side matrix pane. Native data geometry never shrinks to fit.

The matrix viewport receives the scientific pane's remaining height after context,
distribution depth, and gaps. Reserve its vertical scrollbar
gutter deterministically, including that gutter in the native-width layout track.
The baseline keeps the native vertical scrollbar track present even for short
tensors, avoiding changes to content width when vertical overflow changes.
Use the viewport's actual client dimensions, excluding scrollbars, for rendering.
A tall tensor whose native width fits must not gain horizontal overflow merely
because a vertical scrollbar appears. Genuine excess width retains horizontal
scrolling. Short matrices and rank-1 strips retain their full data height in
addition to any horizontal scrollbar chrome.

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

Hover must resolve the exact logical cell at every zoom level. Zooming must not average neighboring values or change which tensor value is reported for a logical coordinate.

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

When the main matrix is zoomed far enough that the local cell neighborhood is directly legible, the magnifier is suppressed automatically. The row/column/value readout and the active logical cell remain available. Zooming back out restores the magnifier without changing logical selection, and the transition should avoid flicker near the visibility threshold.

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
- light, minimal, editor-like UI shell.

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

The matrix viewport and current logical camera resolve pointer coordinates to exact row/column identity; texture-band origins affect sampling, not logical selection identity. Readout uses the retained float32 scalar value's shortest JavaScript round-trip decimal, with explicit `-0`, `NaN`, and signed infinities. Pending cells report unavailable and no numeric value. Stream and transfer updates refresh an active inspection; leaving, disposing, or losing the matrix context clears it. Focus exposes the first visible coordinate, arrow keys inspect adjacent cells with navigation as needed, and Escape/blur clear inspection. The focus outline and coordinate/value text remain independent of hue.
