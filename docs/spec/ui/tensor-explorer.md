# Tensor Explorer

## Authority and design references

This document owns Tensor Explorer-specific behavior, layout, interaction, and scientific-view requirements. Cross-cutting rules remain owned by the documents linked from [`../README.md`](../README.md) and are referenced here rather than repeated.

In particular:

- exact one-value-to-one-pixel rendering, scalar-value luminosity, semantic color, and viewport overflow behavior are defined by [`rendering.md`](rendering.md);
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

The inventory is presented hierarchically using the tensor's logical/model name structure. Selecting a tensor opens one complete Tensor Explorer view; filesystem paths never appear in the UI.

The inspector header should make the current context unambiguous without consuming significant data-view area. It may show the model, logical layer/module path, tensor name, shape, and the fact that the data view uses exact pixel mapping. The UI should use the descriptors already supplied by the inventory/API instead of reconstructing model metadata from names.

The primary screen should remain visually sparse. The current design has three primary data rectangles for a 2D tensor and does not add a fourth legend/control block in the lower-right corner merely to fill space.

## Matrix orientation and geometry

For a logical rank-2 tensor with shape `[rows, columns]`:

- tensor rows map to screen Y;
- tensor columns map to screen X;
- the matrix viewport's native data geometry is therefore `columns × rows` pixels.

The default view does not transpose a matrix for aesthetic reasons. If a future explicit transpose operation is added, it must be visible as a view transformation and must not be confused with native tensor order.

Examples for SmolLM2-style MLP matrices make the convention concrete:

- a tensor shaped `[576, 1536]` is rendered `1536 px` wide × `576 px` high;
- a tensor shaped `[1536, 576]` is rendered `576 px` wide × `1536 px` high.

Exact mapping and overflow/scroll behavior are inherited from [`rendering.md`](rendering.md). Tensor Explorer must not independently rescale a matrix to make it fit.

## Matrix Inspector layout

A rank-2 Tensor Explorer uses three aligned surfaces:

1. **Main matrix** — the exact tensor surface.
2. **Row distributions** — a narrow panel immediately to the right, vertically aligned one-to-one with matrix rows.
3. **Column distributions** — a short panel immediately below, horizontally aligned one-to-one with matrix columns.

The current design uses a small gap between the main matrix and each distribution panel so the surfaces remain visually distinct while preserving obvious alignment.

For the current reference layout, the distribution depth is `100 px`:

- for a `rows × columns` tensor, the row-distribution panel is `100 × rows` pixels;
- the column-distribution panel is `columns × 100` pixels.

This means a `[576, 1536]` tensor produces:

- main matrix: `1536 × 576`;
- row distributions: `100 × 576`;
- column distributions: `1536 × 100`.

The `100 px` dimension corresponds to the current 100-bin visual design. It is a Tensor Explorer visualization parameter, not a tensor dimension.

### Row distributions

Each tensor row owns exactly one horizontal scanline in the right-hand panel. Along that scanline, the panel represents the distribution of values in that tensor row across the shared histogram bins.

The panel must never consume more than one screen row for one tensor row. It is therefore a density/profile image, not a conventional multi-pixel-high bar chart repeated for every row.

### Column distributions

Each tensor column owns exactly one vertical scanline in the bottom panel. Along that scanline, the panel represents the distribution of values in that tensor column across the shared histogram bins.

The panel must never consume more than one screen column for one tensor column.

### Histogram consistency

All row profiles in one tensor use the same bin boundaries, and all column profiles in that tensor use the same bin boundaries, so a cursor moving between rows/columns does not change the histogram coordinate system underneath the user.

The density/intensity normalization may be chosen to keep profiles readable, but it must be stable for the open tensor and must not silently change on hover.

The matrix must be usable before distribution/statistics results have completed. Distribution panels may progressively become available after the tensor itself; they must not block first render. The production and transport of these statistics remain owned by the backend/API specifications.

## Scroll synchronization

When the matrix exceeds the available UI area, the exact data surfaces remain aligned while normal page/panel scrolling is used:

- vertical scrolling of the main matrix keeps the right-hand row-distribution panel on the same rows;
- horizontal scrolling of the main matrix keeps the bottom column-distribution panel on the same columns;
- labels and floating hover UI may remain viewport-relative, but data coordinates must remain exact.

This synchronization is part of the Tensor Explorer layout; it is not a renderer resampling operation.

## Hover inspection

Hover is the primary fine-grained inspection interaction for a 2D tensor.

When the pointer is over a populated matrix pixel, Tensor Explorer resolves the exact tensor cell and exposes at least:

- row index;
- column index;
- logical tensor value.

The current visual treatment uses a compact monospace readout adjacent to the magnifier rather than permanently printing coordinates over the matrix.

Hover must work at native one-pixel resolution. No nearest-cell averaging or multi-pixel hit region may change which tensor value is reported.

If a progressively streamed region has not arrived yet, the UI must not invent a value for that cell.

### Cursor

Inside the matrix, the hover pointer is represented as a small warm-orange/amber cross rather than a large arrow cursor or filled marker. The cross should remain visually small enough that the underlying cell and immediate neighborhood remain legible; a small center gap is preferred to covering the selected pixel with an opaque mark.

The cross is an interaction marker only. It never changes the tensor value or its luminosity mapping.

## Magnifier

Hovering a matrix cell opens a small magnifier near the pointer. The current design is a rounded square card of roughly `170 × 170 px` containing a `9 × 9` neighborhood centered on the hovered cell.

The magnifier:

- enlarges source pixels with nearest-neighbor/pixel-preserving rendering; it does not interpolate between weights;
- preserves the same luminosity and semantic-color rules as the main matrix;
- visually identifies the center cell, for example with a thin warm-orange border/crosshair;
- is placed to the side of the pointer and should choose another side when necessary to avoid covering the inspected neighborhood or leaving the viewport;
- is an inspection aid, not general matrix zoom.

The proof-of-concept product rule that there is no general zoom/pan remains unchanged. The magnifier does not change matrix scale, scroll position, or the one-weight-to-one-pixel main surface.

## Selection encoding: luminosity is data, chroma is interaction

The common renderer already defines weight/scalar value through luminosity and reserves color for semantic information. Tensor Explorer makes that separation concrete for hover/selection.

Conceptually, the shader treats the displayed color as a luminance/chrominance representation:

- `Y` is the scalar-data luminosity produced by the common renderer transfer function;
- `U/V` (or an equivalent two-dimensional chroma representation) encode interaction state.

`Y` must remain unchanged when a cell, row, or column is highlighted.

The standard hover state uses a warm orange/amber chroma vector. `U` and `V` are used together to select the desired hue; the design must not assume that one chroma axis alone can express the required semantic color.

The intended behavior is:

- unselected pixels: neutral/default semantic chroma;
- hovered row: subtle selection chroma, original `Y` unchanged;
- hovered column: subtle selection chroma, original `Y` unchanged;
- hovered cell at the row/column intersection: stronger selection chroma, original `Y` unchanged;
- matching scanline in the row-distribution panel: the same selection semantics;
- matching scanline in the column-distribution panel: the same selection semantics.

The selection channel is full resolution per displayed tensor pixel — semantically equivalent to 4:4:4 chroma. Chroma subsampling such as 4:2:0 is not acceptable for cell selection because it would smear interaction state into neighboring weights.

`Y/U/V` here describe shader semantics, not a requirement to store tensor data in a YUV video/image format. The authoritative GPU tensor representation remains the scalar tensor representation defined by the common renderer. Selection color is computed at draw time.

If a chosen chroma would push the final RGB conversion outside the displayable gamut, reduce/clamp chroma before altering `Y`; preserving the scalar-derived luminosity takes priority over maximum saturation.

### No guide lines over tensor data

Row/column selection must not be implemented by painting opaque horizontal or vertical guide lines across the matrix or distribution data. Such lines overwrite exactly the information the explorer is intended to inspect.

If an additional geometric guide is ever needed for accessibility/contrast, it may appear only as short ticks protruding just outside the corresponding matrix/distribution edges. It must not cross the data surface.

The chroma highlight is the primary row/column linkage mechanism.

## GPU interaction-state constraint

Hover and selection must not require a recolored copy of the tensor or a second copy of the weights in GPU memory.

For the standard single-cell hover state, the renderer should be able to express selection with small interaction state such as the active row index, active column index, hover flag, and semantic-color parameters. The shader derives row/column/intersection chroma from that state while reading the same immutable scalar tensor representation.

A per-pixel selection texture/mask is unnecessary for the normal one-cell hover case and should not be introduced merely to recolor a row and column.

## Visual style

The preferred Tensor Explorer shell is light and editorial rather than terminal-like: warm/off-white page background, white data panels, graphite text, subtle borders and shadows, restrained typography, and generous empty space around labels.

Warm orange/amber is the current interaction accent. It is deliberately distinct from the data luminosity channel.

The UI chrome must stay secondary to the tensor. Avoid heavy panels, decorative legends, or saturated background areas that compete with the matrix.

The unselected tensor color treatment remains governed by [`rendering.md`](rendering.md); the green-tinted design explorations in Canva/Miro are visual experiments and do not override the common rule that scalar value is encoded by luminosity and semantic color is independent.

## Reusable Matrix Explorer primitive

The Matrix Inspector must be implemented as a reusable matrix/tensor exploration primitive rather than as a weight-only screen. The same aligned matrix, hover, magnifier, and linked-selection concepts are intended to be reused later for:

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
- exact matrix pixels as defined by the common renderer;
- aligned right-hand row distributions and bottom column distributions, currently 100 bins deep;
- synchronized scrolling that preserves row/column alignment;
- hover readout for the exact cell value and coordinates;
- small orange cross cursor;
- rounded-square pixel-preserving neighborhood magnifier, currently `9 × 9`;
- row/column/intersection selection through full-resolution chroma while preserving luminosity;
- linked selection state in both distribution panels;
- no guide line painted over tensor or histogram pixels;
- no recolored duplicate of the tensor in GPU memory;
- light, minimal, editor-like UI shell;
- no general zoom or pan in the proof of concept.

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

The matrix and both profiles share the common renderer's device-pixel convention
and snapped data origins. Distribution depth is 100 device pixels, with CSS gaps
for chrome. The matrix's native scroller drives the right profile's Y origin and
the lower profile's X origin; both profiles have the matrix's visible data extent
on their shared axis, excluding native scrollbars. Empty tensors show an explicit
empty state without starting tensor/statistics/distribution operations. Rank-1
uses the common strip with statistics and no distributions.
