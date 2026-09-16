# Tokenizer Explorer

## Status

**Dedicated UI specification accepted.**

This document owns the Tokenizer Explorer's UI behavior, layout, and presentation. Cross-cutting product scope, session/model ownership, tokenizer result semantics, and API schemas remain owned by their existing specifications and are not repeated here.

## Purpose and boundary

The Tokenizer Explorer is an editable text-inspection surface for the tokenizer associated with the current session model. Its purpose is to make the relationship between user-entered text and the tokenizer's actual output immediately visible.

The UI consumes the tokenizer result defined by the API contract. It does not reimplement tokenizer logic in the browser, and it does not attempt to reconstruct or teach BPE, SentencePiece, or another tokenizer algorithm step by step.

The prompt/tokenization surface defined here is a reusable UI component. The standalone Tokenizer Explorer uses it directly, and later inference-oriented views may embed the same component above downstream stages without creating a second tokenizer presentation.

Tokenizer Explorer also supports inspection of the model input embeddings corresponding to the latest successful token sequence, as defined in the accepted input-embedding extension below.

## Visual reference

The current wireframe is the `Inference Explorer` frame in the Miro dashboard:

https://miro.com/app/board/uXjVHnoDEYY=/?moveToWidget=3458764683523016635

The Miro board is a visual reference for the combined prompt/tokenization area. This document is normative if the board and specification diverge. Only the narrow input-embedding lookup described below is an accepted downstream extension; other inference content in the wider wireframe remains future scope.

## Live editing and tokenization

The explorer has one primary editable text surface. Tokenization updates in real time as the user types; there is no separate prompt view followed by a second token view that repeats the same text.

Each edit retokenizes the current text. Token boundaries are allowed to change anywhere in the text after an edit; the UI must not assume that previously rendered token boundaries remain valid for the edited text.

While a replacement tokenization is pending, the last successful tokenization remains visible as clearly stale/updating context instead of disappearing. The editable text is always the current user text, and stale annotations must not be presented as authoritative for that new text. When the matching current tokenization arrives, it replaces the stale presentation atomically, without an intermediate undecorated state.

Retain and move prior decorations through the editor transaction’s normal change mapping; do not clear them merely because the document changed. Stale state uses reduced annotation emphasis and an accessible status without changing source text or panel geometry. A failed recomputation keeps mapped previous annotations explicitly stale alongside the error; a later matching success replaces them atomically.

A result for an older input must never replace the presentation for a newer input if responses complete out of order. Only a result matching the current text, tokenizer options, session, and model may become authoritative.

## Inline token presentation

Ordinary tokens are presented inline as annotations of the prompt itself. The prompt text is not duplicated on another line merely to show tokenization.

For each ordinary token:

- the opening and closing token brackets are gray;
- the text between those brackets is black and represents the token's corresponding user-entered text;
- whitespace belonging to the token is preserved, so a token whose source span begins with a space visually includes that space inside its brackets;
- the token ID is shown in gray as secondary metadata visually associated with that same token, normally aligned beneath it.

Conceptually, a token is therefore rendered as a gray `[` + black source text + gray `]`, with its gray token ID associated with that span. The text itself must not be repeated in a separate token-only row.

When source offsets are available from the tokenizer result, they are used to associate tokens with the exact input substring. API offsets count Unicode code points and must be converted explicitly to JavaScript/DOM UTF-16 positions without normalization.

Overlapping source spans are grouped around their union, with every associated token ID retained in sequence order. The source character appears once, including when multiple byte-level tokens cover the same emoji. Do not split overlapping spans into invented disjoint boundaries.

When a reliable nonempty source span is unavailable, show the tokenizer-native/decoded representation as an explicitly unmapped annotation at its sequence position. This placement does not claim a character mapping. Per-token decoded strings need not concatenate to the input.

Annotations are separate from the editable value and never enter tokenizer requests or source clipboard text. Asynchronous decoration changes preserve native selection, caret, undo/redo, paste and IME composition. Retained stale annotations remain visually distinguishable from the current authoritative tokenization state.

## Special tokens

The API `special` flag identifies a special token ID, not whether its text was inserted. A source-typed special with a supplied source span remains black editable source, with gray brackets and metadata like any other source span.

Backend-inserted special tokens have no source span. Their textual representation and token ID are gray non-editable annotations, separate from the prompt value.

Special tokens are placed at the sequence position reported by the tokenizer, for example before or after ordinary prompt tokens when the tokenizer inserts beginning/end markers. They must remain visually distinguishable from black user-entered text without introducing a separate color scheme per token.

## Visual semantics

The Tokenizer Explorer intentionally uses a minimal monochrome distinction:

- black: user-entered text represented by ordinary token spans;
- gray: token brackets, token IDs, special tokens, and supporting tokenizer annotations.

Token identity is not encoded by assigning a different color to every token. The important visual relationship is the exact segmentation of the editable text and its associated metadata.

## Panel composition

Tokenizer Explorer composes two vertically stacked, visually distinct panels:
**Prompt / Tokens** wraps the existing editable annotation surface, and
**Input Embeddings** uses the same reusable `MatrixExplorer` as Tensor Explorer.
Each has its own full-width compact title bar above its own content card, using
the shared title/body composition from `visual-language.md`. The surrounding
workspace is a neutral layout container, with no third enclosing card. Empty,
loading, unavailable and failed embeddings retain the full-width title/body.
Wrapping the prompt must preserve its inline text/bracket/ID layout and native editor behavior; it must not
introduce another token strip or prompt camera.

The workspace is bounded by the application shell. By default, Prompt / Tokens
sizes to the editor's actual content, including wrapped lines, with a comfortable
minimum and a maximum derived from the available workspace height. It grows as
the prompt grows, then scrolls internally at the maximum. Browser resizing
recomputes the automatic allocation. Input Embeddings fills all remaining height,
including for short token sequences; underfilled matrix content follows the
shared Matrix Explorer's geometry rather than a tokenizer-specific camera.

A focusable horizontal divider is the primary panel-height resize control. Its
12 CSS-pixel hit area occupies the natural separation between the two cards;
the resting state adds no standalone rule or thick handle. Hover, keyboard focus
and active dragging highlight the adjacent card boundary, retaining discoverability.
Pointer dragging adjusts the split continuously. Up/Down change the prompt height
by 16 CSS pixels; Home/End reach the allowed bounds. Expose orientation, the
controlled panel, current height and bounds accessibly, with a resize cursor and
visible keyboard focus. Double-click or two Enter/Space activations within
500 ms reset to automatic sizing for the current content and workspace.

Manual sizing overrides content-based allocation for the mounted workspace only.
Keep the preferred manual height on browser resize, clamping its displayed height
to the new bounds; returning to a larger workspace restores that preference.
Do not store the split in backend or session state. The editor fills the allocated
panel without a competing native resize handle. Reserve the token count/status
row in every generation state and avoid redundant headings or inter-panel space.

The sizing baseline is a 160 CSS pixel Prompt / Tokens minimum, a 180 CSS pixel
Input Embeddings minimum and a 12 CSS pixel divider. Automatic prompt sizing is
capped at 45% of the height left after the divider, within the panel bounds. At
workspace heights too short for both minima, relax them proportionally so the
panels remain bounded and the document never scrolls. These are panel dimensions;
they never scale scientific matrix cells or text. Retokenization status and matrix
scrolling, zoom or selection must not change the split; actual edited or wrapped
content may change the automatic prompt height.

The Input Embeddings header uses the shared Matrix Explorer header composition
for matrix identity, available shape and logical dtype, operation status, and
viewport controls. Do not repeat row/hidden-dimension counts as a separate block
between panels. The matrix inherits the shared square-cell camera, fit-width,
zoom history,
region/range navigation, direct inspection, adaptive magnifier, fixed right/bottom
distribution tracks and non-consuming overlay scrollbars. Short matrices remain
thin strips with aligned data and blank margins, never stretched or transposed.
Do not compute missing distributions or fork matrix functionality
inside the tokenizer component.

## Accepted input-embedding extension

After the latest successful tokenization, Tokenizer Explorer may inspect the session model's input embedding matrix for that exact token sequence. For `N` tokens, the matrix has shape `[N, hidden_size]`; row `i` corresponds to token sequence position `i`, including inserted special tokens and repeated token IDs.

The prompt/tokenization surface and the input-embedding matrix are independent inspection surfaces. Matrix zoom, scrolling, fit, region selection, or other matrix navigation affects only the embeddings view and must not zoom, reflow, recenter, or otherwise change the prompt editor viewport.

The embeddings view exposes the same matrix-inspection and navigation behavior defined for the reusable Matrix Explorer, including exact row/column/value inspection and the accepted zoom/navigation behavior. Token inspection and embedding-matrix inspection are linked by sequence position: identifying token position `i` identifies embedding row `i`, and identifying an embedding row identifies the corresponding token position. This linkage must not change the editable prompt content or the tokenizer result.

Hover or focus on an existing token annotation transiently highlights its row;
matrix hover, focus or cell selection identifies the corresponding annotation.
Transient state clears on leave/blur and remains distinct from any pinned/current
selection provided by a consumer. Link by sequence position, including repeated
IDs, overlapping source spans and inserted specials. Explicit token activation
may minimally scroll an offscreen embedding row into view, preserving matrix
scale and horizontal origin. Hover alone does not reveal rows. Matrix row
interaction must never automatically scroll, zoom, or focus the Prompt panel.
Use controlled row context, semantic selection callbacks and a narrow reveal
intent through Matrix Explorer, without exposing renderer internals. Clear or
fence linkage on source, generation, session replacement and remount.

Native source text selection persistently selects complete embedding rows by
zero-based token sequence position, separately from transient inspection. For
each nonempty editor range, include every token whose reliable nonempty source
span strictly intersects it, converting API code-point offsets to UTF-16 without
normalization. Partial tokens select complete rows; overlapping byte-token spans
include every associated position. Repeated IDs remain distinct positions.
Union multiple native ranges without filling gaps. Boundary contact alone does
not intersect. Whitespace, newlines and source-typed specials follow the same
rule; inserted/unmapped annotations have no fabricated source span and retain
their separate explicit inspection/activation.

Pointer or keyboard selection in either direction replaces this group; an
intentional collapsed selection/caret placement clears it. Pointer departure,
annotation blur, loss of editor focus and matrix inspection do not clear or
replace the group. Selection-only changes never retokenize. This path never
reveals rows, scrolls either surface, zooms, fits, recenters, changes camera
history or resizes the panel split. Offscreen rows stay offscreen until manual
navigation. Matrix interactions never focus, select or scroll the prompt.

Use the shared Matrix Explorer's controlled row guides, including aligned row
distribution projection, independently from exact cell inspection and the
transient drag-to-zoom preview. Clear active guides immediately when text,
options, session/model or authoritative generation changes. A retained stale
matrix and a hidden staging matrix never receive current source linkage. When
the matching visible pair becomes authoritative, derive rows anew from the
still-current source selection; never replay a previous generation's row list.

While a replacement tokenization or embedding result is pending, the last successful embedding matrix remains mounted and visible as clearly stale/updating context rather than disappearing. Only an embedding result matching the current tokenization generation, session, model, options, and ordered token IDs may become authoritative. Older results must never replace newer state. A replacement is promoted atomically after successful complete delivery; initial embedding delivery remains progressive. Failure, cancellation, empty sequences, or unavailability retain the previous matrix only as explicitly stale context alongside the current status. Pending and failed replacement allocations are disposed, and successful replacement or workspace disposal releases superseded Matrix Explorer resources. Updating/error status stays in the existing header slot.

Token-to-row linkage is active only when the visible tokenization and visible embedding matrix belong to the same authoritative generation. If a newer tokenization is already current while an older embedding matrix remains visible as stale context, they must not be linked as if their row identities still matched.

Empty token sequences and models for which input-embedding lookup is unavailable are explicit states, and tokenization remains usable when embedding lookup is unavailable or fails. Failure of a new embedding lookup does not roll back or invalidate a newer successful tokenization. The graceful unavailable message is reserved for `unsupported_representation`; transport, protocol, and rendering failures remain distinct failed states. Architecture and quantization decisions belong to the backend input-table resolver, never browser-side family gates.

The input-embedding extension does not add positional encoding, transformer execution, logits, generation, or another later inference stage.

### Derived scientific data and lifetime

Use the independent typed input-embedding statistics and distribution operations
specified in [the API contract](../api/contract.md#input-embedding-analysis) for
exactly the current ordered token IDs. Reserve the shared viewer's histogram
tracks before allocating its source. Forward the authoritative domain/count
sections through `MatrixUpdates`; completed statistics use the same transfer
behavior as Tensor Explorer. No browser histograms, full-vocabulary fallback,
second scalar payload or invented checkpoint identity/storage dtype is permitted.

The information action uses the shared preview/pinned popover and displays the
visible derived matrix's shape, logical dtype, element count, completed statistics
and available full-range bin domain. Keep luminosity percentile anchors separate
from histogram endpoints. Missing, incomplete or failed analysis stays explicit.
Each operation has compact independent status and cancellation in the title bar;
auxiliary failure or cancellation preserves usable values and current tokenization.
Empty token sequences do not start lookup or analysis operations. Start analysis
once value metadata confirms a supported input table, without waiting for values
to complete; unavailable tables do not trigger speculative analysis requests.

All three streams share the tokenizer generation, session/model/options and exact
ordered IDs. Late results cannot cross generations even when shapes match.
Early auxiliary counts may use one generation-bounded staging buffer, released
on subscription, replacement or disposal; scalar chunks are consumed directly.
The old visible matrix retains only its own scientific data while explicitly
stale. Promote a replacement when its values complete, independently of auxiliary
completion; its pending analysis never borrows prior counts/domain. Keep at most
one visible and one staging matrix, and release all superseded consumers and
scalar/count resources. Header changes, analysis arrival and splitter movement
do not remount the editor/viewer, restart transport or change the split.

## Reuse in later inference views

Later inference exploration may place the tokenizer component directly above embeddings or other model stages. That composition must reuse this same live prompt/tokenization component rather than introducing a parallel tokenizer UI with different interaction or visual semantics.

Beyond the accepted input-embedding row lookup above, this document does not define downstream computation stages. Tensor rendering rules remain in `rendering.md`; positional encoding, transformer execution, logits, and generation require future design.
