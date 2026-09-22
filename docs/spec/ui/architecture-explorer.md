# Architecture Explorer

## Purpose and integration

Architecture Explorer is an additional static model view inside the existing bounded application shell. It does not replace Tensor Explorer, Tokenizer Explorer, or Matrix Explorer. [Product scope](../product.md#architecture-reference-checkpoints-and-bounded-coverage) defines the selected model coverage; [the API contract](../api/architecture-explorer.md) provides the prepared graph and resource references.

Fetch the prepared architecture for the selected session when its view is needed. This is retrieval, not generation. Display available complete/partial coverage or a model-local unavailable reason; the server has already finished preparation before accepting connections. A restart-required result must explain that restarting prepares new or changed models. Do not add a regenerate button, analysis polling, metadata-only onboarding, or a text prompt needed to see the graph.

A failure affects only this capability. Preserve all existing supported exploration workflows, matrix camera behavior, prompt editing, embedding linkage, streaming, and cancellation. Show explicit partial-coverage diagnostics when supplied by the inventory; a fully actionable quantized inventory may report complete.

Architecture-specific navigation, layout, routing, legends, repetition controls, and connection inspection stay inside the existing Architecture Explorer workspace. They must not resize, restyle, or relocate the shared application bar, explorer navigation, model selector, session controls, page background, bottom status bar, Tensor Explorer, Tokenizer Explorer, or shared Matrix Explorer primitives.

## One global navigable diagram

Use a single pannable, zoomable canvas with fit-to-window and centering on a selected component. Hierarchy organizes nested groups on the same canvas; it must not require separate level pages, drill-down routes, or stacked dialogs to understand structure.

An optional isolated component may be the visible root of this same canvas, as
defined below. The complete model remains the global fallback.

Start compact, using declared repetition records to represent repeated layers as a stack with multiplicity and truthful variant information instead of immediately materializing every instance. Allow expansion of a representative interior, selection of a concrete instance without expanding its siblings, focused exploration of a bounded contiguous set of instances when useful, expansion of chosen instances, and an Expand all action that reveals every instance and required mathematical operation. Compressed before/after ranges must remain explicit when only part of a stack is shown. A representative display must clearly identify its selected instance; inspecting a weight must never silently use layer zero for every repetition.

A fresh backend/model/graph view opens only the outer model boundary and its
immediate visible components. This is the hierarchy **after** declarative
interfaces become ports and any presentation-only Model boundary is inserted;
all expandable child interiors, concrete instances and derived groups stay
collapsed. Passive interfaces do not count as a component level. Assess this
candidate's visible layout bounds against the current nonzero graph-body
viewport. The minimum readable initial scale is **0.8**, with a **16 CSS-pixel
inset** and a maximum initial scale of **1**. If fitting the candidate requires a
smaller scale, use the collapsed outer model with its name, declared ports and
expansion action. Compute at most one candidate and one fallback; never display
exhaustive detail and then snap closed. Checkpoint size, model names and hidden
source-node counts do not determine this choice. An unusually large collapsed
interface remains readable at the minimum scale and accessible through pan/zoom.

Expanded containers reserve their own header/summary, visible descendants and
necessary port/routing gutters. Boundary port rows occupy side gutters rather
than an additional empty band above the children. Hidden descendants reserve no
space. Real labels, hit targets, fan-out and crossing routes retain their room.

Keep actual layer order and variant differences visible when grouped. Repetition means repeated structure, not shared weights, states, parameters, or guaranteed equivalence between variants. Preserve external dependencies across collapsed boundaries through meaningful ports/connections, including bypasses that cross hidden ranges. Collapsing must not erase a skip connection, create a false serial path, or treat distinct hybrid blocks as identical. Expanded topology must be recoverable from the same graph without new analysis.

Expanding a group preserves the acted-on location/context; do not reset the camera or automatically fit the entire graph after every expansion. Maintain camera, expanded groups, focused repetition/instance context, and valid node/connection selection while closing inspection or switching explorers during the browser session, keyed by model and graph identity. Server-restart persistence is not required. Reject late responses and clear invalid selections on session/model/graph replacement.

Initial and scope camera work uses the current layout and the renderer's current,
nonzero viewport size. Loading presentation must not displace that viewport.
Readiness includes completion of the applicable camera action, not only layout
calculation. Saved camera restoration, Back and newer scope/model choices take
precedence over obsolete initialization; user camera interaction supersedes a
pending initialization. Ordinary resizing, selection, hover and menus do not
start another fit or layout. Initialization failure is bounded and recoverable.
Fresh initialization aligns the top of visible diagram bounds to the small
graph-body inset and centers horizontally when the diagram fits. The renderer's
actual body excludes the card header and any diagnostic band. Saved camera and
expansion, ordinary resize and explorer/inspection reopening do not reapply this
initial policy. Manual Fit view continues to fit all currently visible detail.

The explicit global **Collapse all / Collapse model** command is the exception
to the no-automatic-fit rule: it exits isolation, collapses to the outer model,
waits for that layout and fits those final bounds once, with maximum zoom 1.
It retains valid source selection even when hidden. A newer scope, model,
expansion or user camera command supersedes pending collapse framing. Ordinary
card/browser plus/minus actions continue to preserve their anchor and never fit
the whole graph.

Viewport culling and asynchronous layout are allowed optimizations; discarding graph records or silently reducing detail is not. Bound layout work and provide an explicit recoverable failure rather than an indefinitely frozen canvas. All reference graphs must be usable when fully expanded on the documented acceptance environment. Record actual layout time, memory, and graph size rather than inventing a performance guarantee.

React Flow and ELK are implementation candidates, not mandatory backend dependencies or reasons to change the graph contract. The layout implementation may be replaced without changing model semantics. Do not add a second matrix renderer, graph editor, export suite, or complex navigation framework in this increment.

## Browser and contextual navigation

A separate left browser card provides navigation beside the canvas card. It uses
ordered source containment, declared repetitions and optional verified template
families from the received graph. Never infer structure from tensor filenames,
sort away source order, or create a second semantic graph. Components with
children use folder/component icons, terminal components use block icons, and
Shared families use overlapping blocks (repeated structure, not weight tying).

The browser header contains a collapse arrow, followed by local search, a
**Model** hierarchy and a **Shared** family list. Disclosure and selectable names
are separate controls. Indentation is capped at four levels; full public names,
paths, exact IDs and concrete instance context remain available on hover/focus.
The browser owns its vertical scrolling; its header and search stay reachable.

Use the Tensor inventory's pane conventions without changing that explorer:
280 CSS-pixel preferred width, bounded to 200–480 and available width minus the
16-pixel divider and 360-pixel canvas minimum. The accessible vertical divider
supports pointer capture, Left/Right in 16-pixel steps, and Home/End. Temporary
viewport constraints do not overwrite preferred width. At 760 pixels or narrower,
the expanded browser occupies a bounded upper row (28%, at least 130 pixels),
with the canvas below and no width divider. Explicit collapse removes the pane,
divider and gap in both layouts, leaving a 40-pixel icon-only restore rail beside
a full-height canvas. Neither layout causes document scrolling or scales graph
labels down. Restore retains width, query, scroll, selection and expansion.
Store only pane preferences in optional local storage with an in-memory fallback;
graph-specific navigation remains scoped to backend/model/graph view lifetime.
Collapse and resize never remount the canvas, refetch, restart numeric consumers
or invoke Fit view.

Component names select the exact source component on both surfaces, including
nonzero repeated instances. Selection does not expand, inspect, reveal, change
scope or camera. A component's **+ / −** and name double-click use the same
expansion controller as its canvas card. Leaves have no disclosure or double-click
action. Keyboard disclosure uses that controller too. Contracting a parent
retains hidden descendants' expansion and selection; unrelated siblings remain
unchanged. No toggle implicitly Fits view. Only explicit **Center** reveals a
hidden selection by opening required ancestors and its exact repetition window,
without opening its own children or every sibling.

Search filters eligible components from the full received graph by real label, public containment/module
path, exact identity and supported description source key. It includes collapsed
interiors and verified families/instances, grouped under Model and Shared with
paths and stack/instance context to distinguish equal names. Results are a flat
filtered list, not automatically opened real tree branches. Typing changes no
selection, expansion, scope, camera, layout, retrieval or numeric work. Clicking
or Enter selects only; Center and navigation remain explicit. Clearing search
restores the ordinary tree and its scroll, retaining expansion changes made
while filtering. Clearing selection does not clear search. Canvas selection never
clears a hiding filter or opens the collapsed browser. Arrow keys navigate rows;
Escape from a row returns focus to search, and Escape in search clears the query.

A Shared family shows its supplied label and actual count. Its disclosure opens
only an ordered presentation list, including nonconsecutive declared indices.
Concrete rows share source selection and expansion with their Model occurrence.
Selecting a family offers **Explore structure**, which enters the existing
structure-only view without choosing instance zero or permitting weights. A common
canvas selection in that mode is not a selected concrete Model/Shared instance;
explicit browser source selection retains its own exact identity without choosing
a weight-bearing instance or silently changing scope.
Absent templates show an honest empty state and preserve ordinary navigation.
Verified instance switching, correspondence, cancellation and exact parameter
binding retain their established rules. Stale source/family IDs are invalidated
with their graph. Shared scope expansion maps concrete IDs through verified roles
to the existing anchor state, never through independent browser flags.

The minimal canvas header has current Model/component context on the left,
selected node/connection with applicable **Inspect**, **Explore component** or
**View in model**, **Center** and clear controls toward the right, and **Fit view**
and **View options** at the far right. Omit inapplicable selected-item controls.
No global horizontal component strip, Find component trigger or Shared structures
dropdown remains. Preserve explicit isolation/shared context, Back and compact
concrete instance/previous/next/window controls. Repetition entry controls live in
the browser. Navigation focus and selection remain distinct.

View options provides exhaustive **Show all operations**, collapse, center
selection, zoom, dimensions (initially off), context, unused-interface and derived
MLP preferences, plus applicable group/layer/MLP/state navigation. Fit view changes
the camera; Show all operations changes visible detail. Unused interfaces refers
to unconsumed interface branches, not every auxiliary signal. State focus identifies
its filtered context; partial coverage stays visible and full diagnostics remain
available on demand. Options dismiss on Escape with focus restored to the trigger
or when focus moves outside. Controls use neutral/graphite/amber styling, English
accessible names and visible focus, with bounded overflow at narrow widths.

Selection, typing, menus, emphasis and inspection preserve the mounted canvas,
graph retrieval and numeric lifetimes, source/projection records, generated
coordinates/routes, layout invocation count and camera. Explicit reveal,
expansion, presentation preferences and camera commands retain their effects.

## Source-preserving visible projection

Keep the received graph unchanged. Visible presentation records retain source node IDs, repetition/instance identity, and exact original edge/port references. Compose boundary forwarding without traversing computational operations. Model, stack, layer, MLP and state focus are expansion/filter states of the same canvas, with an explicit return to global context. Stack window size depends on the view/context rather than a globally fixed layer count.

A reversible derived MLP group requires matching parameter/module ownership and exact gate/up/SiLU/multiply/down topology. Mark it as derived; unmatched operations remain explicit. A presentation boundary alias is not a new mathematical operation or backend identity.

Prefer received Attention/MLP groups identified by the API's optional
`semantic_role` attribute. Keep their source-backed labels and original node
identity for expansion, inspection, search and component focus. Do not add a
derived MLP container inside an explicit MLP group. The existing bounded fallback
remains reversible for valid legacy graphs without explicit MLP ownership; it
does not recognize additional attention patterns or reconstruct missing math.
Search includes description source keys so concise labels retain access by the
original operation path.

Declared interfaces remain inspectable when unconsumed branches are filtered. Determine consumption from source connectivity, with an explicit reversible filter. State focus shows existing prior/next dependencies owned by the selected instance, keeps K/V distinct and identifies excluded flows. Exhaustive expansion restores original operations and interfaces without a representative or layer-count limit. Interfaces converted to boundary ports and excluded tool capabilities never return as cards or component rows.

## Declarative interfaces on container boundaries

Distinguish passive interface values from computations with one source-backed
presentation index shared by projection, browser/search and inspection. An input
leaf is convertible when it has output ports only, no incoming edges, parameters,
resource references or formula, and only data edges. An output leaf is convertible
when it has one receiving input, at most one incoming data edge, no outgoing edge,
parameters, resource references or formula, and zero or one unused output
descriptor. The latter includes dense logits declarations: retain that unused
port in inspection without inventing a forwarding operation or duplicate output.
Optional operation identifiers, roles and descriptive attributes do not alone
make a declaration computational. Actual operations/groups and state/context
records remain components, regardless of names such as positions, mask or logits.
Unknown shapes stay unknown. Ambiguous records remain visible and inspectable
with an explicit presentation notice; never silently discard their evidence.

A converted declaration belongs to its source parent, or to the model boundary
when it is a source root. Reuse the sole remaining source root group only when
it covers all retained components after excluding declarations and tool-only
leaves. Otherwise show a marked presentation-only **Model** container around
retained roots in source order. It represents the graph's described scope and
coverage, not a claim of complete multimodal/checkpoint support. Its identity is
not a source ID or module path and has no parameter binding. Language model and
an external LM head stay separate children; visual encoder/predictor models acquire
no text interfaces. Use the browser's Model heading for the presentation boundary,
without a fictional module or an Inputs/Outputs subtree. Real containment,
repetition, concrete instances and parameter identities are unchanged.

Inputs sit on the left; terminal outputs on the right. Names come from declaration
and port descriptors, shapes from exact endpoints. Reuse an existing container
port only through verified original edges and same-endpoint group forwarding;
never merge different signals by label/shape or pass through a computation. New
boundary ports are presentation aliases retaining original endpoints and full
node/edge provenance. Keep expanded boundary anchors and internal producer/consumer
routes, and collapsed named ports with their hidden path metadata. A boundary
input may source an internal route and a boundary output may receive one. Absorbed
forwarding retains source paths as port evidence. Preserve directed path
multiplicity, fan-out, residual bypasses, different masks and separate K/V states.
Unconnected interfaces stay named and inspectable without fabricated connections.
Unused-branch filtering does not remove converted port names or metadata access.

Isolated components use their exact declared interface (or an operation's actual
ports). Passive external declarations resolve there without duplicate external
cards. Genuine external computational/state/context dependencies retain compact
context aliases. Exit/re-entry remains external and wholly external bypasses stay
excluded. Verified shared-instance correspondence also rebinds interface metadata;
structure-only views never invent a concrete instance or offer weights.

Converted declarations never appear as independent Model, Shared or component
search rows. Interface-name/ID/source-key queries return the owner once, annotated
with matched ports. Selection records a typed owner/port target and changes no
camera, expansion, scope, layout or retrieval; explicit Center retains its normal
reveal behavior. Ports are not expandable children. Normalize saved declaration
selection to owner/port, clear removed tool selections with an explanation and
invalidate stale port targets on graph replacement. Search clearing, scope/Back,
explorer/pane switches and exhaustive display cannot resurrect removed entries.

Hover/focus follows exact transparent connection chains and stops at operations;
activation pins the port connection set for explicit inspection. Unconnected ports
remain selectable. The single existing modal exposes original declarations,
complete port descriptors, attributes, diagnostics and connection provenance;
container inspection includes its converted interfaces. Source-free presentation
boundaries identify themselves truthfully. No interface action starts numerical
work. Keep live focus restoration, camera and resource lifetime guards.

Reserve side rows/gutters for names and hit targets; grow containers vertically
for large interfaces and use existing graph pan/zoom. Long labels have full
hover/focus disclosure. Show dimensions controls shape annotations, never names
or connections. Interaction does not move geometry. Deterministic conservation
checks and desktop/constrained-width browser checks must verify these properties
alongside surviving computation, source identity and independent graph coverage.

## Optional isolated component exploration

**Explore component** is an explicit action on a selected source group,
operation, or supported derived MLP. Selection, inspection and expansion in
place retain their existing actions. Isolation shows only that component and
its chosen internal detail; hidden ancestors, siblings and unrelated connections
contribute no layout boxes or bounds. It is a view over the same immutable
received graph, not a new analysis, route, page or modal stack.

Every real scope-crossing connection retains compact input/output context with
exact original node, port and edge provenance. These are presentation aliases,
not additional operations or backend ports. Genuine fan-out shares only its real
source signal; equal labels or shapes cannot merge externally distinct signals.
Mask, position, residual and state roles remain exact. A wholly external bypass
is excluded. A dependency that exits and re-enters remains visibly external;
never splice it into a fabricated internal path. Existing boundary forwarding
and port/branch/shared-trunk hit semantics remain in use and stop at operations.
Hover, focus and pinning do not expand hidden nodes or invoke layout.

Identify isolation visibly and show breadcrumbs for the concrete component's
real model/stack/instance location. Nested isolation is bounded navigation history
on one canvas. **Back** restores the preceding scope's camera, expansion,
repetition window, filters and valid node/connection selection. **View in model**
returns to the global graph and intentionally reveals and centers the selected
concrete component. Search and the browser cover all eligible components; declarative interface queries resolve to their owning boundary ports. Selecting outside the current
scope records exact identity without leaving isolation; explicit View in model
or Explore component remains available. Center or expansion of an outside-scope
component deliberately returns to Model before acting and retains the isolated
snapshot for Back. Context ancestors may appear in the browser as Model context;
they are not represented as computational nodes inside the isolated canvas.
Within scope, both surfaces use that scope's common expansion. Back restores its
prior state. Breadcrumb navigation outside a scope restores global context. Keep
view snapshots local, bounded and keyed by backend/model/graph identity without
copying the semantic graph per entry. Reject obsolete layout/retrieval results
on session/model/graph replacement.

Entering a new scope may compute layout and establish a readable initial camera;
subsequent selection, menus and emphasis may not. **Fit view** in isolation
measures the visible component and its context only. **Expand component** expands
only the current scope. The exhaustive command is explicitly labelled
**Show all operations in model** and exits isolation before revealing every
source instance/operation. Existing unused-interface and context filters remain
explicit and reversible; complete interfaces and excluded source context remain
inspectable. On preparation/layout failure keep the source and a recoverable
previous/global view, with retry. Invalid graphs retain normal validation errors.

With isolation inactive, unchanged graph/view inputs retain the prior projection,
layout inputs, route identities and interaction results. Keep scope membership and
reversible navigation at small testable seams; use the existing cancellable
layout worker and connection algorithms.

## Optional verified shared structures

A received template annotation offers **Explore structure** from its selected
concrete component members and from families in the browser’s **Shared** list. This remains an optional isolated view on the same canvas.
Ordinary overview, exhaustive access, isolation, routing and numerical inspection
remain independent of template metadata. No frontend pattern matching establishes
equivalence; the API's validated exact role mappings are authoritative.

Entry from a selected component retains its concrete instance and selected
operation. Entry from a selected browser family is explicitly **structure only**:
show the common verified topology, shapes and attributes with no weight-bearing
instance selected and no parameter action. The shared canvas covers the declared
interior and explicit boundary forwarding; external producers/consumers remain
concrete instance context. With an instance selected, inspection exposes its exact
interface connections and original source identities.

The instance selector shows actual stack/layer location separately from position
within the verified family. Previous/next traverses only that ordered family,
including nonconsecutive source indices. Switching rebinds source node, port,
edge and parameter references through the exact correspondence, preserves selected
operation or pinned connection, and reuses compatible geometry and camera without
new layout work. Shapes, operation attributes and parameters come from the chosen
source records; repeated structure never means shared or substituted weights.
Close active inspection using its existing cancellation and lifetime guards so
obsolete numerical responses cannot populate the replacement instance.

**View in model** requires a concrete instance and reveals that component;
**Back** restores the preceding shared context. View history retains bounded
navigation state, not graph/geometry copies. A graph replacement clears stale
shared correspondence with an explanation. A shared-view preparation/layout
failure offers recovery to ordinary exploration, which remains fully available.
Use the existing renderer, cancellable layout worker and connection-hit behavior.

## Automatic layout and directional routing

The graph communicates computational direction primarily from left to right. Apply that ordering recursively at every expanded scope: the model, repeated stacks, concrete layers, attention blocks, and other expanded groups. Successive stages in a genuine serial dependency chain advance horizontally in their actual dependency order; a serial set of layers or operations must not degrade into a vertical list merely because it is nested inside a group.

Use vertical separation for genuine parallel branches, auxiliary inputs, residual bypasses, independent state paths, and independent components. Parallel Q/K/V, gate/up, or similar branches may occupy separate rows when the source topology supports them. The layout must not derive semantic order from labels, path names, or alphabetical sorting when graph dependencies provide the real order.

Expanded group geometry is determined by the visible children, their labels, ports, and connection clearances. Collapsed descendants and their internal routes do not contribute to visible layout bounds. When a horizontal computation is wider than the Architecture panel, preserve computational order and use graph pan/zoom and compact repetition navigation rather than wrapping the serial chain vertically, shrinking labels until unreadable, resizing the application shell, or creating document-level scrolling.

Connection routing must keep distinct signals visually distinguishable. Routes should avoid unrelated node bodies and labels, preserve exact source and destination ports, and use clear destination arrowheads. A shared trunk is valid only for genuine fan-out from the same source signal/port; unrelated signals that happen to share a label, shape, or corridor must remain individually traceable. Group-boundary forwarding may be composed into one visible route, but must retain the original edge and port identities represented by that route.

Changing hover, keyboard focus, or pinned connection selection must not move nodes, change port positions, refit the camera, or recompute layout. A layout failure is recoverable and explicit; it must not silently fall back to an unreadable vertical stacking mode.

## Clean default presentation and dimensions

Cards retain their title and independent header controls, followed by one muted,
source-backed formula or operation signature when supplied. Formula text is
inert display text; the UI never infers equations or adds missing bias. Existing
input/output labels (including `out`), directions, handles and port identities
remain visible. A card then shows only its own explicitly referenced tensors,
deduplicated by exact parameter identity across `parameter_ids` and parameter
references. Groups do not aggregate descendant parameters.

Tensor rows place a subtle matrix button to the left of the real name, including
rank-1 biases. A single explicit owning-module reference may establish a removable
prefix; all remaining path segments stay verbatim. Ambiguous or absent ownership
keeps the original name. Full names remain available on hover/focus and in
accessible identity. Relevant computational scalar attributes follow the tensors
with their original names and actual values and no matrix button. Use a small
operation-specific mapping, never a configuration dump or invented defaults;
missing/unknown values stay missing/unknown. Semantic roles, provenance and
diagnostics are not constants.

**Show dimensions** is initially off and consistently controls tensor-row,
input/output and connection shape annotations. Shapes are API logical shapes,
including constants, symbols, display-only expressions and unknowns; never packed
geometry, flattened ranks or evaluated expressions. Names, constants, values and
controls remain visible with dimensions off. Full inspection descriptors remain
independent of this preference.

Keep content within the same rounded card, without Inputs/Parameters headings,
repeated type pills, Greek aliases or a permanent metadata panel. Size each card
for its own content and ports; expanded group headers reserve space above children.
Long text uses bounded width and full-text hover/focus disclosure without moving
geometry. Long own-parameter lists may use a truthful `+N more` action opening
the ordinary inspector. Verbose descriptions, configuration, provenance and
diagnostics remain in inspection. Coverage and unknown regions remain visible
enough to avoid interpreting a partial diagram as complete.

Use the shared [visual language](visual-language.md). Architecture nodes, groups, controls, ports, and connections use the existing neutral surfaces, graphite text, hairline borders, restrained shadows, and warm amber interaction accent rather than introducing a new application theme or per-layer categorical palette. Graph zoom is its own camera. It must never zoom the prompt editor or impose a transform on a Matrix Explorer surface.

## Connection hover, focus, and selection

Connections are inspectable from both their lines and their ports. Hovering or keyboard-focusing a visible connection emphasizes its complete visible route from the actual source port to the actual destination port, including composed group-boundary segments, endpoint markers/labels, and destination arrowhead. Hovering an exact input port emphasizes the incoming connection or connections terminating at that port and their sources. Hovering an exact output port emphasizes all outgoing connections from that port and their destinations; genuine fan-out remains visible as fan-out, while each branch stays individually targetable from its own line.

Resolve these interactions from exact port identity and source-edge mappings. Following a boundary pass-through keeps one represented connection visually continuous across scopes, but interaction stops at computational operations: hovering an input does not recursively highlight every downstream descendant. Hidden nodes are not automatically expanded and no fabricated internal path is revealed. A collapsed group shows the visible external connection it represents while retaining traceability to the underlying source edges for inspection.

Connection and port hit areas may be larger than their visible strokes/markers so thin routes remain practical to target at normal graph zoom. Ambiguous crossings must still leave distinct connections targetable through an unambiguous segment or endpoint. Port interaction must not initiate graph editing or accidentally trigger an unrelated node action.

Hover/focus emphasis is temporary and uses the existing warm amber interaction semantics together with non-color cues such as stroke/marker emphasis and directional arrowheads. Nonparticipating connections may be visually subdued but remain readable. Click/keyboard activation may pin a connection for inspection; leaving a hover or focus restores the pinned selection rather than clearing it. Moving between a port and its connected line should not produce a flickering loss of context.

## Simple modal inspection and resource navigation

A single mouse click on a card label or ordinary body selects only, including
presentation-backed cards. It does not inspect, expand, navigate, change focus,
invoke layout/retrieval, or move the camera. Double-click toggles an expandable
card once, using the same current state and action as `+`/`−`: expand when
collapsed, contract when expanded. A leaf or empty group has no expansion
control; double-click has no additional effect beyond the preceding selection
clicks. Supported derived groups and repetition ranges use their actual contents
and presentation identity, never an arbitrary substituted layer. Inspection is
always explicit. Card double-clicks
must not zoom the canvas. Background pan/zoom remains unchanged.

Independent header controls are ordered `+/-` (when expandable), navigation,
then `(i)`. The navigation control shows four outward diagonal arrows for
**Explore component** for valid components in Model view and descendants during
isolation. Only the active isolated root shows four inward diagonal arrows for
**View in model**. Resolve the action and target together from the pressed card's
exact source or supported derived-MLP identity, independently of prior selection.
Exploring a descendant pushes the preceding scope snapshot and makes that child
the new isolated root; its descendants remain explorable. **Back** restores each
preceding scope without collapsing it. The root's **View in model** reveals,
selects and centers that exact root in the global model. Synthetic repetition
ranges, context/boundary aliases and other cards without an existing target show an unavailable control with an
accessible explanation. Structure-only shared views require a concrete instance
before navigating; concrete shared cards use the current verified source mapping,
not the template representative. Controls have English target-specific names and
tooltips, visible focus and independent click/double-click hit behavior. Ports,
connections and expanded-group children retain their own targets.

The canvas and browser consume one authoritative component selection
and expansion state per current model/graph/scope view, with changes from either
representation immediately reflected by both. Contracting a parent hides its
descendants but retains their nested expansion choices and unrelated sibling
state. A valid selected source descendant stays selected even while hidden;
hiding it does not select its parent or a synthetic range. Graph/model replacement
still clears invalid identities. Reversible scope snapshots retain the previous
scope's state without copying the semantic graph. Explicit toggles preserve the
acted-on location through anchored layout and never implicitly fit the model.

Explicit `(i)` and keyboard inspection open one closable modal with identity,
function, known dimensions, provenance/diagnostics, and parameter choices.
Keyboard selection, ArrowLeft/ArrowRight expansion, and explicit inspection and
navigation controls remain available without double-clicking. Group collapse
remains available through `−` and explicit keyboard/toolbar controls. No
permanent inspector column, compulsory page transition, or nested modal stack
is required.

A tensor row's matrix button opens this same single modal with the exact clicked
parameter already selected, without a preliminary component inspector, dropdown
choice or explorer-tab change. Resolve it against the active graph/model/session
and use only its verified `inspection.tensor_id`. Concrete repeated/shared views
bind the chosen instance; structure-only shared views expose no numeric action.
Unavailable, fused, unresolved and unsupported-rank capabilities retain a visibly
non-actionable button with the supplied accessible reason. Pointer and keyboard
activation cannot toggle, isolate, center or select another card or parent.
Summary rendering and hover never fetch tensor values.

For an available logical rank-1/rank-2 weight, compose the existing Matrix Explorer with the existing logical tensor endpoints and progressive subscriptions. The backend may provide native or admitted decoded weights; the UI performs no quantization decoding. Show its authoritative statistics/profiles when supported, preserving their independent arrival. Camera, readout, scalar-value fidelity, and resource lifetime remain owned by [matrix composition](architecture.md#matrix-explorer-composition), [rendering](rendering.md), and [Tensor Explorer](tensor-explorer.md). The graph's outer CSS/camera transform must not rescale this scientific surface. In particular, do not roll back the accepted Matrix Explorer zoom/navigation to implement graph inspection.

An unavailable representation, fused region, unresolved binding, or higher-rank tensor shows its metadata and localized reason without trying to open a substitute matrix. Do not display packed integers as dequantized values or flatten patch-projection weights. A weight error does not close or invalidate the graph.

Closing the modal restores focus and leaves camera/expansion intact. Escape closes it; focus stays contained while open. Keyboard users can select/expand/inspect components and connections without pointer-only access. Release numeric subscriptions, operation handles, CPU/GPU resources, and obsolete callbacks on close or replacement. A late result, including a same-shaped or same-ID return from an obsolete generation, must not populate a newer selection. Restore focus to the activating matrix button when it remains connected. Existing stream cancellation must preserve other consumers of shared work.

Resource references are semantic: model scope, modules, parameters, and an optional tokenizer. They contain no UI routes. The initial numeric action is weight inspection; future Tokenizer Explorer or computation-view links can be added by the UI without rewriting the static graph. Do not display buttons for unimplemented endpoints or fabricate runtime activation resources.

## Language and V-JEPA presentation

A standalone tokenizer capability is auxiliary tooling, not a model component.
Exclude its card and browser/search row only when its source is a portless,
parameterless `context` leaf with only tokenizer references and no incident edge,
formula or operation. Keep its source descriptor unchanged. A connected context,
state, real visual encoder or operation is not excluded because it references a
tokenizer. Do not add an input-preparation card, fabricate a tokenization edge,
or change Tokenizer Explorer or cross-explorer navigation. Other Qwen modalities
retain their supplied context and coverage limitations.

For the selected V-JEPA 2 model, use symbolic visual input, patch preparation, encoder, context/target selection inputs, predictor, and representation outputs. Both stacks expand in the same global canvas. No text-tokenizer error, vocabulary head, video player, upload selector, preprocessing control, action planner, or training graph is introduced. Native matrix/vector weights use the same modal; higher-rank parameters retain explicit shape/limitation information.

## Acceptance

Deterministic component and real-browser tests cover compact and fully expanded views, concrete repeated instances, bounded repetition focus, hybrid ordering, cross-group dependencies, shape toggling, centering/context retention, accessible node/connection inspection, native weight streaming, unavailable bindings, and late-result rejection. Exercise graph/model/session switching, close during loading, cache-unavailable and restart-required states, and repeated creation/disposal without leaked renderer resources.

Automatic-layout acceptance must prove generated geometry, not merely a configured left-to-right flag or manually adjusted coordinates. Known serial stages must progress horizontally without overlap at the root and recursively inside expanded groups and repetition windows, while genuine parallel/state/residual branches retain appropriate vertical separation. Cover different layer counts, orders, variants, and node sizes so a checkpoint-specific hand-tuned layout cannot satisfy the tests.

Connection interaction tests must actually hover/focus source ports, destination ports, and line segments. Verify exact highlighted connection sets for single connections, genuine fan-out, multiple same-shaped ports, nested group-boundary forwarding, residual paths, and separate state/K/V routes. Hover, focus, and pinning must leave node coordinates, camera state, and layout invocation state unchanged.

Visual acceptance uses the established neutral/amber language and verifies readable labels, visible direction, non-overlapping nodes/labels, and distinguishable multi-signal routing at representative desktop widths. Compare the surrounding application before and after with only the Architecture panel interior allowed to change; the application shell, Tensor Explorer, Tokenizer Explorer, and shared Matrix Explorer presentation must not acquire geometry, typography, color, or interaction regressions from Architecture-specific work.

Integrated tests use the built UI, real HTTP backend, and the repository's existing browser/WebGL2 harness. Verify that architecture failures do not break existing tensor/tokenizer workflows and that absence of a tokenizer on V-JEPA does not trigger phantom tokenization calls. A structural graph fixture proves UI behavior, not actual checkpoint support; reference evidence is governed by [analysis validation](../backend/architecture-analysis.md#validation-and-evidence).

## Model-supplied provenance

For a graph with transport scope `model_defined`, retain a visible notice that
its structure was supplied by the checkpoint author. Structural and inventory
validation do not establish equivalence to the implementation. The notice remains
visible when exploring a component, without adding sport-specific behavior or
changing source-graph navigation. Other graph scopes retain their current UI.
