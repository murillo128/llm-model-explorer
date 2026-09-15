# Architecture Explorer

## Purpose and integration

Architecture Explorer is an additional static model view inside the existing bounded application shell. It does not replace Tensor Explorer, Tokenizer Explorer, or Matrix Explorer. [Product scope](../product.md#architecture-reference-checkpoints-and-bounded-coverage) defines the selected model coverage; [the API contract](../api/architecture-explorer.md) provides the prepared graph and resource references.

Fetch the prepared architecture for the selected session when its view is needed. This is retrieval, not generation. Display available complete/partial coverage or a model-local unavailable reason; the server has already finished preparation before accepting connections. A restart-required result must explain that restarting prepares new or changed models. Do not add a regenerate button, analysis polling, metadata-only onboarding, or a text prompt needed to see the graph.

A failure affects only this capability. Preserve all existing supported exploration workflows, matrix camera behavior, prompt editing, embedding linkage, streaming, and cancellation. New quantized tensor inventories must show their explicit partial-coverage diagnostic rather than implying all parameters are numerically inspectable.

Architecture-specific navigation, layout, routing, legends, repetition controls, and connection inspection stay inside the existing Architecture Explorer workspace. They must not resize, restyle, or relocate the shared application bar, explorer navigation, model selector, session controls, page background, bottom status bar, Tensor Explorer, Tokenizer Explorer, or shared Matrix Explorer primitives.

## One global navigable diagram

Use a single pannable, zoomable canvas with fit-to-window and centering on a selected component. Hierarchy organizes nested groups on the same canvas; it must not require separate level pages, drill-down routes, or stacked dialogs to understand structure.

Start compact, using declared repetition records to represent repeated layers as a stack with multiplicity and truthful variant information instead of immediately materializing every instance. Allow expansion of a representative interior, selection of a concrete instance without expanding its siblings, focused exploration of a bounded contiguous set of instances when useful, expansion of chosen instances, and an Expand all action that reveals every instance and required mathematical operation. Compressed before/after ranges must remain explicit when only part of a stack is shown. A representative display must clearly identify its selected instance; inspecting a weight must never silently use layer zero for every repetition.

Keep actual layer order and variant differences visible when grouped. Repetition means repeated structure, not shared weights, states, parameters, or guaranteed equivalence between variants. Preserve external dependencies across collapsed boundaries through meaningful ports/connections, including bypasses that cross hidden ranges. Collapsing must not erase a skip connection, create a false serial path, or treat distinct hybrid blocks as identical. Expanded topology must be recoverable from the same graph without new analysis.

Expanding a group preserves the acted-on location/context; do not reset the camera or automatically fit the entire graph after every expansion. Maintain camera, expanded groups, focused repetition/instance context, and valid node/connection selection while closing inspection or switching explorers during the browser session, keyed by model and graph identity. Server-restart persistence is not required. Reject late responses and clear invalid selections on session/model/graph replacement.

Viewport culling and asynchronous layout are allowed optimizations; discarding graph records or silently reducing detail is not. Bound layout work and provide an explicit recoverable failure rather than an indefinitely frozen canvas. All reference graphs must be usable when fully expanded on the documented acceptance environment. Record actual layout time, memory, and graph size rather than inventing a performance guarantee.

React Flow and ELK are implementation candidates, not mandatory backend dependencies or reasons to change the graph contract. The layout implementation may be replaced without changing model semantics. Do not add a second matrix renderer, graph editor, export suite, or complex navigation framework in this increment.

## Automatic layout and directional routing

The graph communicates computational direction primarily from left to right. Apply that ordering recursively at every expanded scope: the model, repeated stacks, concrete layers, attention blocks, and other expanded groups. Successive stages in a genuine serial dependency chain advance horizontally in their actual dependency order; a serial set of layers or operations must not degrade into a vertical list merely because it is nested inside a group.

Use vertical separation for genuine parallel branches, auxiliary inputs, residual bypasses, independent state paths, and independent components. Parallel Q/K/V, gate/up, or similar branches may occupy separate rows when the source topology supports them. The layout must not derive semantic order from labels, path names, or alphabetical sorting when graph dependencies provide the real order.

Expanded group geometry is determined by the visible children, their labels, ports, and connection clearances. Collapsed descendants and their internal routes do not contribute to visible layout bounds. When a horizontal computation is wider than the Architecture panel, preserve computational order and use graph pan/zoom and compact repetition navigation rather than wrapping the serial chain vertically, shrinking labels until unreadable, resizing the application shell, or creating document-level scrolling.

Connection routing must keep distinct signals visually distinguishable. Routes should avoid unrelated node bodies and labels, preserve exact source and destination ports, and use clear destination arrowheads. A shared trunk is valid only for genuine fan-out from the same source signal/port; unrelated signals that happen to share a label, shape, or corridor must remain individually traceable. Group-boundary forwarding may be composed into one visible route, but must retain the original edge and port identities represented by that route.

Changing hover, keyboard focus, or pinned connection selection must not move nodes, change port positions, refit the camera, or recompute layout. A layout failure is recoverable and explicit; it must not silently fall back to an unreadable vertical stacking mode.

## Clean default presentation and dimensions

Default nodes show concise operation/component labels, relevant type distinctions, groups, and connections. Details do not occupy permanent side panels. Dimensions are hidden initially; Show dimensions adds endpoint tensor shapes to connections. Shapes distinguish constants, declared symbols, expressions, and unknown values; expressions are rendered as text, not evaluated.

Full paths, formulas, descriptions, configuration attributes, provenance, diagnostics, and parameter references belong in inspection rather than permanently filling the canvas. A graph describes mathematical operations and selected meaningful shape changes, not every PyTorch instruction or allocation. Coverage and unknown regions remain visible enough to avoid interpreting a partial diagram as complete.

Use the shared [visual language](visual-language.md). Architecture nodes, groups, controls, ports, and connections use the existing neutral surfaces, graphite text, hairline borders, restrained shadows, and warm amber interaction accent rather than introducing a new application theme or per-layer categorical palette. Graph zoom is its own camera. It must never zoom the prompt editor or impose a transform on a Matrix Explorer surface.

## Connection hover, focus, and selection

Connections are inspectable from both their lines and their ports. Hovering or keyboard-focusing a visible connection emphasizes its complete visible route from the actual source port to the actual destination port, including composed group-boundary segments, endpoint markers/labels, and destination arrowhead. Hovering an exact input port emphasizes the incoming connection or connections terminating at that port and their sources. Hovering an exact output port emphasizes all outgoing connections from that port and their destinations; genuine fan-out remains visible as fan-out, while each branch stays individually targetable from its own line.

Resolve these interactions from exact port identity and source-edge mappings. Following a boundary pass-through keeps one represented connection visually continuous across scopes, but interaction stops at computational operations: hovering an input does not recursively highlight every downstream descendant. Hidden nodes are not automatically expanded and no fabricated internal path is revealed. A collapsed group shows the visible external connection it represents while retaining traceability to the underlying source edges for inspection.

Connection and port hit areas may be larger than their visible strokes/markers so thin routes remain practical to target at normal graph zoom. Ambiguous crossings must still leave distinct connections targetable through an unambiguous segment or endpoint. Port interaction must not initiate graph editing or accidentally trigger an unrelated node action.

Hover/focus emphasis is temporary and uses the existing warm amber interaction semantics together with non-color cues such as stroke/marker emphasis and directional arrowheads. Nonparticipating connections may be visually subdued but remain readable. Click/keyboard activation may pin a connection for inspection; leaving a hover or focus restores the pinned selection rather than clearing it. Moving between a port and its connected line should not produce a flickering loss of context.

## Simple modal inspection and resource navigation

Clicking/keyboard-activating an inspectable node opens one closable modal with its identity, function, known dimensions, provenance/diagnostics, and parameter choices. Group expansion remains a distinct simple affordance. No permanent inspector column, compulsory page transition, or nested modal stack is required.

For an available native rank-1/rank-2 weight, compose the existing Matrix Explorer with the existing logical tensor endpoints and progressive subscriptions. Show its authoritative statistics/profiles when supported, preserving their independent arrival. Camera, readout, scalar-value fidelity, and resource lifetime remain owned by [matrix composition](architecture.md#matrix-explorer-composition), [rendering](rendering.md), and [Tensor Explorer](tensor-explorer.md). The graph's outer CSS/camera transform must not rescale this scientific surface. In particular, do not roll back the accepted Matrix Explorer zoom/navigation to implement graph inspection.

An unavailable representation, fused region, unresolved binding, or higher-rank tensor shows its metadata and localized reason without trying to open a substitute matrix. Do not display packed integers as dequantized values or flatten patch-projection weights. A weight error does not close or invalidate the graph.

Closing the modal restores focus and leaves camera/expansion intact. Escape closes it; focus stays contained while open. Keyboard users can select/expand/inspect components and connections without pointer-only access. Release numeric subscriptions, operation handles, CPU/GPU resources, and obsolete callbacks on close or replacement. A late result must not populate a different node/model's modal. Existing stream cancellation must preserve other consumers of shared work.

Resource references are semantic: model scope, modules, parameters, and an optional tokenizer. They contain no UI routes. The initial numeric action is weight inspection; future Tokenizer Explorer or computation-view links can be added by the UI without rewriting the static graph. Do not display buttons for unimplemented endpoints or fabricate runtime activation resources.

## Language and V-JEPA presentation

For language models, show tokenization as a distinct context block before the neural model when applicable. It explains the source of token IDs and preserves the tokenizer reference; it does not execute tokenization or require a new cross-explorer action. When the source graph provides tokenizer capability only as context and no computational ports/edges, the UI may show a clearly contextual association but must not fabricate a data-flow edge, port, or claim that the tokenizer produced positions or masks. Other Qwen modalities remain labelled context, outside detailed language coverage.

For the selected V-JEPA 2 model, use symbolic visual input, patch preparation, encoder, context/target selection inputs, predictor, and representation outputs. Both stacks expand in the same global canvas. No text-tokenizer error, vocabulary head, video player, upload selector, preprocessing control, action planner, or training graph is introduced. Native matrix/vector weights use the same modal; higher-rank parameters retain explicit shape/limitation information.

## Acceptance

Deterministic component and real-browser tests cover compact and fully expanded views, concrete repeated instances, bounded repetition focus, hybrid ordering, cross-group dependencies, shape toggling, centering/context retention, accessible node/connection inspection, native weight streaming, unavailable bindings, and late-result rejection. Exercise graph/model/session switching, close during loading, cache-unavailable and restart-required states, and repeated creation/disposal without leaked renderer resources.

Automatic-layout acceptance must prove generated geometry, not merely a configured left-to-right flag or manually adjusted coordinates. Known serial stages must progress horizontally without overlap at the root and recursively inside expanded groups and repetition windows, while genuine parallel/state/residual branches retain appropriate vertical separation. Cover different layer counts, orders, variants, and node sizes so a checkpoint-specific hand-tuned layout cannot satisfy the tests.

Connection interaction tests must actually hover/focus source ports, destination ports, and line segments. Verify exact highlighted connection sets for single connections, genuine fan-out, multiple same-shaped ports, nested group-boundary forwarding, residual paths, and separate state/K/V routes. Hover, focus, and pinning must leave node coordinates, camera state, and layout invocation state unchanged.

Visual acceptance uses the established neutral/amber language and verifies readable labels, visible direction, non-overlapping nodes/labels, and distinguishable multi-signal routing at representative desktop widths. Compare the surrounding application before and after with only the Architecture panel interior allowed to change; the application shell, Tensor Explorer, Tokenizer Explorer, and shared Matrix Explorer presentation must not acquire geometry, typography, color, or interaction regressions from Architecture-specific work.

Integrated tests use the built UI, real HTTP backend, and the repository's existing browser/WebGL2 harness. Verify that architecture failures do not break existing tensor/tokenizer workflows and that absence of a tokenizer on V-JEPA does not trigger phantom tokenization calls. A structural graph fixture proves UI behavior, not actual checkpoint support; reference evidence is governed by [analysis validation](../backend/architecture-analysis.md#validation-and-evidence).
