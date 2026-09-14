# Architecture Explorer

## Purpose and integration

Architecture Explorer is an additional static model view inside the existing bounded application shell. It does not replace Tensor Explorer, Tokenizer Explorer, or Matrix Explorer. [Product scope](../product.md#architecture-reference-checkpoints-and-bounded-coverage) defines the selected model coverage; [the API contract](../api/architecture-explorer.md) provides the prepared graph and resource references.

Fetch the prepared architecture for the selected session when its view is needed. This is retrieval, not generation. Display available complete/partial coverage or a model-local unavailable reason; the server has already finished preparation before accepting connections. A restart-required result must explain that restarting prepares new or changed models. Do not add a regenerate button, analysis polling, metadata-only onboarding, or a text prompt needed to see the graph.

A failure affects only this capability. Preserve all existing supported exploration workflows, matrix camera behavior, prompt editing, embedding linkage, streaming, and cancellation. New quantized tensor inventories must show their explicit partial-coverage diagnostic rather than implying all parameters are numerically inspectable.

## One global navigable diagram

Use a single pannable, zoomable canvas with fit-to-window and centering on a selected component. Hierarchy organizes nested groups on the same canvas; it must not require separate level pages, drill-down routes, or stacked dialogs to understand structure.

Start compact, grouping repeated layers. Allow expansion of a representative interior, selection of a concrete instance, expansion of chosen instances, and an Expand all action that reveals every instance and required mathematical operation. A representative display must clearly identify its selected instance; inspecting a weight must never silently use layer zero for every repetition.

Keep actual layer order and variant differences visible when grouped. Preserve external dependencies across collapsed boundaries through meaningful ports/connections. Collapsing must not erase a skip connection, create a false serial path, or treat distinct hybrid blocks as identical. Expanded topology must be recoverable from the same graph without new analysis.

Expanding a group preserves the acted-on location/context; do not reset the camera or automatically fit the entire graph after every expansion. Maintain camera, expanded groups, and selection while closing inspection or switching explorers during the browser session, keyed by model and graph identity. Server-restart persistence is not required. Reject late responses and clear invalid selections on session/model/graph replacement.

Viewport culling and asynchronous layout are allowed optimizations; discarding graph records or silently reducing detail is not. Bound layout work and provide an explicit recoverable failure rather than an indefinitely frozen canvas. All reference graphs must be usable when fully expanded on the documented acceptance environment. Record actual layout time, memory, and graph size rather than inventing a performance guarantee.

React Flow and ELK are implementation candidates, not mandatory backend dependencies or reasons to change the graph contract. The layout implementation may be replaced without changing model semantics. Do not add a second matrix renderer, graph editor, export suite, or complex navigation framework in this increment.

## Clean default presentation and dimensions

Default nodes show concise operation/component labels, relevant type distinctions, groups, and connections. Details do not occupy permanent side panels. Dimensions are hidden initially; Show dimensions adds endpoint tensor shapes to connections. Shapes distinguish constants, declared symbols, expressions, and unknown values; expressions are rendered as text, not evaluated.

Full formulas, descriptions, configuration attributes, and parameter references belong in inspection rather than permanently filling the canvas. A graph describes mathematical operations and selected meaningful shape changes, not every PyTorch instruction or allocation. Coverage and unknown regions remain visible enough to avoid interpreting a partial diagram as complete.

Use the shared [visual language](visual-language.md); graph zoom is its own camera. It must never zoom the prompt editor or impose a transform on a Matrix Explorer surface.

## Simple modal inspection and resource navigation

Clicking/keyboard-activating an inspectable node opens one closable modal with its identity, function, known dimensions, provenance/diagnostics, and parameter choices. Group expansion remains a distinct simple affordance. No permanent inspector column, compulsory page transition, or nested modal stack is required.

For an available native rank-1/rank-2 weight, compose the existing Matrix Explorer with the existing logical tensor endpoints and progressive subscriptions. Show its authoritative statistics/profiles when supported, preserving their independent arrival. Camera, readout, scalar-value fidelity, and resource lifetime remain owned by [matrix composition](architecture.md#matrix-explorer-composition), [rendering](rendering.md), and [Tensor Explorer](tensor-explorer.md). The graph's outer CSS/camera transform must not rescale this scientific surface. In particular, do not roll back the accepted Matrix Explorer zoom/navigation to implement graph inspection.

An unavailable representation, fused region, unresolved binding, or higher-rank tensor shows its metadata and localized reason without trying to open a substitute matrix. Do not display packed integers as dequantized values or flatten patch-projection weights. A weight error does not close or invalidate the graph.

Closing the modal restores focus and leaves camera/expansion intact. Escape closes it; focus stays contained while open. Keyboard users can select/expand/inspect components without pointer-only access. Release numeric subscriptions, operation handles, CPU/GPU resources, and obsolete callbacks on close or replacement. A late result must not populate a different node/model's modal. Existing stream cancellation must preserve other consumers of shared work.

Resource references are semantic: model scope, modules, parameters, and an optional tokenizer. They contain no UI routes. The initial numeric action is weight inspection; future Tokenizer Explorer or computation-view links can be added by the UI without rewriting the static graph. Do not display buttons for unimplemented endpoints or fabricate runtime activation resources.

## Language and V-JEPA presentation

For language models, show tokenization as a distinct context block before the neural model when applicable. It explains the source of token IDs and preserves the tokenizer reference; it does not execute tokenization or require a new cross-explorer action. Other Qwen modalities remain labelled context, outside detailed language coverage.

For the selected V-JEPA 2 model, use symbolic visual input, patch preparation, encoder, context/target selection inputs, predictor, and representation outputs. Both stacks expand in the same global canvas. No text-tokenizer error, vocabulary head, video player, upload selector, preprocessing control, action planner, or training graph is introduced. Native matrix/vector weights use the same modal; higher-rank parameters retain explicit shape/limitation information.

## Acceptance

Deterministic component and real-browser tests cover compact and fully expanded views, concrete repeated instances, hybrid ordering, cross-group dependencies, shape toggling, centering/context retention, accessible modal/focus, native weight streaming, unavailable bindings, and late-result rejection. Exercise graph/model/session switching, close during loading, cache-unavailable and restart-required states, and repeated creation/disposal without leaked renderer resources.

Integrated tests use the built UI, real HTTP backend, and the repository's existing browser/WebGL2 harness. Verify that architecture failures do not break existing tensor/tokenizer workflows and that absence of a tokenizer on V-JEPA does not trigger phantom tokenization calls. A structural graph fixture proves UI behavior, not actual checkpoint support; reference evidence is governed by [analysis validation](../backend/architecture-analysis.md#validation-and-evidence).
