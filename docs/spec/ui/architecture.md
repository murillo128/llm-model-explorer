# UI architecture

## Runtime

The browser UI is built with React, TypeScript, and Vite.

The UI and backend are independent deployables and may run on different computers. The UI therefore uses a configurable backend base URL rather than assuming the API shares its host or origin.

The UI owns user interaction, session orchestration, requests for explicit computation steps, progressive stream consumption, and visual presentation. It does not own model filesystem access, quantization decoding, or mathematical model execution.

## Renderer boundary

WebGL2 rendering is implemented as a TypeScript module independent from React. React components compose application views and controls; the renderer owns GPU resources and drawing.

Reusable tensor/matrix/vector rendering primitives must not be embedded as one-off React-specific implementations. Later attention, activation, and matrix-operation views should be able to reuse the same renderer primitives.

WebGL2 is used for visualization and visual transformations, not as the authoritative model-compute runtime. PyTorch on the backend owns mathematical computation.

## Application scope

The proof-of-concept application exposes two principal modules: Tensor Explorer and Tokenizer Explorer. Their detailed UI and interaction specifications are intentionally defined in separate documents and are not duplicated in this architecture specification.

The UI may keep a `session_id` across a page refresh so it can reconnect to an in-memory backend session while that backend process remains alive.

## Application shell

React composes one reusable global application bar, a bounded central workspace,
and a bottom status bar. The shell owns explorer navigation, model selection,
session actions, connection context, and low-priority model/session metadata.
Shell dimensions, responsive controls, and presentation are owned by
[`visual-language.md`](visual-language.md#application-shell-and-panel-geometry).
The same viewport frame also bounds configuration loading and recovery.

The document does not scroll during normal application use. The workspace takes
the height left by the two bars, and each explorer owns its bounded pane/scroll
composition (see [Tensor Explorer](tensor-explorer.md#scroll-synchronization)).
Explorer switching replaces the workspace without duplicating
page headings or model identity. This containment must preserve renderer-native
data geometry and progressive consumption; it does not alter explorer internals.
Session creation, recovery, refresh, deletion, and consumer lifetimes continue to
use the existing session controller and API contract.

## Progressive results

The UI must consume long-operation responses incrementally. It must not wait for an entire tensor payload before allocating/filling the rendering representation and beginning display.

Consumer cancellation must abort the active stream and use the operation cancellation contract where required.

## Matrix Explorer composition

`ui/src/matrix-explorer/` owns the reusable React scientific composition above
`MatrixViewport`: viewport lifetime, local scrolling, optional aligned profiles,
compact contextual header slot, and hover/focus magnifier/readout. It accepts
logical float32 descriptors and progressive subscriptions without application,
model, navigation, or HTTP dependencies. Tensor Explorer remains responsible for
its operation handles, protocol validation, cancellation and result status.

A source object identifies one immutable descriptor and delivery generation.
Replacing it detaches the prior subscription and releases its viewport; retained
callbacks cannot affect a later allocation, including returning to the same source.
Subscriptions restart at offset zero on remount. Scalar chunks are consumed
synchronously without retaining a second scalar array or accumulated chunk list.
Transfer/statistics updates change draw state without uploading scalar values.

Matrix-only composition omits auxiliary panels when the caller has no authoritative
distribution artifact. Full composition reserves the existing aligned 100-bin
uint32 profiles and accepts their progressive counts independently. Both reuse the
same exact geometry, transfer and inspection path; scientific semantics remain
owned by [rendering](rendering.md) and [Tensor Explorer](tensor-explorer.md).
Native zero-based cell, row and column callbacks let parents link context without
accessing renderer internals; leaving, blur and source disposal clear selection.
