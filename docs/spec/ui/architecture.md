# UI architecture

## Runtime

The browser UI is built with React, TypeScript, and Vite. UI and backend are independent deployables and may run on different computers; use a configurable backend base URL, not an assumed shared origin.

The UI owns interaction, session orchestration, explicit-computation requests, progressive stream consumption, and presentation. It does not own filesystem access, quantization decoding, mathematical execution, or semantic reconstruction of a model from checkpoint names.

## Renderer boundary

WebGL2 tensor rendering remains a TypeScript module independent from React. React composes views; the renderer owns GPU resources and drawing. Reusable tensor/matrix/vector primitives are not one-off React implementations. Later attention, activation, and matrix-operation views reuse the same boundary. PyTorch remains authoritative for computation.

The Architecture Explorer graph canvas is a separate presentation concern. It consumes a backend-produced semantic graph and owns layout, grouping, pan, and zoom. It need not use the scalar WebGL2 tensor renderer for node/edge drawing. Its camera never transforms Matrix Explorer or the prompt editor.

## Application scope

The initial modules are Tensor Explorer and Tokenizer Explorer. The accepted Architecture Explorer increment adds a third view; [architecture-explorer.md](architecture-explorer.md) owns its interaction and [the API contract](../api/architecture-explorer.md) owns its graph. Acceptance in the specification does not mean this view is already implemented.

The UI may retain `session_id` across refresh to reconnect while the backend session exists. Explorer capability failures remain local. Do not require a text tokenizer to open a valid V-JEPA 2 session, or require a graph analysis to use an otherwise supported tensor/tokenizer workflow.

## Application shell

React composes one reusable application bar, a bounded workspace, and a bottom status bar. The shell owns explorer navigation, model selection, sessions, connection context, and low-priority metadata. Geometry and styling are owned by [visual-language.md](visual-language.md#application-shell-and-panel-geometry), including configuration/recovery views.

Normal use does not scroll the document. Each explorer owns its bounded composition inside the remaining workspace. Switching explorers does not duplicate page headings or model identity. Session lifecycle and consumer ownership continue through the existing controller and API. Preserve accepted matrix-camera and tokenizer stale-result behavior; the new graph does not redesign either explorer.

Global navigation renders Architecture Explorer, Tokenizer Explorer, then Tensor
Explorer in DOM, keyboard, and visual order. This ordering does not determine the
active view: a fresh application still starts in Tensor Explorer, while an
explicitly selected explorer remains active across ordinary shell updates.

Keep path-free catalogue diagnostics for rejected PEFT adapter candidates visible in the workspace notices, including when no selectable model is available. Rejected adapter compositions never appear as model options and cannot start sessions.

## Progressive results

Consume numerical long-operation responses incrementally; begin rendering before the entire tensor arrives. Cancellation aborts the active stream and uses operation cancellation where required.

Architecture is already prepared at startup and retrieved as bounded structured JSON. Its response has graph identity and capability states, not progressive numeric values or an analysis job to poll. Keep graph and modal requests fenced by session/model/graph identity; late callbacks must not overwrite a later selection.

## Matrix Explorer composition

`ui/src/matrix-explorer/` owns the reusable React scientific composition above `MatrixViewport`: viewport lifetime, per-instance square-cell camera and local scrolling, optional aligned profiles, compact contextual header slot, and hover/focus magnifier/readout. It accepts logical float32 descriptors and progressive subscriptions without model, navigation, or HTTP dependencies. The consuming explorer owns protocol validation, operations, cancellation, and result status.


A source object identifies one immutable descriptor and delivery generation. Replacing it detaches subscriptions and releases its viewport; old callbacks cannot affect a later allocation, including a return to the same source. Subscriptions restart at offset zero on remount. Scalar chunks are consumed synchronously without a second scalar array or accumulated chunk list. Transfer/statistics updates change draw state without scalar reuploads.

Matrix-only mode omits absent auxiliary panels. Full mode reserves the existing aligned 100-bin uint32 profiles with independent progressive delivery. Geometry, camera, scientific transfer, and inspection semantics remain owned by [rendering.md](rendering.md) and [tensor-explorer.md](tensor-explorer.md). Native cell/row/column callbacks allow semantic linkage without renderer internals; disposal clears selection.

Architecture weight inspection is another consumer of this same composition in a closable modal. It must not fork the renderer, copy its camera into the graph, or impose graph scale on the numeric surface. Graph nodes/resource references remain independent of UI routes so future Tokenizer Explorer or specialized-view links need only UI integration, not redesigned model descriptions.

## Connection and application feedback

The shell footer reports last observed backend reachability independently of session
existence: Connecting initially, Connected after a response (including HTTP or
protocol errors), Disconnected after transport failure, and Reconnecting only while
an actual request is pending after failure. Cancellation and session expiration do
not imply disconnection. Existing refresh/recovery actions remain explicit; no
heartbeat, polling, automatic numerical replay, or new backend endpoint is used.

Connection transitions use only the footer. Consequential transport failures may
produce one dismissible shell toast per request, while the affected explorer keeps
its local failed/retry state. Failed session close has an explicit retry action;
expiration retains a visible fresh-session recovery state independently of toasts.
Capability diagnostics and ordinary stream progress remain local. Safe public copy
must never interpolate arbitrary backend error text or filesystem paths.

Feedback is fenced by backend, session/selection and request lifetime. Replacing a
context clears its notifications and timers; obsolete requests cannot publish into
the replacement. Feedback updates and dismissal preserve explorer consumer,
renderer and camera identities. A single bounded host displays at most three
notifications, deduplicates repeat delivery of the same request outcome, keeps
errors until dismissed, and expires brief informational outcomes only while neither
hovered nor focused. Notifications never steal focus.


## Model information and diagnostic lifetime

The existing Session options entry exposes on-demand **Model information**, with
a labelled **Diagnostics** section and compact warning/error count. It is reachable
without node selection or a rendered graph, includes already-observed architecture
and tensor-inventory findings with capability, severity, safe source context and
complete messages, and retains dismissed inline findings. Model-supplied provenance
has its own explanation. Opening this section never retrieves architecture, tensor
bytes or tokenizer results. Explicitly distinguish unobserved capabilities from
observed capabilities with no findings.

Retain only current findings and minimal dismissal identity; never copy semantic
graphs or keep a historical notification log. Scope observations to the backend
and exact current model/session lifetime, rejecting late callbacks. Replacing or
clearing a session clears current observations. Dismissal identity uses model,
capability, graph/content generation (session generation when no graph identity is
available), diagnostic code, supplied source location, severity and message. It is
stable across rerenders, inspector closure, explorer switches and graph expansion.
New findings and severity escalation remain visible. Graph replacement invalidates
that model's stale dismissal identities; backend replacement resets the owner.
No disk persistence is needed. Architecture-specific inline notices remain local
to Architecture Explorer; inventory/tokenizer recovery and scientific surfaces
retain their existing behavior. Connection state and transient events continue
through the separate footer/toast routing above.
