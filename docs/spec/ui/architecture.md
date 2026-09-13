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

## Progressive results

The UI must consume long-operation responses incrementally. It must not wait for an entire tensor payload before allocating/filling the rendering representation and beginning display.

Consumer cancellation must abort the active stream and use the operation cancellation contract where required.
