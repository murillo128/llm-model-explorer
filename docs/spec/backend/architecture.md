# Backend architecture

## Runtime and responsibilities

The backend is implemented in Python. FastAPI/Starlette provides the HTTP service layer. PyTorch is the tensor and compute runtime.

Heavy tensor computation must use PyTorch operations rather than Python element-by-element loops. CUDA is the optimized execution path when selected and available; CPU must remain fully supported with the same API behavior. The UI must not need to know which device the backend uses. No project-owned C, C++, or Rust compute layer is required.

The backend owns local model discovery and access, tensor materialization, tokenizer execution, derived operations, statistics, sessions, long-operation lifecycle, scheduling, and persistent derived artifacts. The Architecture Explorer increment adds static semantic analysis described in [architecture-analysis.md](architecture-analysis.md), not another numerical runtime.

## UI-driven execution

The backend never advances model inference autonomously. Every inference step is initiated by a UI request; a future Play mode requests successive steps. Tensor materialization, statistics, and tokenization continue to use explicit requests.

Static architecture preparation is an explicit startup exception to demand-driven derived work. It does not authorize background inference, tokenization, preprocessing, or eager weight materialization.

## Blocking architecture preparation

For the accepted architecture increment, service readiness is gated on preparation. After validating general configuration and opening the existing services/cache, discover eligible local checkpoints, establish their content snapshots, and reuse or produce their architecture results. No application request is served until all candidates have a terminal architecture outcome. No user session or HTTP long-operation ID is needed for this phase.

Process pending models sequentially for this first increment, bounding working memory to one analysis at a time. Reuse already valid graph artifacts. Logs identify the logical model, cache reuse or analysis, and terminal outcome; report hashing, analysis, and cache-read timing separately. Full-file fingerprint reads may still be needed on a warm start; do not promise instant readiness because graphs are cached.

Terminal outcomes are available (complete or partial within scope) or unavailable with a safe diagnostic. A model-local analysis failure must not prevent later models from being processed or final service readiness. Invalid checkpoints remain subject to catalogue diagnostics in [models.md](models.md). Global failures that make the configured backend unsafe or unusable still fail startup. Graceful shutdown during preparation stops at safe bounded-work boundaries, aborts unpublished artifacts, and releases resources; it does not proceed to readiness.

There is no architecture directory watcher, UI regeneration action, on-demand analysis, or post-start background preparation. Models added or changed require the next restart for a new architecture result. This rule does not replace the existing discovery/session behavior of the other explorers. Detected stale snapshots may never be served or rebound to new weights.

## Deployment

Backend and UI are separate deployables and may run on different computers. Host, port, model root, artifact cache, compute device, and CORS remain CLI-configured. No configuration file or extra public service is required. Architecture preparation is CPU/metadata work and does not require a GPU or a separate inference process.

The proof of concept has no authentication and assumes a trusted network or equivalent trusted environment. Existing deployment assumptions remain unchanged.

## Read-only model source

Files below the model root remain read-only. Never rewrite weights, tokenizer files, or model configuration. Derived results go only to the artifact cache. Do not execute checkpoint Python code, download absent assets, or call a model forward pass to build a diagram. Bounded integrity hashing is not loading the checkpoint into resident numeric tensors.
