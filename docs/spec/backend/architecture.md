# Backend architecture

## Runtime and responsibilities

The backend is implemented in Python. FastAPI/Starlette provides the HTTP service layer. PyTorch is the tensor and compute runtime.

Heavy tensor computation must use PyTorch operations rather than Python element-by-element loops. CUDA is the optimized execution path when selected and available; CPU must remain a fully supported execution path with the same API behavior. The UI must not need to know which device the backend is using.

No project-owned C, C++, or Rust compute layer is required for the proof of concept. Optimized native execution is obtained through PyTorch and its underlying CPU/CUDA kernels.

The backend owns local model discovery and access, tensor materialization, tokenizer execution, derived tensor operations, statistics, sessions, long-operation lifecycle, device scheduling, and persistent derived artifacts.

## UI-driven execution

The backend never advances model inference autonomously. Every computational step is initiated by a UI request. A future Play mode is implemented by the UI requesting successive steps.

This rule applies to the proof of concept as well: opening a tensor, calculating its statistics, and tokenizing input are explicit requests. Later inference operations extend the same execution model.

## Deployment

Backend and UI are separate deployable processes and may run on different computers. The backend exposes a configurable host and port. CORS is configurable to permit the intended UI origin.

The proof of concept has no authentication and assumes a trusted local network or otherwise trusted environment.

The backend is configured primarily through command-line options. Required configuration includes the local model root, artifact-cache directory, compute device selection, and network binding. A configuration file is not required for the proof of concept.

## Read-only model source

Files below the configured model root are treated as read-only. The backend must never rewrite model weights, tokenizer files, or model configuration. Every derived representation or result is written only to the artifact-cache directory.
