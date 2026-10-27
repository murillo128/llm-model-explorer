# Bounded neuron execution

## Admission and ownership

This accepted increment adds real execution only for a complete local
`HuggingFaceTB/SmolLM2-135M` **Base** checkpoint with native Safetensors weights
and its local tokenizer. The initial reference revision is the one in
[product scope](../product.md#architecture-reference-checkpoints-and-bounded-coverage).
Validate identity/provenance, the reviewed SmolLM2 configuration and every required
weight shape/dtype/binding; a `llama` model type, matching dimensions, filename or
architecture graph alone is insufficient. Instruct, arbitrary Llama checkpoints,
other families, quantized weights and PEFT compositions are unsupported here.
Never import checkpoint Python, use remote code, download assets, or execute a
model-owned `architecture.json`. A sidecar neither grants nor removes independently
validated native execution support.

The fixed execution adapter owns admission, the real forward pass, hooks and
numerical truth. Static architecture analysis remains descriptive and independently
available. The [neuron API](../api/neuron-explorer.md) owns public records, limits,
experimental semantics and bytes. The [neuron view](../ui/neuron-explorer.md) owns
interaction. No general execution engine, graph interpreter or provider layer is
introduced.

## Concrete binding

For each actual layer, resolve its MLP, gate/up/down weights and intermediate
width from the reviewed adapter and pinned inventory. Gate and up have shape
`[intermediate_size, hidden_size]`; down has shape
`[hidden_size, intermediate_size]`. Neuron `j` denotes coordinate `j` of
`a = SiLU(gate(x)) * up(x)`, immediately before `down_proj`, and therefore links
gate row `j`, up row `j`, and down column `j`. It is not a weight cell, hidden-state
coordinate or semantic concept. Repeated layers are distinct instances.

Capability lookup uses bounded metadata and installed runtime information without
loading a model. Advertise the fixed adapter/version, actual configured device and
effective computation dtype; do not claim availability based on a static graph or
successful tensor decoding. A graph link is optional and must refer to a verified
concrete node in that exact prepared graph. Tensor bindings use existing inventory
IDs and complete-weight navigation, not a new slicing endpoint.

## Execution lifetime and integrity

Ordinary listing, session creation, static inspection and tokenization remain lazy.
An explicitly requested valid neuron operation may load the admitted checkpoint
for computation. Perform schema, capability, vocabulary, module/neuron, context,
size and finite-parameter validation **before** model loading or result allocation.
Apply the published limits without truncation, automatic token insertion, chat
templates, target reduction or sampling fallbacks.

Use PyTorch/native kernels in evaluation/inference mode with no gradient tracking.
Initial CPU computation uses float32; optional CUDA uses the same float32 logical
execution policy. Do not silently enable autocast, TF32, dropout or a different
precision/device. Record the actual device, adapter implementation revision,
runtime revision and PyTorch version. Device kernels need not be bit-identical;
reproduction claims must retain this environment and use justified numerical
tolerances. Captures, logits, softmax and interventions must be finite, including
after float32 conversion, or fail as specified by the API.

Each operation owns its temporary activations, hooks, condition state and any KV
state. Every condition starts from the same unmodified model and exact prompt;
continuations then have independent contexts. A hook replaces only the selected
activation coordinates at the declared point, never model weights. No cumulative
factor/delta may leak across conditions, tokens, requests or sessions. Scope uses
absolute input positions even when an incremental forward call contains one token.
Do not infer absolute position from the hook tensor's local row index.

Preserve [session pinning and scheduling](sessions-and-execution.md). Source guards
cover all required assets before/after loading and computation and throughout
delivery, with a final guard before COMPLETE. Mutation invalidates the pinned
operation; never reuse a loaded instance for changed content. Multiple sessions
remain independent. Serialize expensive GPU work through the existing per-device
queue. Shared immutable weights require exclusive or request-local hook ownership;
mutable intervention/continuation state is never shared.

Cancellation, disconnect, session deletion and failures settle owned work, remove
hooks, release queued work and temporary buffers at safe boundaries. An indivisible
kernel need not be interrupted. A subsequent clean baseline must behave like a
fresh baseline. These request-owned results are not artifact-producing work and
do not introduce cross-session experiment deduplication.

## Privacy and evidence

Do not automatically write prompt text/IDs, generated output, activations or
comparison tables to files, logs, traces or the artifact cache. This restriction
also covers failure diagnostics and framework debug callbacks. Retain only
operation-lifetime buffers; there is no backend experiment history. Existing
model-weight and static artifact lifetimes are unchanged.

Contract fixtures prove layout and validation only. Runtime acceptance separately
requires independent small mathematical oracles, hook isolation, all three scopes,
zero/scale/add, exact token forwarding, EOS/context boundaries, mutation and
cancellation cleanup, followed by the complete local reference checkpoint. A
fixture or static graph cannot establish executable-model support.
