# DeepSeek-V2-Lite MLA and MoE description

## Reviewed checkpoint and source

The packaged `deepseek-v2-lite` description (revision 1) is bounded to the
configuration and parameter names used by the selected DeepSeek-V2-Lite CausalLM.
Its primary source is the pinned
[`slowfastai/DeepSeek-V2-Lite-bnb-4bit` checkpoint at `9fc357346aeba67950a34a86f3520fc276ca2daf`](https://huggingface.co/slowfastai/DeepSeek-V2-Lite-bnb-4bit/tree/9fc357346aeba67950a34a86f3520fc276ca2daf).
The configuration and bundled `configuration_deepseek.py` and
`modeling_deepseek.py` were read as inert text. No checkpoint module was imported
or executed. The producer identity includes that exact source revision; the root
model group retains it in provenance. The selected checkpoint is an NF4,
double-quantized conversion of `deepseek-ai/DeepSeek-V2-Lite`.

`tests/fixtures/deepseek-v2-lite-reference.json` contains the pinned config and
selected physical name/dtype/shape records from both Safetensors headers. The
headers were requested with bounded HTTP Range reads; their SHA-256 values and
entry counts are retained. The two-shard index declares 31,196 tensor names.
Only index/config/header metadata was read during this review; no model payload
was downloaded or loaded. These compact header samples confirm native embedding,
norm and router records alongside flattened U8 NF4 weights and their quant-state
companions for the dense MLP, MLA, routed expert and final expert examples.

## Mathematical coverage

`architecture_analysis/deepseek_v2.py` describes the CausalLM evaluation path:
token embedding; all 27 ordered decoder layers; final RMSNorm and vocabulary
projection. Inputs, logits and symbolic per-layer cache states use root model
container ports. The optional tokenizer is an unconnected context node outside
the neural graph.

Each layer has pre-attention normalization, MLA, an attention residual,
post-attention normalization and a second residual. MLA preserves the direct
query projection (`q_lora_rank=null`), query split/recombine, compressed KV latent
and separate shared rotary-key branches, latent normalization, KV expansion,
rotary application, reconstructed key/value cache, causal attention and output
projection. The YARN frequency and score-scaling attributes come from the pinned
configuration. No cache sample or position is supplied.

Layer 0 uses the dense gate/up/SiLU/product/down path. Layers 1–26 contain a
softmax router and greedy top-6 selection; the configured unnormalized selected
scores receive the routed scaling factor. Each layer retains 64 direct child
expert groups, each with its own gate/up/down parameter bindings and a SwiGLU
operation followed by per-token routing weighting. A symbolic dispatch and
weighted scatter sum restore results at their source token positions. The two
shared experts remain one combined MLP of width 2,816, and its output is added
separately to the routed result. No router decision or token is evaluated.

Every model parameter has a separate logical binding. Native BF16 records resolve
to numeric inventory IDs when available. A reviewed NF4 U8 matrix binds to its
logical parameter while retaining the physical weight and available quantization
companions as storage metadata; inspection remains unavailable unless a verified
logical tensor view exists. Quantization metadata is not represented as neural
operations, and this description adds no decoder.

## Selection and coverage boundaries

The checker accepts the reviewed 27-layer Lite structure, its direct-query MLA
path, exact YARN values, softmax/greedy top-6 routing, one expert-parallel rank,
and the captured NF4 configuration or compatible native storage. Structural
unknowns, changed dimensions, expert counts, top-k behavior, routing options,
RoPE values, or query rank are rejected. Missing or incompatible parameters are
localized as unresolved in a partial graph. Extra unexplained storage prevents a
complete-coverage claim.

Focused tests use an independently written tensor geometry oracle and check all
27 layers, all 1,664 routed expert instances, every logical parameter, dense/MoE
exception, MLA split/recombine and rotary dependencies, routing and shared
branches, root ports, packed storage roles, graph closure, templates and the
32 MiB response limit. Network, model construction/calls, tracing, and tensor
materialization entry points are guarded during graph generation.

This is source/config/header-fixture evidence, not actual full-checkpoint
acceptance. The integrated-reference issue still owns local model admission,
content fingerprint, complete physical inventory, runtime/cache/browser
acceptance and associated environment measurements. No inference correctness,
numerical equivalence or universal DeepSeek-family coverage is claimed.
