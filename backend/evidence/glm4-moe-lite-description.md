# GLM-4.7-Flash static architecture description

## Pinned sources

The structural reference is `cyankiwi/GLM-4.7-Flash-AWQ-4bit` at
`25624b53414e585bcf7dcb9584667c3106c6089b`. Its reviewed `config.json` is
captured in `tests/fixtures/glm4-moe-lite-reference.json`; the captured file
SHA-256 is `079f8fcd477b07378deafc21ecc5c3b974bf109ce5c9115854d97f88fa7622b9`.
The implementation source is Hugging Face Transformers at
`c8b81b63232be35ab1774dd3cabbf499d8b9808f`, specifically
[`configuration_glm4_moe_lite.py`](https://github.com/huggingface/transformers/blob/c8b81b63232be35ab1774dd3cabbf499d8b9808f/src/transformers/models/glm4_moe_lite/configuration_glm4_moe_lite.py)
and
[`modeling_glm4_moe_lite.py`](https://github.com/huggingface/transformers/blob/c8b81b63232be35ab1774dd3cabbf499d8b9808f/src/transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py).
Both were read as inert text. The analyzer does not import or execute checkpoint
or Transformers code.

## Evaluated path and auxiliary prediction metadata

The checked CausalLM path has 47 ordered decoder layers. Transformers defaults
the MLP layer list to one dense layer followed by sparse layers and derives
`qk_head_dim` as `qk_nope_head_dim + qk_rope_head_dim` (configuration source
lines 118–122). The reference config also records
`num_nextn_predict_layers: 1`. The model source constructs the configured
`num_hidden_layers` layers (line 595), loops over that configured range in
`Glm4MoeLiteModel.forward` (line 642), and the CausalLM path applies `lm_head`
to the returned hidden state (lines 707–720). It does not call a next-token
prediction layer. The same source explicitly ignores unexpected
`model.layers.47.*` checkpoint keys (line 573). The graph retains the `nextn`
count and this source-backed treatment as model provenance; it creates no
nextn node or execution edge.

## Graph structure

Each decoder layer includes input RMSNorm, latent attention, an attention
residual, post-attention RMSNorm, a dense or MoE feed-forward path, and the
second residual. Latent attention keeps the low-rank query path (`q_a_proj`,
RMSNorm, `q_b_proj`) and the low-rank KV path (`kv_a_proj_with_mqa`, latent
RMSNorm, `kv_b_proj`). It splits each head into 192 non-RoPE and 64 RoPE
dimensions, applies the configured interleaved default RoPE to the rotary
branches, reconstructs 20 query/key/value heads, and retains attention,
softmax, value aggregation, output projection, and the symbolic compressed KV
cache. No cache contents or token positions are evaluated.

The first layer uses the 10,240-wide dense gated MLP. The remaining 46 layers
have a 64-expert router with top-4 selection and one shared expert. Router
selection uses sigmoid scores plus the selection correction buffer for expert
choice; the selected weights come from the uncorrected sigmoid scores, are
normalized with the source epsilon, and are multiplied by the configured 1.8
routed scaling factor. Every routed expert has a stable per-layer/per-expert
identity and a symbolic dispatch, SwiGLU computation, and weighted contribution
to the scatter sum. The shared expert uses the configured 1,536-wide MLP and
is added independently to the routed sum.

Transformers stores routed expert weights as stacked rank-3 `gate_up_proj` and
`down_proj` parameters. Each logical per-expert gate, up, and down binding is a
fused region that names its exact expert index and row/axis slice when the
inventory exposes those stacked tensors. When the admitted inventory exposes
individual logical expert matrices, each binds to its exact expert name instead.
All 64 expert identities remain in the graph. The accepted compressed-tensors
W4A16 matrix groups are retained as packed weight, scale, and logical-shape
storage records where admitted; they are not neural operations and do not gain a
new decoder here. Their inspection remains unavailable in this architecture
description.

## Bounds and tests

Selection rejects changed expert/top-k counts, latent ranks/head dimensions,
layer policy, router normalization/scaling, unsupported top-k methods, changed
RoPE treatment, tied embeddings, contradictory derived dimensions, and unknown
configuration fields. Unrecognized or missing parameter storage produces a
localized partial graph. The only extra-layer storage prefix ignored is
`model.layers.47.*`, as explicitly handled by the pinned implementation and
only under the checked `num_nextn_predict_layers: 1` reference.

The focused graph tests use the pinned configuration fixture plus an
independently authored logical-parameter geometry oracle. They check full layer
and expert identities, MLA branches, dense/MoE order, routing and aggregation,
the no-nextn execution path, compressed storage metadata, graph closure and
the response budget. Tests guard imports and network sockets so graph
construction cannot load a model, call model execution, trace it, or connect to
the network. Actual local checkpoint admission and integrated acceptance remain
the responsibility of issue #178; this description does not claim that those
large weights were opened or validated here.
