# Kimi Linear static architecture description

## Pinned sources

The structural reference is `cyankiwi/Kimi-Linear-48B-A3B-Instruct-AWQ-4bit`
at revision `5d029d1844aa64ec302e14466be7d0353c6e697f`. The downloaded source
`config.json` SHA-256 is
`5ae21d4b11c57634354a8dacca93062347882f3cc727a9f69e9249e4fdc45eb8`; the
source is
[`config.json`](https://huggingface.co/cyankiwi/Kimi-Linear-48B-A3B-Instruct-AWQ-4bit/blob/5d029d1844aa64ec302e14466be7d0353c6e697f/config.json).
The
reviewed structural projection and independently recorded expectations are in
[`tests/fixtures/kimi-linear-reference.json`](../tests/fixtures/kimi-linear-reference.json)
(SHA-256 `b14cc9d914e00fc85d2f2c2a2e2862e9b773e8b233f91ad93a5792a7b2187f9d`).

The implementation source is Moonshot's Kimi Linear repository at revision
`d64a5299ded33ab2609617e05f6bd2cf9b6eef35`, specifically
[`configuration_kimi.py`](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/d64a5299ded33ab2609617e05f6bd2cf9b6eef35/configuration_kimi.py)
and
[`modeling_kimi.py`](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/d64a5299ded33ab2609617e05f6bd2cf9b6eef35/modeling_kimi.py).
Their downloaded file SHA-256 values are `79422aca3ee6c89d201e0c15c4c9a6db517ba83d87ecdc4e41fa0f71297238d9`
for `configuration_kimi.py` and `d79b365e37378881b9f1585007a56e236ca27a414920943cb85d1dacb75dda99`
for `modeling_kimi.py`.
Both were read as inert text. The analyzer does not import or execute checkpoint
or Transformers code.

The target model index and one bounded 8 MiB HTTP range of the first Safetensors
shard were inspected only to confirm names and header geometry. The range
contained the Safetensors header and no tensor payload bytes. Selected records
confirm KDA convolution weights `[4096, 1, 4]`, `A_log` `[1, 1, 32, 1]`,
`dt_bias` `[4096]`, a full-attention query weight `[6144, 2304]`, and a packed
expert `w1` weight `[1024, 288]` with BF16 scales `[1024, 72]` and logical shape
`[2]`. This is metadata evidence only; it is not actual checkpoint acceptance.

## Evaluated path

The strict configuration selects exactly 27 ordered decoder layers: KDA at
one-based layers `[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21,
22, 23, 25, 26]`, with MLA at the remaining seven. Layer 0 uses the dense MLP;
layers 1–26 use the configured MoE block. A changed layer list, KDA geometry,
latent-attention geometry, routing policy, or unknown structural field fails
selection.

The packaged Kimi description is revision `2`; this revision records the
source-verified KDA padding/compaction dependencies and invalidates graphs made
from the earlier draft description.

Each KDA group retains q/k/v projections and causal short convolutions, the
fused decay gate from `f_a_proj`, `f_b_proj`, `A_log` and `dt_bias`, the sigmoid
beta projection, normalized q/k delta-state update, gated RMSNorm, and output
projection. It exposes three symbolic convolution-history states and the
symbolic recurrent matrix state. When a padding mask is supplied, the reviewed
forward path compacts valid token rows before every KDA projection, convolution,
and recurrent update. The graph carries symbolic original-token indices and
per-example cumulative sequence offsets through those operations, then scatters
the projected output back to `[B, S, hidden]` with zero-filled padded positions.
This preserves convolution and recurrent sequence boundaries without evaluating
token values, unrolling sequence positions, or providing captured state.

Each MLA group retains the full q projection, compressed KV projection and
normalization, KV expansion, split/recombined query and key subspaces, causal
attention scores, scaling, mask, softmax, value aggregation and output
projection. The reviewed `mla_use_nope=true` path leaves the configured 64-wide
query/key slices unrotated and broadcasts the key slice across 32 heads. KV
history is symbolic and per layer.

MoE router logits are computed from float32 inputs and weights, then sigmoid
scores are used for routing. The one configured expert group contains all 256
experts; the source's group score is the sum of the two highest corrected
scores. Expert top-8 selection ranks the correction-adjusted scores. The source
adds the correction in-place through `scores_for_choice`, which is a view of
the sigmoid score tensor; its later gather therefore uses the mutated,
correction-adjusted sigmoid scores. Those selected weights are renormalized
and multiplied by `2.446`. A symbolic dispatch/compute/scatter operation
retains the hidden-state, selected-ID and selected-weight inputs and describes
the per-expert token partition, weighted SwiGLU computation and source-position
scatter without materializing 256 sets of redundant edge records. Every one of
the 256 expert identities in every MoE layer has a distinct repeated group and
exact `w1`, `w3`, and `w2` parameter bindings. The grouped operation refers to
that repeated expert set; each expert group carries the source provenance and
its own three parameter IDs. The one shared SwiGLU expert is evaluated
independently and added to the routed sum.

Every decoder layer includes both RMSNorms and both residual additions. The
root keeps declared token, mask, and symbolic prior/next state values as
container ports; final RMSNorm and the CausalLM head produce logits. No token,
runtime route, activation, recurrent state, or KV cache value is invented.

## Physical storage and limits

The architecture uses logical parameter identities independently from physical
format. Admitted compressed-tensors W4A16 records remain packed weight, scale,
and logical-shape storage metadata; no quantization auxiliary becomes a graph
operation, and this change adds no decoder. If storage is missing or has
incompatible geometry, the corresponding parameter is unresolved and the graph
is partial. All configured layer and expert identities are retained under the
32 MiB graph budget; construction returns `unsupported_size` rather than
truncating if a future representation exceeds that limit.

The response-budget fixture models all 19,968 routed-expert matrices as
compressed-tensors packed groups and keeps other parameters native. The complete
metadata-only graph remains below 32 MiB; no model payloads are read for this
simulation.

Focused tests check the exact layer order and independent logical parameter
geometry, all 27 layer identities, every expert in all 26 MoE layers, KDA state
ports, MLA latent-attention paths, routing, residuals, storage bindings,
negative configuration fixtures, graph closure and response size. They guard
network connections, checkpoint imports, model construction, `forward()` and
`generate()` calls, tracing, and `torch.load`. Actual quantized-checkpoint/browser validation
remains assigned to the integrated-reference issue; this description does not
claim that the model weights were opened or validated.
