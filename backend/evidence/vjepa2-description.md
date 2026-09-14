# V-JEPA 2 description evidence

## Source and checkpoint provenance

The packaged `transformers-vjepa2` description revision 1 was reviewed against
Transformers v4.57.6, commit `753d61104116eefc8ffc977327b441ee0c8d599f`:

- [configuration_vjepa2.py](https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/vjepa2/configuration_vjepa2.py)
  owns the normalized defaults and their meaning.
- [modeling_vjepa2.py](https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/vjepa2/modeling_vjepa2.py)
  owns the implementation relationships reviewed below. This is source inspection,
  not execution, tracing, or a training-paper reconstruction. Hugging Face's source
  is Apache-2.0; this description encodes reviewed mathematical relationships and
  does not vendor its numerical implementation.
- [Selected checkpoint](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f),
  revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`, supplies the exact config and
  Safetensors metadata. The checkpoint declares `transformers_version: 4.53.0.dev0`;
  that declaration is distinct from the implementation revision reviewed here.

`tests/fixtures/vjepa2-reference.json` retains that configuration and all 587
observed tensor names, dtypes and shapes, independently of the description's
parameter generator. Its SHA-256 records identify both source files, configuration
bytes, and the raw 62,544-byte Safetensors header. Development read only config and
header ranges (8 length bytes plus header); no checkpoint payload was downloaded.
No upstream `original/model.pth` was needed.

## Reviewed relationships

| Source implementation | Description and verification |
| --- | --- |
| `VJEPA2Embeddings`, `VJEPA2PatchEmbeddings3D` | Symbolic B,F,C,H,W input, permutation, conditional frame repetition for a clip shorter than the tubelet, Conv3d with kernel equal to stride, flatten of patch axes and sequence transpose. Full patch weight is `[1024,3,2,16,16]`; bias is `[1024]`. |
| `VJEPA2Encoder.forward` | All 24 actual blocks run before final LayerNorm. The encoder receives no context-position selection. Raster positions are generated from patch sequence length. |
| `VJEPA2RopeAttention` | Separate Q/K/V projections and biases, split/transpose heads, temporal/height/width rotary treatment of Q and K, scaled QK product, noncausal softmax, weighted V, head merge, biased output projection. Each rotated axis has `2 * floor(floor(head_dim/3)/2)` channels; the remainder is unchanged. Position decoding uses the configured crop/patch grid. There is no learned positional tensor to bind. |
| `VJEPA2Layer` and `VJEPA2MLP` | Pre-attention norm, attention residual, pre-MLP norm, biased up projection, GELU, biased down projection and second residual; encoder/predictor widths, heads and intermediate sizes remain distinct. |
| `VJEPA2Predictor.forward` | Context selection occurs on final encoder representations, then projection to predictor width. Context/target position indices feed sorting and rotary attention. All 12 predictor instances remain present. Final norm precedes inverse sorting, target-suffix extraction and projection back to encoder width. |
| `VJEPA2PredictorEmbeddings` | Existing mask-token bank `[10,1,1,384]`; select default bank index `1 % num_mask_tokens`, repeat/gather target tokens, concatenate context then target tokens and their positions. No vocabulary or text tokenizer. |
| `VJEPA2Model.forward` | Full encoder representations, context-selected representations, target-selected encoder representations and predicted target representations. The target is gathered from the same encoder, not a fabricated teacher. |

The mathematical graph describes the reviewed evaluation path with the predictor
included, one context and one target mask entry per batch item, and no optional
attention-head masks. Positions and input sizes remain symbolic; neither masks nor
video are supplied. These conditions are recorded on both stack groups. The
source's list-of-mask repetition/batching and optional head-mask execution paths
are not claimed as additional supported runtime modes. There is no execution
endpoint in this increment. Nonzero dropout/stochastic-depth configurations and
unreviewed activation or structural fields are rejected, rather than presented
with a misleading complete default graph.

Known configuration defaults are normalized only from the reviewed config class.
Configured counts are checked against storage, not fixed to 24/12. Extra tensors,
wrong geometry, unsupported dtypes, contradictory image/crop size, unreviewed
classes/options, and custom-code mappings fail selection. Missing expected tensors
retain localized unresolved bindings and partial coverage, including an absent
predictor; they never pass complete reference acceptance. Complete architecture
and native numeric availability remain independent.

Native rank-1/rank-2 parameters resolve to existing inventory identities. The patch
weight and mask bank preserve their original higher ranks with `unsupported_rank`
inspection, without a flattened substitute or actionable modal tensor ID.

## Evidence and reproducibility

`vjepa2-tiny.json` is a distributable concrete metadata fixture reduced from the
observed names/geometries: two encoder blocks, one predictor block, widths 12/6,
heads 2/1, and small patch sizes. Tests write deterministic complete local weights
from this fixture, so startup/UI acceptance consumers can reuse it without model
or processor construction. No tokenizer assets are included.

`test_vjepa2_architecture.py` checks source-derived dependency assertions, every
reference parameter and repeated instance, native identities, symbolic dimensions,
3D rotary positioning, absent positional edges, missing predictor/weights,
contradictory configurations and storage, default normalization and the reviewed
bias-free Q/K/V option. Expected topology is hand asserted from the reviewed
source; it is not produced by the implementation under test. The reference oracle
is captured upstream storage; it is not generated by `parameter_shapes`.

Read-only local fixture analysis guards model construction/calls, processors,
network, tracing, full tensor loaders and ModelSource numeric reads. A subprocess
rejects even imports of torch, Transformers and Safetensors by the graph core and
V-JEPA description. Separate numeric tests verify existing bounded source reads
for a native predictor matrix and a concrete encoder layer's vector. Fingerprinting
continues to use the previously tested bounded source path before analysis.

From `backend/`, using the pinned CPU development environment:

```sh
uv run --locked pytest tests/test_vjepa2_architecture.py tests/test_architecture_analysis.py
uv run --locked python scripts/generate_architecture_records.py --check
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest
```

This establishes fixture/source/metadata conformance only. Full-local-checkpoint
acceptance (local content fingerprint, complete weight bytes, startup/cache/HTTP,
real browser layout and memory measurements) remains the later reference gate.
The remote header is not a substitute for admitting a complete local checkpoint,
and no inference or numerical equivalence is claimed.

Local validation passed 33 focused description tests and the full backend suite
(786 passed, 19 CUDA-only skips) on Python 3.12.14 with pinned CPU dependencies.
Lint, formatting, strict typing, generated-record drift checking, independent API
conformance (112 architecture cases), source/wheel builds and registration from the
installed wheel also passed. The full reference metadata produces 782 nodes,
1,079 edges and 587 parameters; its serialized graph is 1,818,621 bytes with this
fixture identity. All 585 rank-1/rank-2 parameters bind to native inventory IDs;
the two higher-rank parameters retain metadata with unavailable modal inspection.
