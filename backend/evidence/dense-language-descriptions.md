# Dense language description evidence

## Implementation and source review

`architecture_analysis/dense.py` packages two descriptions, `qwen3-dense` and
`smollm2-dense`, revision `1`. Call `register_dense_descriptions(registry)` from
the startup composition owner. The registry and guarded input seam are unchanged;
this increment introduces no checkpoint parsing, HTTP, cache, or execution path.

The primary reviewed source is Transformers v4.57.6, exact revision
[`753d61104116eefc8ffc977327b441ee0c8d599f`](https://github.com/huggingface/transformers/tree/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models).
Review covered `configuration_qwen3.py`, `modeling_qwen3.py`,
`configuration_llama.py`, `modeling_llama.py`, and the embedding tying helpers in
`modeling_utils.py`. The source revision participates in producer/graph identity;
configuration declarations and observed storage have separate provenance records.

The checkpoint-declared library revisions were also checked:

- Qwen3: v4.55.4,
  [`d79b2d981f28b2730d402244ac3c2e9a8c054eee`](https://github.com/huggingface/transformers/tree/d79b2d981f28b2730d402244ac3c2e9a8c054eee/src/transformers/models/qwen3).
  The selected full-attention operations, Q/K normalization, biases, projection
  geometry, MLP branches, and tying agree with the primary reviewed description.
- SmolLM2: v4.40.1,
  [`9fe3f585bb4ea29f209dc705d269fbe292e1128f`](https://github.com/huggingface/transformers/tree/9fe3f585bb4ea29f209dc705d269fbe292e1128f/src/transformers/models/llama).
  Its configured `pretraining_tp=1`, bias-free MLP, hidden/head geometry,
  split-half rotary convention, and RMSNorm agree with the described path.
  The newer Llama implementation's explicit head dimension and optional MLP
  bias are separately tested small variants, not claims about the old library.

These are original static semantic descriptions based on source inspection;
no upstream implementation is vendored or executed. Attribution: the Qwen
implementation credits the Qwen team, Alibaba Group and HuggingFace (2025);
Llama credits EleutherAI and HuggingFace (2022), with GPT-NeoX/OPT lineage and
Meta architectural modifications. The reviewed Transformers files use
[Apache License 2.0](https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/LICENSE).
The parameter names/configuration facts retain their upstream identity.

## Bounded semantic coverage

The shared fragment follows the source's common mathematics: embedding, ordered
pre-normalized decoder layers, separate Q/K/V projections, head reshape and
transpose, rotary position treatment, GQA repetition, QK product and scaling,
causal masking, softmax, value product, output head merge/projection, two residual
additions, SiLU gate multiplied with the independent up branch, down projection,
final RMSNorm, and vocabulary projection. Every layer instance has its own
parameters, children, edges, and repetition identity. Module references identify
actual modules; synthetic mathematical operations do not invent module resources.

Qwen3 normalizes Q/K over **head_dim** before transpose/rotary; its attention
width need not equal hidden width. SmolLM2 does not have those two normalizations.
Both use direct RMSNorm scale, without a `1 + weight` offset; epsilon comes from
configuration (selected Qwen3 `1e-6`, SmolLM2 `1e-5`). Qwen3 MLP biases are not
supported because its reviewed constructor has none. Reviewed Llama MLP biases
and attention biases have explicit parameter bindings when enabled.

The graph describes evaluation-mode full-sequence mathematics. Batch and sequence
length are symbolic; position IDs and the causal mask are inputs. Dropout is inactive
in evaluation mode. There is no prompt, calculated activation, captured KV cache,
training loss, or generation timeline. A tokenizer appears only as a capability
context node when the caller reports that capability.

Configuration selection uses the reviewed defaults and an explicit field set.
Qwen's omitted head dimension defaults to 128 and omitted KV-head count to 32;
an explicit null KV count uses the query-head count. Llama's omitted/null head
size uses hidden size divided by query heads and its omitted/null KV count uses
query heads. Both default to untied embeddings and epsilon `1e-6`. The checked
field defaults are in `dense_config.py`; declared dimensions must be positive safe
integers, GQA must divide evenly, and head dimensions must be even for split-half
rotary. Known nonstructural metadata does not change the selected mathematics.

Scaled/interleaved rotary, sliding/hybrid attention, unknown activations/options,
unknown classes, MoE, unreviewed quantization, and contradictory configuration
return `unsupported_architecture`. Unsupported options are not substituted with a
Llama template. Extra unexplained storage, missing parameters, incompatible
geometry/dtypes, and incomplete packed groups prevent complete coverage.

Native parameter inspection requires the complete admitted tensor's exact shape,
dtype, and ID. GPTQ binding checks the logical `[output,input]` geometry against
all four physical companions and preserves their roles; it exposes no numeric ID
or decoder. Tied embedding/output names become aliases when configuration declares
tying and only one stored table exists. Either saved name can be retained. If
both names have stored tensors, metadata cannot prove their numerical identity:
the graph preserves both observed bindings and reports partial `unverified_tie`.
Absent targets remain unresolved and never manufacture an inspection endpoint.

## Selected reference metadata evidence (not full-checkpoint acceptance)

Only configuration JSON and Safetensors headers were read remotely during
source review. No weights were downloaded or loaded. The committed compact oracle
`tests/fixtures/dense-reference-metadata.json` records the exact configuration,
global storage, and repeated layer storage. Before reduction, each actual layer's
header entries were compared against the retained pattern; no layer exceptions
were found. Test fixture expansion is independent of the production description.

| Reference | Checkpoint revision | Layers | Physical / native records | Graph nodes / edges / parameters |
| --- | --- | ---: | ---: | ---: |
| [JunHowie/Qwen3-0.6B-GPTQ-Int4](https://huggingface.co/JunHowie/Qwen3-0.6B-GPTQ-Int4/tree/b9d87006067b0c0c2dea836370d6288e14f112ab) | `b9d87006067b0c0c2dea836370d6288e14f112ab` | 28 | 898 / 114 | 962 / 1350 / 311 |
| [HuggingFaceTB/SmolLM2-135M Base](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2) | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | 30 | 272 / 272 | 970 / 1386 / 273 |

Counts exclude the optional tokenizer context node. Both metadata graphs have
complete semantic coverage, account for every physical record, and serialize to
less than 3 MB. Both headers retain only `model.embed_tokens.weight`; the output
parameter aliases that native table. The extra logical parameter is that alias.
Qwen3 has 196 quantized projection parameters, with 784 packed/auxiliary physical
records. Numeric inventory is independently partial for that checkpoint.

This evidence is **not actual full-local-checkpoint acceptance**. That belongs to
integrated acceptance with complete approved local assets, fingerprints, exact
local bytes, startup/cache timings, environment/memory measurements, and browser
native-value inspection. No extra unquantized Qwen checkpoint is required here.
Neither source inspection nor these fixtures claims inference correctness or
numerical equivalence of the quantization.

## Reproduction and checks

Using Python 3.12 and the repository-pinned backend dependencies, target the
current worktree with `PYTHONPATH=src` when reusing an existing environment:

```sh
cd backend
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest -q
cd ..
python api/validate_contract.py
```

The focused suite has 44 passing cases. It verifies full reference layer counts
and order, all physical correspondences, normalization differences, branch and
residual edges, GQA/reshape geometry, per-instance parameter attachment, native
and GPTQ availability, tied/untied/reverse-tied/missing/broken cases, biases,
unsupported structural variants, checked defaults, incremental size limits,
independent API semantic validation, and graph serialization/validation.

Reduced local fixtures use asymmetric, exactly representable values and compare
all available tensor data against their independent value sequence. Static tests
use read-only checkpoint directories and intercept network, model construction,
forward calls, generation, tracing, safetensors loading, tensor reads, and CUDA
entry points. Package-import guards ensure registration adds no numerical-library
dependency. Full validation results are recorded in the PR.

Local validation on the issue candidate: **797 passed, 19 skipped** (CUDA device
unavailable); Ruff lint/format and strict mypy passed. The independent API check
passed with 283 resolved references, 88 instance cases, 112 architecture cases,
and 76 reproducible wire fixtures. Existing Starlette/httpx deprecation warnings
remain. The run reused `/tmp/issue-43-venv` with the pinned package versions and
explicit `PYTHONPATH=src`; it did not modify dependency versions.
