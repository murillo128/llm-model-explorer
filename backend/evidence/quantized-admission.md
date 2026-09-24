# Quantized checkpoint admission

The catalogue now separates complete physical Safetensors descriptors from native
logical tensor bindings. Physical records contain name, dtype, shape, element and
byte counts, shard and absolute file offset; they have no numeric tensor ID.
`CatalogueEntry.physical_tensors()` and `ModelSource.physical_tensors()` expose
that guarded inventory to backend analysis. `ModelSource.configuration()` remains
a bounded, snapshot-checked read. Processor JSON remains a fingerprinted local
asset; it is not executed. Filesystem protections and hashing are unchanged.

Admission recognizes the reviewed Qwen3 GPTQ Int4, Qwen3.5 ModelOpt NVFP4, and
pinned bitsandbytes NF4/double-quantized configuration/layout combinations.
Unknown options/encodings and incomplete or inconsistent physical groups fail
admission. Validation does not check calibration quality or promise other
GPTQ/NVFP4/bitsandbytes formats.

For these encodings, complete F32/F16/BF16 `weight` and rank-one `bias`, `A_log`,
and `dt_bias` fields retain their physical shape as their full logical shape.
Packed groups and all their auxiliary scales are excluded before native binding;
unknown native fields remain physical metadata only. These are storage-role
rules, not semantic graph parameter records. They do not infer an alias, split a
fused weight, or invent a region endpoint. Generic higher-rank native tensor data
continues to work. Unquantized inventories retain all native records, including
buffers, scalars, and names that resemble quantization auxiliaries.

Unknown/orphan physical records produce `coverage: partial` and a safe exclusion
diagnostic; fully accounted inventories can report complete coverage. Packed
companions never receive logical IDs and return `tensor_not_found` on
data/statistics/distributions and internal row access. Embedding resolution
accepts verified native Llama input tables and the admitted NF4 table through
shared logical row access; GPTQ/NVFP4 tables and V-JEPA remain unsupported there.

## Reviewed metadata and fixture provenance

For the following three earlier references, only JSON metadata and the
Safetensors length prefix/header were fetched. No weight payload, decoder, or
remote model-loading code was used for those metadata checks. NF4 range evidence
uses one complete saved projection from its pinned reference and is recorded
separately in [the SmolLM2 NF4 comparison](bnb-nf4-smollm2-reference.md).
The following exact upstream revisions were inspected:

| Reference | Revision | Header bytes (excluding 8-byte prefix) | Physical / native logical records |
| --- | --- | ---: | ---: |
| [JunHowie/Qwen3-0.6B-GPTQ-Int4](https://huggingface.co/JunHowie/Qwen3-0.6B-GPTQ-Int4/tree/b9d87006067b0c0c2dea836370d6288e14f112ab) | `b9d87006067b0c0c2dea836370d6288e14f112ab` | 98,392 | 898 / 114 |
| [AxionML/Qwen3.5-0.8B-NVFP4](https://huggingface.co/AxionML/Qwen3.5-0.8B-NVFP4/tree/2ac1e750cda67cc8538d731f6216f77b9c3a6f72) | `2ac1e750cda67cc8538d731f6216f77b9c3a6f72` | 128,352 | 1,046 / 302 |
| [facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/tree/b3c1679b7c34d3255ef3547f27c7b226aefab26f) | `b3c1679b7c34d3255ef3547f27c7b226aefab26f` | 62,544 | 587 / 587 |

The counts above come from applying `logical_locations` to the captured complete
headers. They are metadata checks, **not actual checkpoint acceptance**. No full
checkpoint was installed for this validation; integrated acceptance must record
actual local asset bytes and validate complete reference checkpoints.

`tests/fixtures/quantized-configs.json` preserves the selected configuration
fields and source revisions. `tests/test_quantized_models.py` builds tiny synthetic
payloads with the same layout relationships:

- GPTQ: Int4 values packed along the input axis into I32 `qweight` of shape
  `[input/8, output]`; I32 `qzeros` `[input/128, output/8]`; F16 `scales`
  `[input/128, output]`; I32 `g_idx` `[input]`. Reviewed Qwen3 layer-0 `q_proj`
  shapes are `[128,2048]`, `[8,256]`, `[8,2048]`, `[1024]`. Fixture input/output
  dimensions are reduced to 128/8 while retaining group size 128.
- NVFP4: U8 `weight` `[output,input/2]`; F8_E4M3 `weight_scale`
  `[output,input/16]`; scalar F32 `weight_scale_2` and `input_scale`.
  Reviewed language layer-0 `linear_attn.in_proj_a` shapes are `[16,512]`,
  `[16,64]`, `[]`, `[]`. Fixture input/output dimensions are reduced to 16/2.
  Config declares ModelOpt NVFP4 with group size 16 for weights/activations;
  the inspected `hf_quant_config.json` agrees with these settings.
- V-JEPA: native encoder and predictor tensors, including rank-five weights,
  with processor metadata and no text tokenizer or alternative `original` weights.
  The native path requires neither a language architecture nor a tokenizer.

## Validation

Using Python 3.12 and the repository-pinned CPU dependencies:

```sh
cd backend
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest -q
```

Results: lint, formatting, and typing pass; **628 passed, 19 skipped** (optional
CUDA tests; no CUDA device). Existing Starlette/httpx deprecation warnings remain.
The local run reused an existing environment with the pinned dependency versions
and set `PYTHONPATH=src` to target this worktree.

Coverage includes indexed and non-indexed groups, companion dtype/shape errors,
missing/duplicate shards, index disagreement, traversal and out-of-root symlinks,
physical widths/gaps/overlaps, unsafe dimensions, unsupported configuration and
orphan scales, conflicting native/packed weights, explicit partial/empty logical
inventories, relocation and changed snapshots, metadata-only discovery, native
IDs and exact F32/F16/BF16 stream bytes, warm conversion reads, higher-rank native
access, and unsupported input-embedding requests. Existing baseline tests also
check all finite half-precision patterns and F32 nonfinite/signed-zero bits.

`api/validate_contract.py` passes: 283 references, 88 instance cases, 112
architecture cases, and 76 reproducible wire fixtures. Four actual TestClient
responses (GPTQ/NVFP4, with and without native tensors) were separately checked
against the published `TensorInventory` schema and its cross-field semantics.
No API schema or generated binding change is required.
