# Admitted packed tensor decoding

## Numeric reference and ownership

`quantized_decoding.decode_range` reads a bounded flat range of a logical
`[output, input]` matrix and returns owned CPU float32 values. It accepts only
the two storage layouts validated by checkpoint admission. Configuration and
structural admission are unchanged.

The independent scalar oracles in `tests/quantized_oracles.py` use `struct` and
scalar arithmetic, without production conversion helpers or PyTorch lookup
tables. They record these exact reviewed implementations:

- [AutoGPTQ qlinear_cuda.py at 9f7d370](https://github.com/AutoGPTQ/AutoGPTQ/blob/9f7d37072917ab3a7545835f23e808294a542153/auto_gptq/nn_modules/qlinear/qlinear_cuda.py):
  GPTQ v1 low-first Int4 packing, restoration of zero points by adding one
  after extracting the nibble (without modulo wrapping), and stored `g_idx`.
- [GPTQModel at 40759cd](https://github.com/ModelCloud/GPTQModel/blob/40759cdf06c17ea6c57637fa075f8c10547b4a6f/gptqmodel/utils/model.py):
  the selected producer's GPTQ v1 convention. Canonical float32 decoding avoids
  the additional half-precision output rounding of a half-precision matmul path.
- [ModelOpt NVFP4 at 82f1d21](https://github.com/NVIDIA/Model-Optimizer/blob/82f1d216d1a9022e60b8e1143a77f36c00b885a4/modelopt/torch/quantization/qtensor/nvfp4_tensor.py):
  low-first E2M1, E4M3 block scales multiplied by the global weight scale in
  float32 before multiplication by the decoded value. `input_scale` is absent
  from weight dequantization. Standard E2M1 negative zero is preserved; the
  upstream Python lookup collapses it to positive zero, a sign-bit distinction
  called out separately from nonzero numerical agreement.

Production arithmetic and gathering use PyTorch kernels. Each gather coalesces
requested unique storage positions into contiguous file spans and reads them
into owned bytes, then restores requested order with PyTorch. Python loops over
I/O spans only. Read lengths and total copied bytes are proportional to the
requested chunk, including strided GPTQ columns. Snapshot and short-read checks
detect concurrent mutation/truncation without dereferencing live file mappings.
There is no full-checkpoint float32 allocation or new cache product.

Fixture tests cover complete matrices, every nibble position, signed zeros,
zero-point endpoints, multiple/nontrivial repeated group mappings, half and
float32 scales, E4M3 boundaries, arbitrary range alignment, separated companion
shards, invalid group indices, safe ranges, source mutation, and bounded owned
output. Their synthetic bytes are distinct from actual-checkpoint evidence.

The corrected standalone decoder checkpoint passes 34 focused tests, Ruff and
mypy using the locked Python 3.12 CPU environment. The six concurrent-truncation
cases cover both formats before a file read, after a read, and immediately before
the vectorized gather; each yields `model_content_changed` without a process
crash. The earlier live-mapping implementation was rejected at intermediate
review after a SIGBUS reproduction and is replaced by owned file reads.

## Logical integration and actual checkpoints

The inventory binds one logical matrix to each complete validated group; its
companions account for physical storage without becoming fake parameters.
Unknown/orphan records retain a named partial-coverage diagnostic. Tensor data,
ordered row access, statistics and distributions share the same materialization
and existing per-tensor cache product. Native bytes and cache keys are unchanged.
Packed producers have separate versioned recipes. Graph bindings retain packed
provenance and expose actual logical IDs; the analyzer revision invalidates old
graph artifacts with unavailable inspections. All three semantic validators check
logical identity and complete packed geometry. OpenAPI schemas and generated
TypeScript bindings need no shape change.

The installed exact reviewed checkpoints produced these guarded inventories:

| Checkpoint | Physical records | Logical tensors | Decoded matrices | Coverage |
| --- | ---: | ---: | ---: | --- |
| Qwen3 GPTQ Int4 | 898 | 310 | 196 | complete, no diagnostics |
| Qwen3.5 NVFP4 | 1,046 | 488 | 186 | complete, no diagnostics |

[Reference comparison evidence](quantized-reference-comparison.json) records
checkpoint and upstream revisions, content fingerprints, reference-source hashes,
decoder-source hash, exact ranges, decoded digests and comparison metrics. The
offline reproduction script is `scripts/check_quantized_reference.py`; it checks
locally supplied source files and never executes checkpoint or downloaded code.
One complete projection per format plus ranges across other layers were compared
in chunks of at most 65,536 values:

- GPTQ: 1,105,522 float32 values, including the complete `[1024,1024]` layer-0
  key projection, with zero canonical bit mismatches.
- NVFP4: 68,722 float32 values, including the complete `[16,1024]` layer-0
  `in_proj_a`, with zero canonical bit mismatches.

Upstream inference-half rounding and the ModelOpt Python LUT's negative-zero
collapse are measured separately; neither is misreported as canonical equivalence.
These observations establish the tested ranges, not whole-checkpoint inference
correctness or support for additional quantization variants.

Both actual Qwen checkpoints also pass the existing cold/warm TCP architecture
reference checks, with complete inventories and complete graph coverage. The
deterministic production-browser acceptance opens their synthetic packed weights
through the real Matrix Explorer and verifies adjacent exact cell values; existing
native inspection and resource-release checks pass. Actual-checkpoint browser
reference runs remain separate from those deterministic browser results.

## Final local validation

Using the repository-locked Python 3.12 / PyTorch 2.14 CPU and Node 24.14 environments:

- Backend Ruff/format/mypy pass; full backend suite: **1,057 passed, 19 skipped**
  (CUDA unavailable).
- API validation: **283 references, 88 instance cases, 144 architecture cases,
  76 wire fixtures**; regeneration is reproducible. UI generated bindings remain
  unchanged.
- UI API check, typecheck, lint and production build pass; unit tests:
  **664 passed across 28 files**.
- Network acceptance: **19 passed, 6 skipped** (optional reference/CUDA modes).
  Separate manifest-enabled Qwen3 and Qwen3.5 cold/warm architecture checks:
  **2 passed**.
- DPR1 production-browser architecture acceptance: **5 passed, 4 skipped**
  (actual-reference browser mode not enabled). This includes both packed formats,
  native inspection, modal closure and resource release.

Reproduction uses `backend`'s Ruff, format, mypy and pytest gates;
`api/validate_contract.py` with and without `--write`; UI `api:check`, `typecheck`,
`lint`, `test`, `build`; `pytest acceptance`; and
`npm run test:acceptance -- architecture.spec.ts --project=dpr1` under Xvfb.
For the installed reference checks, set `LMEX_ARCHITECTURE_REFERENCES` and run
`pytest acceptance/test_architecture.py -k 'complete_local_architecture_reference and (qwen3 or qwen35)'`.
Use an absolute worktree `PYTHONPATH` when reusing an external Python environment.

Local UI dependencies were reused through temporary symlinks. Vite's runner
config loader and a temporary Vitest filesystem allowlist handled those external
dependencies without changing product configuration. Headed Chromium ran outside
the filesystem sandbox with temporary XDG config/cache directories and the
already installed browser cache. CI uses its normal isolated dependencies.
