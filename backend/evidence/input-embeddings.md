# Input embedding lookup evidence

## Reproduction

Use the locked backend development environment and the current checkout. From
`backend/`, run Ruff lint/format, strict mypy, and:

```sh
python -m pytest tests/test_embeddings.py tests/test_quantized_models.py tests/test_quantized_decoding.py
python scripts/smoke_input_embeddings.py /local/models/CHECKPOINT
```

From the repository root, run `python -m pytest acceptance/test_embeddings.py`
with the backend environment, and `python api/validate_contract.py` with the
independent API environment. From `ui/`, run:

```sh
npm run check
xvfb-run -a npm run test:browser -- tests/tokenizer-embeddings.spec.ts
xvfb-run -a npm run test:acceptance -- --grep 'real (qwen|A→B→A|ordered embeddings)' --output=/tmp/embedding-production-results
```

Use isolated `UI_TEST_PORT` values on a shared host. The production acceptance
suite requires working X11 and Chromium WebGL; the execution sandbox did not
provide these, so browser validation used the host. Separate output directories
avoid another Playwright run deleting in-flight artifacts.

## Resolver and delivery coverage

The three explicit mappings cover Llama/SmolLM2, Qwen3, and the Qwen3.5 language
component. Deterministic F32/F16/BF16 fixtures exercise ordered/duplicate IDs,
first/last rows, empty requests, invalid IDs, and indexed shards. Qwen3.5 tests
supply conflicting top-level and vision dimensions to prove that only nested
`text_config` dimensions control the text table. Missing/wrong input names,
output-head-only storage (even with tied-weight metadata), vision lookalikes,
ambiguous architectures, duplicate physical/logical candidates, custom mappings,
invalid dimensions, and V-JEPA fail closed.

Admitted GPTQ and NVFP4 checkpoint fixtures retain actionable native input tables.
After integration of shared packed row access, additional synthetic embedding
tables compare exact GPTQ/NVFP4 response bytes with the independent scalar oracles
in `tests/quantized_oracles.py`. The tokenizer introduces no decoder.

Read instrumentation begins after session pinning; pinning intentionally hashes
complete assets in bounded blocks. Resolution performs no numeric reads or
allocations. Native row lookup never calls `iter_tensor`; each read/allocation
stays within `CHUNK_ELEMENTS`, including rows wider than a chunk. Empty input has
no numeric read/allocation. Tests retain cancellation, session deletion,
disconnect, partial-send failure, independent consumers, content mutation during
and between reads, memory failure, and truncated-stream behavior. No numeric
artifact is published by embedding lookup.

The existing UI classifier distinguishes unsupported capability from HTTP,
transport, protocol, and rendering failures. Controller tests and component
browser regressions retain cancellation, prompt/session generation fencing,
stale matrices, and failed replacement disposal. Production-browser Qwen3 and
Qwen3.5 fixtures now render exact linked rows, preserve duplicate sequence
positions, and recover after switching through a genuinely unsupported model.

## Local reference results (2026-09-15)

All three operator-provisioned local checkpoints were available. The smoke script
compared endpoint float32 bytes with independent `safetensors.safe_open` slices,
without downloading assets or materializing the full table. IDs were
`[vocab_size - 1, 0, 128, 1, 128]`.

| Local checkpoint | Input table shape | Delivered shape | Payload bytes | Exact bytes |
| --- | --- | --- | --- | --- |
| SmolLM2-135M Base | `[49152, 576]` | `[5, 576]` | 11,520 | PASS |
| Qwen3-0.6B-GPTQ-Int4 | `[151936, 1024]` | `[5, 1024]` | 20,480 | PASS |
| Qwen3.5-0.8B-NVFP4 | `[248320, 1024]` | `[5, 1024]` | 20,480 | PASS |

Payload SHA-256, in the same order:

- `238e4ae3d863f067f2d276cf8995b621cda8f1aaf9dfe48ef4a37bbe58fd3eb3`
- `c613f42e432cfe7ee2f48e1c850b402aaaa1736fc0f33c58d432c16ba7bee8ad`
- `496cf066c353f8def4fc8ad589dea125a1e2b3aa63c863d3717ba83652e9b84d`

Each reference run produced zero numeric cache entries. Structured architecture
artifacts created at startup are counted separately. These checks establish the
native input tables in the actual quantized checkpoints; synthetic packed-table
checks establish shared row-path reuse. They do not claim support for arbitrary
architectures or encodings. CPU is sufficient for this row I/O capability.

## Validation observed

- Backend Ruff lint/format and strict mypy: PASS.
- Independent API contract: 283 references, 88 instance cases, 112 architecture
  cases, and 76 reproducible wire fixtures: PASS.
- UI typecheck, lint, unit suite, and production build: PASS.
- Tokenizer component browser regression: 26 passed (desktop and narrow).
- Post-integration production embedding acceptance: 8 passed (DPR 1 and DPR 2),
  covering both Qwen mappings, ordered progressive rows, and generation changes.
- Post-integration TCP embedding acceptance: 4 passed.

Full exact-head application, backend, and UI regression gates are recorded by
the linked PR checks; local focused coverage is described above. Optional CUDA
and absent-reference checks must retain their explicit skips.
