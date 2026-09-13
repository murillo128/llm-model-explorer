# Input embedding lookup evidence

## Reproduction

From `backend/` with the locked development environment:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest
uv run --locked python scripts/smoke_input_embeddings.py /local/models/SmolLM2-135M
```

From the repository root, the independent API environment runs
`python api/validate_contract.py`; the backend environment runs
`python -m pytest acceptance` for TCP product acceptance. Application CI also
runs the UI and production-browser gates.

## Observed coverage

Local Python 3.12 validation: Ruff lint/format and strict mypy passed; the full
backend suite passed 540 tests with 16 optional CUDA skips. Independent API
validation passed (88 instance cases, 76 reproducible wire fixtures). TCP
acceptance passed 11 tests with two optional capability skips.

`tests/test_embeddings.py` consumes the accepted asymmetric input table and
request fixtures. F32, F16 and BF16 results preserve exact values, row order,
duplicates, first/last vocabulary rows, one token and empty input. It covers
invalid types/IDs, unknown sessions, malformed JSON, unsupported/ambiguous
architecture, inconsistent config/table dimensions, indexed shards, CORS,
metadata limits and unsafe computed output sizes.

Read instrumentation starts **after session pinning**, whose existing content
fingerprint intentionally hashes all assets in bounded reads. Resolver work
performs no numeric reads or allocations. Lookup never invokes `iter_tensor`;
physical reads and native allocations stay within 65,536 elements, including
rows wider than that bound. For `[19, 0, 1, 1, 1]` at hidden size 3, only four
physical rows are read: the contiguous `[0, 1]` span is coalesced and the last
repeated row uses the preceding block. Empty input has no numeric read/allocation.
The tests verify no cache files after successful or failed requests.

Real TCP barriers establish DATA delivery before completion and responsive
metadata and real tokenization while an embedding read is held in a worker.
Operation cancellation, session deletion, disconnect and partial-send failure
close readers and release registries while another consumer completes.
Mutation between blocks and during a physical read is rejected; memory failures
and truncated output never end with COMPLETE or reveal private source names.
Shared `embeddings.json` wire and metadata cases also exercise the common LMEX
writer and strict schema validators.

## Local reference smoke

The supplied `HuggingFaceTB/SmolLM2-135M` Base checkpoint was available. The
optional script compared endpoint bytes with independent `safetensors.safe_open`
row slices from the same local weights, without downloading or loading a model.

- Token IDs: `[49151, 0, 128, 1, 128]`.
- Shape: `[5, 576]`; float32 payload: 11,520 bytes.
- Exact byte equality: PASS.
- Payload SHA-256: `238e4ae3d863f067f2d276cf8995b621cda8f1aaf9dfe48ef4a37bbe58fd3eb3`.
- Cache entries: zero.

CPU is the validated compute environment. CUDA execution is unnecessary for
this row IO capability; existing optional CUDA analysis tests skip when absent.
The resolver deliberately supports the explicit Llama causal-LM checkpoint
layout only; other architectures require a separate accepted resolver.
