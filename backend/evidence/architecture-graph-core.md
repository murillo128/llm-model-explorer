# Static architecture graph core evidence

## Boundary and oracle

The core consumes guarded metadata from `ModelSource`; family semantics, file
parsing, startup/cache/HTTP lifecycle and frontend layout are separate consumers.
There are no built-in family adapters or model acceptance claims in this change.

Expected graph records and contextual mutations come from
`api/fixtures/architecture.json`, published independently of this implementation
at `d65c92a5ed48665147c14d5aee5b302fbec2a568`. Its exact-head independent
[audit](https://github.com/murillo128/llm-model-explorer/pull/93#issuecomment-5668243680)
returned PASS and specifically covered graph/reference/alias closure,
containment/repetition, directional ports, symbols and native inspection geometry.
The core reconstructs that full expected instance graph and compares records; it
does not generate its expected graph using the production builder. Tests consume
all applicable graph-level mutations; response envelopes and session matching are
outside this package. Two positive cases additionally supply physical descriptors
because the API fixture context only supplies numeric inventory: the rank-1 case
keeps alias storage consistent with the new native descriptor, and the zero-size
case supplies its declared physical geometry.

## Checks and properties

- Generated strict Pydantic records are checked byte-for-byte against current
  OpenAPI generation. Closed unions, required/optional fields, bounds and shape
  discriminators are exercised by the shared negative cases.
- Record identity depends on semantic keys, graph identity, and record category;
  graph identity includes content fingerprint, scope, packaged description,
  reviewed implementation, analyzer and schema revisions. Reordered key lookup,
  changed revisions/content, and actual checkpoint relocation are checked.
- Validation checks graph-wide identity uniqueness, reference closure, explicit
  group-boundary ports, acyclic containment, ordered repetition membership and
  variants, terminating aliases, declared expression symbols, safe dimension
  products, and diagnosed known shape disagreement. Chains of 1,500 groups and
  aliases verify iterative traversal. Expressions remain inert text.
- Physical storage is verified against the guarded inventory even when numeric
  inspection is unavailable. Available inspection requires exact existing native
  rank-1/rank-2 identity and geometry; packed storage and fused regions stay
  descriptive. Missing parameters and explicit unknown regions remain partial;
  failed or interrupted builds produce no graph.
- Byte checks consume the published exact 32 MiB/one-byte-over oracle. Reduced
  budgets exercise early builder termination; oversized typed records are rejected
  before serialization copies. Deep metadata, text and UTF-8 bounds are checked.
- Read-only relocated synthetic checkpoints are analyzed with network, numerical
  model construction, generation, tracing, GPU, tensor materialization, safetensors
  loaders and checkpoint-code import guards. A fresh subprocess also rejects any
  import of torch, transformers, safetensors or httpx by the package. Fingerprinting
  occurs through the existing bounded snapshot path before these analysis guards;
  it is not claimed to be a tensor read.
- Existing guarded GPTQ/NVFP4 fixture metadata verifies that description inputs
  preserve complete physical inventory, omit file/offset fields, and expose only
  admitted native numeric identities.

## Reproduction

From `backend`, using the pinned development environment:

```sh
uv run --locked python scripts/generate_architecture_records.py --check
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest
```

Local validation passed all 125 focused graph tests, the full backend suite
(753 passed, 19 CUDA-only tests skipped), lint, formatting, generated-record drift
checking, and strict typing. It used Python 3.12.14 with the repository-pinned CPU
dependencies.
The backend CI also checks Python 3.14 and the installed wheel outside the source
tree. Actual Qwen3, Qwen3.5 and V-JEPA 2 acceptance remains the responsibility of
their later family/reference consumers; this report establishes structural and
metadata-only behavior, not inference correctness or numerical equivalence.
