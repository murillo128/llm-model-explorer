# Shared API conformance fixtures

The source of truth is [`docs/spec/api/openapi.yaml`](../docs/spec/api/openapi.yaml),
with numeric and cross-field semantics in [`contract.md`](../docs/spec/api/contract.md)
and byte framing in [`binary-streaming.md`](../docs/spec/api/binary-streaming.md).
This directory contains validation tooling and generated examples, not a second
contract or an application codec. No model download, server, PyTorch, or GPU is
needed.

From the repository root, using Python 3.12:

```sh
python3 -m venv /tmp/lmex-contract-venv
/tmp/lmex-contract-venv/bin/python -m pip install -r api/requirements.txt
/tmp/lmex-contract-venv/bin/python api/validate_contract.py
/tmp/lmex-contract-venv/bin/python api/validate_contract.py --write
git diff --exit-code -- api/fixtures
```

The default command validates OpenAPI 3.1 with `openapi-spec-validator`, checks
component JSON Schemas with the Draft 2020-12 validator, resolves every internal
reference (including discriminator mappings), and validates positive and negative
schema/cross-field instances. It then regenerates the fixture in memory and
requires exact equality with the committed file. `--write` runs the same checks
and replaces the generated file. The dedicated `Validate API contract` workflow
runs both commands and checks that regeneration leaves no diff.

## Consuming the fixture

[`fixtures/conformance.json`](fixtures/conformance.json) is deterministic UTF-8
JSON. It contains:

- `wire_cases`: exact complete HTTP body bytes in `wire_hex`, with an expected
  terminal outcome (`complete`, `error`, `cancelled`, or `reject`). Successful
  cases include decoded META, concatenated DATA hex, and optional human-readable
  numeric expectations. Error/cancellation cases with partial DATA identify it
  as incomplete. Sparse histogram entries index the concatenation of the row
  and column sections; all omitted entries are zero.
- `schema_cases`: named instances for the OpenAPI component in `schema`, and a
  `valid` expectation. Cross-field-invalid cases may pass JSON Schema alone;
  consumers also enforce the linked prose constraints.
- `tokenizer`: a synthetic response, explicit code-point array, UTF-16 length,
  and source substrings. It demonstrates emoji, a combining mark, overlapping
  spans, a source-typed special ID, and a backend-inserted special without a span.
  IDs and token text here are illustrative, not output claimed from a real model.
- `numeric_cases`: an asymmetric little-endian uint32 oracle and an overflow
  boundary that must yield `unsupported_size` without allocating a giant tensor.

Hex is only the repository fixture representation. Runtime DATA is raw binary;
nonfinite float labels in decoded fixture expectations are explanatory strings,
never JSON-encoded tensor values on the API. Exact float32 bit patterns (including
negative zero and the chosen NaN payload) are in `data_hex`. Compare floating
statistics with the tolerance in the contract; compare integer counts and
materialized bytes exactly.

Python and TypeScript codec tests should feed every wire case as one chunk,
as single bytes, and using repeated chunk lengths `[1, 2, 3, 5, 8, 13]`. Also
exercise splits inside each header and float32/uint32 word. These chunkings
must preserve outcomes. Only the frame payloads, not network chunks, require
four-byte alignment. JSON member order is not significant: these golden bytes
are deterministic examples, not a required producer serialization order.

`reject` reasons describe the targeted defect, not public error codes or a
required parser exception taxonomy. Rejection may be detected earlier when a
case has multiple defects. The oversized control-frame cases contain only a
header: reject the declared length before buffering. The `large-data-truncated`
case declares a nearly 4 GiB DATA frame but supplies no payload; run it with a
bounded destination/sink to verify incremental handling without allocating a
frame-sized staging buffer. It is intentionally a tiny file, not a large blob.

## Input-embedding extension

[`fixtures/embeddings.json`](fixtures/embeddings.json) uses the same `wire_cases`
and `schema_cases` formats for the accepted post-PoC capability. Keeping it
separate lets the backend implementation child adopt these producer fixtures
without changing the existing named-tensor producer in this contract issue.
The UI decoder consumes both files; this does not add an embedding request
method or compose an embedding view.

`source` contains a tiny synthetic input table and its vocabulary/hidden sizes.
`request_cases` checks IDs against that source, including the exclusive upper
bound. An out-of-range ID is structurally valid JSON Schema but fails the
source-dependent request check with `validation_error`. `association_cases`
separates valid wire metadata from request matching: an otherwise valid result
with reordered/different IDs must not be displayed for the originating request.
The backend/UI implementation children must adopt these contextual cases in
addition to the codec fixtures. Numeric rows here are repository oracles only,
never a JSON matrix response format.

`api/validate_contract.py` validates and deterministically regenerates both files.
It checks explicit expected rows/bytes, shape products, request ranges, metadata
size limits, and positive/negative schemas. The shared stream cases cover
asymmetry, duplicates, singleton/empty sequences, early/partial error and cancel,
invalid metadata, and malformed/truncated framing.

After changing OpenAPI, run `npm run api:generate --prefix ui`, then
`npm run api:check --prefix ui`. Generated `types.ts` and `schemas.json` must
match the contract exactly; a second generation must leave their bytes unchanged.
`npm run check --prefix ui` also checks binding drift and shared codec fixtures.

The generator constructs frames and asserts small numeric reference results;
it does **not** parse streams or claim that a future codec passes these cases.
Codec behavior, incremental memory use, HTTP/CORS headers, session cancellation,
and error redaction need implementation tests in their owning issues.

## Static architecture extension

[`fixtures/architecture.json`](fixtures/architecture.json) contains one small synthetic
response/context and deterministic mutation cases. It is a structural oracle, not
acceptance evidence for a real checkpoint or a runtime architecture route. The
context pins the session model, logical inventory and tokenizer availability.
`edits` and `context_edits` contain string-key paths (array indices are decimal),
with either a replacement `value` or `delete: true`.

`schema_valid` tests the generated closed unions and local bounds independently
in Python and the UI's Ajv runtime. `valid` additionally requires the graph/context
checks in `architecture_conformance.py`: identity/reference closure, containment,
repetition order, directional group ports, declared symbols, alias termination,
physical versus logical geometry, and real native inspection membership. Schema
success alone does not establish graph conformance. Runtime implementation children
must consume the contextual cases when adding retrieval and graph consumers; this
publication adds no client request method or backend architecture route.

`byte_cases` describe repeated 4 KiB chunks plus a tail, exercising the exact 32 MiB
boundary and early rejection without allocating a large response. `bounded_size`
stops consuming at overflow; `serialized_size` counts UTF-8 JSON incrementally.
Runtime producers must bound construction/serialization and consumers must bound
reads before parsing, returning localized `unavailable/unsupported_size` without
truncation. These fixture helpers do not claim runtime enforcement has shipped.
`http_cases` preserve pinned-content mutation as HTTP 409 `model_content_changed`.

Baseline native inventory responses now include `coverage: complete` and empty
`diagnostics`, preserving every existing descriptor and numeric golden byte. Partial
inventories (including empty ones) are explicit; their quantized producer belongs
to the checkpoint-admission child. The validator regenerates all three fixture
files together; existing tensor/tokenizer/embedding golden files remain unchanged.
