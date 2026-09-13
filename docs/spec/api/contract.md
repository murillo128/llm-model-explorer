# API contract

## Ownership

The API is a first-class, contract-first project boundary. It is not generated conceptually from Python implementation classes and is not owned by either backend or UI.

Backend and UI both implement the current contract. The proof of concept does not expose public API versions such as `/v1` and does not require backwards compatibility with older contract revisions. Both sides evolve together against the latest accepted contract.

Conventional HTTP request/response operations are described by an independent OpenAPI contract. Large numeric result streams additionally follow the binary streaming specification in `binary-streaming.md`.

## Design rules

Capabilities use explicit, strongly typed operations rather than a universal `execute` endpoint carrying an operation-name string.

JSON is appropriate for commands, descriptors, metadata, and ordinary small responses. Tensor values and other large numeric payloads must not be serialized as JSON or Base64.

The UI never sends or receives backend filesystem paths for models.

## Proof-of-concept capabilities

The contract must cover at least these logical capabilities:

- discover the available local models and their public metadata;
- create and address a session bound to one model;
- retrieve a hierarchical tensor inventory with names, shapes, ranks, physical dtype/representation metadata where useful, and other small descriptors;
- request a complete tensor in its logical visualization representation as a long, streamed operation;
- request reusable tensor statistics as an independent operation/artifact;
- tokenize text using the selected model's real Hugging Face tokenizer and return token IDs, token text/decoded representation, offsets when available, and special-token information;
- cancel an active long operation by `operation_id`.

Exact endpoint paths and machine-readable schemas belong to the OpenAPI artifact and must preserve these semantics.

## Errors

Errors detected before a streaming response begins use normal HTTP status codes and a structured JSON error body.

After a binary stream has begun, execution errors are represented by an error frame in that stream because the HTTP status has already been committed.

Cancellation is a distinct operation outcome and must not be collapsed into an unspecified generic server error.
