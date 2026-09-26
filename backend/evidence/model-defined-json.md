# Model-owned JSON architecture definition evidence

## Implemented boundary

`architecture.json` is an optional, fixed-name, data-only checkpoint asset. Its
versioned authoring schema is distinct from the runtime graph schema. The loader
rebinds local IDs and named native parameters to the pinned inventory and retains
the existing graph closure, shape, hierarchy, repetition and publication checks.
The result uses `model_defined` scope and explicitly disclaims verification of
correspondence to the author's model code. Existing packaged descriptions are
used only when the asset is absent; malformed present assets do not fall back.

## Observed focused validation

On 2026-09-17 the scoped preparation check completed successfully on Linux CPU,
using the repository-locked Python 3.12 environment, uv 0.12.13 and Node 24.14.0.
It ran the following before publishing the feature implementation:

- Full backend Ruff lint/format and strict mypy checks.
- `backend/tests/test_model_defined_architecture.py` and
  `backend/tests/test_model_defined_service.py`, including the HTTP integration
  test with the normal Transformers dependency installed.
- Independent API contract validation and fixture generation.
- Backend graph-record generation, portable JSON Schema generation and UI API
  binding generation using the repository tools.
- UI API freshness, TypeScript, ESLint and the ArchitectureExplorer component
  tests, including the visible model-supplied provenance notice.
- `git diff --check` and an explicit changed-path publication allowlist.

The separate offline editing environment passed 42 focused cases and skipped
only the HTTP test because Transformers was absent there. That limited result
was not used as a substitute for the locked-environment check above. The full
existing application/backend/API/UI CI gates remain the authoritative regression
and merge evidence; focused checks alone do not establish their outcome.

## Independent test inputs and limitations

The tests construct small, explicit native Safetensors without using the loader
to generate expected graph relationships. They check actual tensor values,
read-only files, cold/warm cache identity, content mutation, safe failures,
precedence, missing/wrong parameters, unsupported versions, invalid graph
references, explicit repetitions and unfamiliar operation names. Guarded tests
prohibit model construction, execution, network and tensor-payload access during
analysis. Checkpoint Python is neither imported nor executed.

Version 1 supports complete native F32/F16/BF16 parameter bindings, not arbitrary
slices or model-supplied quantization decoders. Rank-above-two parameters retain
their shape but cannot open in the existing matrix modal. Missing storage remains
explicitly partial. Formulas and symbolic expressions are display-only.

This evidence establishes structural validation and integration of author-owned
metadata, not mathematical equivalence to `forward()`, trained model quality,
universal checkpoint support, numerical inference correctness, or CUDA behavior.
The worked example under `examples/model-owned-architecture/` is a generic
linear/GELU encoder, not an NFL adapter or a trained reference model.
