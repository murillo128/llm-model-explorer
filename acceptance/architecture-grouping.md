# Reviewed Architecture components — issue 121

## Contract and baseline

The activation baseline is `a82a1033b18eb575b41f13c6782cee496bd5b1a6` on
`codex/epic-issue-124`, after #119 and #120. This change makes reviewed Attention
and MLP components explicit in every actual layer; it preserves the accepted
logical native/GPTQ/NVFP4 inspection contract. It introduces no new model family,
inference path, tensor geometry, graph schema or layout algorithm.

Dense Qwen3/Llama receive `.self_attn` and `.mlp` module groups; V-JEPA receives
`.attention` and `.mlp` groups in both stacks. Existing Qwen3.5 `.self_attn` and
`.linear_attn` groups are reused, with an additional `.mlp` group. All are real
modules in the already reviewed source revisions documented in
`backend/evidence/`. V-JEPA's head/rotary/GELU operation keys retain their original
identity even though those keys are not lexically inside the corresponding
module prefix. Membership comes from the authored source description.

The affected description revisions become `2`. Reviewed implementation revisions
are unchanged. New graph and record IDs are expected; tensor inventory IDs and
parameter names are unchanged. `semantic_role` and concise operation labels add
navigation information; description provenance retains each semantic source key.
Norms/residual additions remain in the containing layer. Qwen3.5's prior/next
KV, convolution and delta states remain within their original Attention owner.

## Operation-level conservation checkpoint

Before source changes, `architecture_grouping_cases.py` exported observed graphs
from the activation baseline. Its inputs reuse independently authored metadata
fixtures, not the producer's parameter-shape generator. The exporter records the
builder's explicit semantic keys as injective ID correspondence; no label/shape
matching or tensor-ID remapping occurs.

`ui/scripts/check-architecture-semantics.mjs` reuses #119's independent
`semanticSnapshot` oracle. The committed compact baseline contains SHA-256 values
for each semantic section, not generated weights or whole graphs. Seven cases
cover dense Qwen3, Llama, biased Llama, partial Llama, hybrid full/linear Qwen3.5,
and V-JEPA encoder/predictor with and without QKV bias. The comparison excludes
only the expressly added `semantic_role` attribute. Every existing operation,
formula/attribute, directed port dependency, multiplicity, shape, parameter
binding/storage/alias/inspection identity, repetition/instance owner, open
interface and diagnostic remains in the independent oracle.

The baseline can be reproduced by exporting the same cases with the three
producer modules and graph core from the pinned activation commit; the export
helper itself is test-only and compatible with that commit. Expected hashes must
not be refreshed from a changed producer to resolve a computational mismatch.
For detailed diagnosis, compare the before/after `semanticSnapshot` sections;
the full exports stay outside Git.

```sh
PYTHONPATH=backend/src:backend/tests backend/.venv/bin/python \
  backend/tests/architecture_grouping_cases.py /tmp/architecture-grouping
node ui/scripts/check-architecture-semantics.mjs /tmp/architecture-grouping
```

The hand-authored family topology assertions still run, now allowing exact
transparent group forwarding. Their set-based lookup is only a convenience for
individual expected connections; the separate multiset comparison remains the
conservation oracle. New component tests enumerate dense/V-JEPA membership,
per-layer interface and residual/norm ownership, exact module references,
Qwen3.5 state ownership, unused declared layer ports, repeat generation and
revision changes. Existing #119 positive/negative controls cover nested/unused
interfaces, residual drops, swapped equal-shaped ports, incorrect instance
weights, merged K/V state, duplicated signals and independent stacks. Missing
semantic-role metadata alone still validates without changing coverage.

Graph-count growth is accounted for independently of semantic equality:
dense adds two groups and seven forwarding edges per layer; Qwen3.5 adds one
group and two edges per layer; V-JEPA adds two groups and five edges per layer.
No non-group node or parameter is added or removed.

## Local checkpoint checks

The focused family suite passed 109 tests. Grouping/cache/service passed 113
tests, including warm retrieval with generation forbidden, post-readiness cache
loss, per-producer revision invalidation and immutable publication. Backend lint,
formatting, strict typing and generated-record drift checks passed. The seven
independent conservation cases passed against the activation baseline. All 29
issue-119 oracle controls passed. Independent API validation passed OpenAPI 3.1,
283 reference resolutions, 88 instance cases, 144 architecture cases and 76
reproducible wire fixtures.

These observations are synthetic/source-metadata evidence. Complete local
checkpoint and built-browser evidence is recorded separately after UI integration;
fixture success is not a claim of actual reference acceptance or inference
equivalence.


## Intermediate independent review and complete local checkpoints

The independent read-only [checkpoint review](https://github.com/murillo128/llm-model-explorer/pull/144#issuecomment-5686215639)
returned **PASS**, safe to integrate UI preference, for
`a82a1033b18eb575b41f13c6782cee496bd5b1a6..a83783ef06e1330cbb778f069b4223ffee1ef2bc`.
It was not final-capable. The reviewer additionally removed only newly added
groups and independently checked original ordered hierarchy, interfaces,
directed-edge multiplicity and forwarding shapes. No material findings.

The full backend suite passed 1,224 tests with 19 explicit CUDA-only skips.
The existing TCP Architecture acceptance passed all seven tests with no skips,
including all four approved local reference checkpoints. It ran from an isolated
archive of the reviewed backend commit, with CPU native threads bounded to one,
the existing approved reference manifest, offline model access and fresh temporary
caches. The [reference receipt](evidence/architecture-grouping/references.json)
records fingerprints, inventories, upstream and producer revisions, cold/warm
readiness, memory and exact graph identities. Startup measurements include model
root discovery/hashing; they are not isolated per-model compute benchmarks.

The four fixture checkpoints remain a separate test input. No checkpoint weights,
cache payloads or private model paths are committed. The subsequent browser gate
checks source groups/labels, recursive horizontal geometry, native shared-trunk
and port hover, and logical weight modals against the built UI.
