# Integrated static Architecture Explorer acceptance

The implementation includes all four packaged descriptions, startup preparation,
structured artifact retrieval, the global canvas, and native weight inspection.
All four approved complete checkpoints are available locally and passed the
production HTTP reference gate. Production-browser reference validation is
recorded separately below. Source review, metadata fixtures, and synthetic local
checkpoints are distinct from actual-reference evidence.

## Reproduce

Install the locked dependencies using `acceptance/README.md`, then run
`acceptance/check.sh`. This includes the new TCP tests and production browser
cases at DPR 1 and 2, alongside existing tensor, tokenizer, embedding, native
camera and continuity regressions. No test application downloads model assets.

For an operator-owned full reference run, create a JSON manifest outside Git:

```json
{
  "qwen3": {
    "repository": "JunHowie/Qwen3-0.6B-GPTQ-Int4",
    "revision": "b9d87006067b0c0c2dea836370d6288e14f112ab",
    "directory": "/absolute/path/to/Qwen3-0.6B-GPTQ-Int4"
  },
  "qwen35": {
    "repository": "AxionML/Qwen3.5-0.8B-NVFP4",
    "revision": "2ac1e750cda67cc8538d731f6216f77b9c3a6f72",
    "directory": "/absolute/path/to/Qwen3.5-0.8B-NVFP4"
  },
  "vjepa2": {
    "repository": "facebook/vjepa2-vitl-fpc64-256",
    "revision": "b3c1679b7c34d3255ef3547f27c7b226aefab26f",
    "directory": "/absolute/path/to/vjepa2-vitl-fpc64-256"
  },
  "smollm2": {
    "repository": "HuggingFaceTB/SmolLM2-135M",
    "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    "directory": "/absolute/path/to/SmolLM2-135M"
  }
}
```

Those revisions identify the metadata reviewed by the implementation children;
record the revision **actually installed**, not a copied example. If unavailable,
use `"revision": null` plus a nonempty `revision_absence_reason`. Revision identity
is operator attestation; the harness independently checks the admitted physical
inventory and configuration against the reviewed reference metadata and records the local content
fingerprint. It does not certify upstream weight provenance from a model name.
Each directory must be a complete immediate child of its configured parent root.
Symlinks to assets outside that root are rejected by normal admission.

```sh
export LMEX_ARCHITECTURE_REFERENCES=/absolute/path/to/references.json
export LMEX_REFERENCE_MODEL_DIR=/absolute/path/to/SmolLM2-135M
export LMEX_REQUIRE_ARCHITECTURE_REFERENCES=1
export LMEX_EVIDENCE_DIR=$(mktemp -d /tmp/lmex-architecture-evidence-XXXXXX)
acceptance/check.sh
```

Ordinary model-free CI explicitly skips absent references. The strict setting
turns missing references into failures. An invalid supplied directory, wrong
selected repository, incomplete/mismatched storage, missing advertised tokenizer,
partial graph, or unavailable graph fails; no fixture substitutes for it.
CUDA remains an independent optional numeric check.

`architecture_reference.py` records selected file names/bytes, fingerprint,
operator revision provenance, quantization metadata, graph scope/counts/identity,
structured producer manifests, cold/warm readiness, separately logged hashing,
analysis/cache timings and Linux backend peak RSS. Cold means a fresh artifact
cache, not a flushed operating-system page cache. Browser attachments record
layout time, graph counts, viewport, browser version and a JS heap snapshot;
that snapshot is not peak whole-browser RSS or GPU memory. The backend prepares
all admitted sibling checkpoints, so readiness/RSS cover the configured parent
root; per-model preparation logs distinguish their work. Keep raw graphs, caches,
logs, traces, weight bytes and local-path manifests outside Git.

## Deterministic coverage and its limits

| Evidence | What it establishes |
| --- | --- |
| `acceptance/test_architecture.py` | Four complete reduced descriptions over production CLI/TCP; plus a bounded Kimi Linear metadata fixture through the production prepared-graph endpoint, covering all 6,656 routed experts without full-sized weight files; independent API closure/context validation; complete recognized logical inventories; exact native and admitted decoded matrix bytes; read-only files; cold/warm graph identity and unchanged artifacts; unsupported architecture isolation; missing predictor gives partial scope; stale sessions, restart-required and cache loss |
| Integrated startup guard | Model construction/calls, tracing, tensor reads, tokenizer construction, full Safetensors access and outbound connection entry points fail if used by startup or graph retrieval |
| `ui/acceptance/architecture.spec.ts` | Built UI plus real service; all backend graph records in the component picker and expanded layout; original edge IDs preserved; concrete last-instance selection; dimensions state; later-layer native vector bytes at the GPU upload boundary; localized unavailable binding; modal focus/camera restoration; explorer continuity; no phantom V-JEPA tokenizer calls; close during conversion streaming and repeated cleanup; Kimi's full graph remains selectable and expands at DPR 1/2 |
| `backend/tests/test_architecture_service.py` | Startup socket is not listening until preparation finishes; sequential terminal outcomes; interruption before publication; no warm or GET regeneration |
| Description tests and child evidence | Independent source-derived language branches, hybrid layer order, both V-JEPA stacks, geometry, aliases, packed/fused bindings, unresolved cases and guarded metadata-only analysis |
| `ui/tests/architecture*.spec.ts` | Component graph stress, shape labels, collapsed dependency boundaries, accessible modal, late callbacks, model/session replacement and worker/renderer disposal |
| Existing production suite | SmolLM2 fixture tensor/tokenizer/embedding, exact pixels, camera and continuity, sharing and cancellation regressions; the resumed integration also passes the actual SmolLM2 Base regression at DPR 1 and 2 |

The tiny checkpoints reuse independently authored component metadata. Their
weights follow `(index % 29 - 14) / 8`; dense native weights use F16 to exercise
conversion streaming. Synthetic packed samples use independently authored scalar
formulas. The actual-reference browser path verifies native samples with local
Safetensors slices and selected later-layer GPTQ/NVFP4 samples with the independent
bounded scalar oracle in `backend/scripts/check_quantized_reference.py`.
Full mathematical topology comes from the source-derived
component assertions, not from treating analyzer output as its own oracle.

## Execution evidence

The current run's concise results and unresolved prerequisites are recorded in
[architecture evidence](architecture-evidence.md). Neither static analysis nor
these checks claims inference correctness, quantization numerical equivalence,
universal checkpoint support, or hardware throughput.

The original [navigation integration acceptance](architecture-integration.md)
found retained obsolete layouts during repeated isolated-view returns. Its FAIL
receipt remains unchanged. After the separately accepted
[lifetime correction](architecture-lifetime.md), the
[resumed integration acceptance](architecture-integration-completion.md) passes
the full component-browser, production DPR 1/2, all four actual-reference, TCP,
semantic-conservation, exterior-pixel and bounded-resource gates. CUDA remains
unavailable and is recorded as SKIP; CPU and fixture results do not imply CUDA
acceptance.
