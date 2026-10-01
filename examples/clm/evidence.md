# CLM acceptance evidence

Initially observed on 2026-09-30; reference and focused checks refreshed on
2026-10-01 after the filesystem/publication correction, using the immutable inputs in
[the reproduction recipe](README.md#immutable-reference-inputs). This report
establishes static inspection of the complete native CLM-v0.1-8B export; fixture
results and actual-reference results are separate below. Weights, generated
graphs, caches, browser captures and full logs remain outside Git.

## Actual pinned reference

Both Hub repositories' selected files passed `hf cache verify`. Missing-file
warnings concerned unselected repository assets, not any required checkpoint,
configuration or tokenizer file. The native Qwen encoder comprises five shards
(16,381,516,776 bytes), hidden size 4096, 36 layers, 32 query heads, 8 KV heads,
head dimension 128, intermediate size 12288 and vocabulary size 151936, in BF16.
The 75,557,149-byte head payload supplies two F32 MLPs: hidden size 4096, width
1536, depth 3, projection size 512, GELU, LayerNorm enabled and residual disabled,
plus a scalar learned logit scale. Its SHA-256 is
`b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5`.

The exporter revision is `clm-inspection-export-1`, with source-file SHA-256
`0228d95c6618135e40abe47f545fd5db57ada152ff6cc8336eb35532eab6755b`.
The exported provenance records all selected base/head inputs, original tensor
shapes/dtypes, CLM source hashes, producer identity and namespace mapping.
Canonical CLI validation returned `valid`, coverage `complete`, no diagnostics.
The regenerated package's input hashes, head payload, native inventory and CLM
source pins match the original export. The pinned encoder config has no identity
or revision declarations; provenance explicitly records `operator_selection`
for both bindings, supported by the operator's checksum verification above.

| Observed exported property | Result |
| --- | --- |
| Native tensor inventory | 416 tensors; complete backbone and both heads |
| Model identity | `Contrastive-LM/CLM-v0.1-8B@e939398d4556fcd9400c76fa8c5a513202f42b0a` |
| Architecture | `model_defined`, `complete`, model-supplied origin notice retained |
| Graph | 2,638 nodes, 4,008 edges, 415 parameter identities |
| Encoder repetitions | Two independent invocations, 36 concrete layers each |
| Definition / observed JSON response size | 3,480,149 / 8,150,576 bytes |
| Cold / warm startup | 51.942 / 25.098 seconds, with native Qwen and CLM together |
| Warm cache | Cache retrieval logged; graph identity unchanged |

Graph identity:
`c19100411b53559d5e9b4acf8c239440c4807d7ba875f1e8f313794d95259227`.

`check_reference.py` compared complete production TCP tensor streams with the
original `.pt` tensors or original Safetensors, after canonical float32
conversion. All bytes matched. Representative first four values:

| Tensor | Native dtype / shape | Values |
| --- | --- | --- |
| `clm.state_head.out.weight` | F32 / [512, 1536] | -0.7088263035, 0.3176046312, 0.3624875546, 1.0252525806 |
| `clm.action_head.out.weight` | F32 / [512, 1536] | 1.3761154413, -0.8851161599, 1.1981379986, 1.5786347389 |
| `model.norm.weight` | BF16 / [4096] | 2.125, 1.921875, 2.046875, 2.03125 |

Pinned tokenizer output for `Hello world, España 😀` matched the local original
Tokenizer: `[9707, 1879, 11, 56508, 90316]`. A request containing these IDs plus a
duplicate `9707` returned every exact original embedding row, in order. The
first row begins `[-0.035400390625, -0.0033721923828125, -0.033935546875,
0.01251220703125]`.

The production Chromium DPR1 reference test passed without retries (65.829
seconds): select the complete CLM model, expand Action head, open the native
`clm.action_head.out.weight` matrix and observe no UI error. Its attached summary
reported the same graph identity and [512, 1536] logical shape. The rendered
matrix screenshot was also inspected.

## Deterministic fixtures and repository gates

The 36 CLM backend cases passed, including independent float64 scalar numerical
oracles across head options/native dtypes, exact storage preservation and weight
sharing, restricted-load rejection, incomplete export rejection, actual startup
and streams, tokenizer rows, runtime network/checkpoint-load traps, malformed
binding/edge failures, relocation stability and content/cache invalidation.
Six same-shape repository/revision conflict regressions failed on the original
exporter (`883868493eb48ad1c018cd0b546beeaad30c8400`) with `DID NOT RAISE` and pass
after the repair. They change only identity/revision declarations, preserve the
original tensors and geometry, and require unchanged input bytes, no published
destination and no selectable CLM entry. A matching-declarations case verifies
acceptance and exact source declarations in provenance. Secondary declarations
must agree with primary fields; none may silently override a conflicting field.
The separate reduced-fixture production-browser test passed without retries (8.8 seconds).
These fixtures do not substitute for the reference results above.

Filesystem regressions belong to `backend/tests/test_clm_export.py`. Against the
reviewed exporter at `f08ef0cc71ad817b270d43aac31974c00821ad97`, four symlink-parent
or traversal aliases into the encoder (shared/copy modes) failed with `DID NOT
RAISE`, and two writable-parent/read-only-ancestor cases failed at staging with
OS `PermissionError`. These six cases pass after the repair. Permission tests ran
as non-root UID 1000 and proved the ancestor actually rejects directory creation;
access checks were not mocked. Alias rejection preserves the complete input tree,
file bytes, permissions and symlink targets.

Three real catalogue/validator controls passed before and after the correction.
During valid, malformed-binding and interrupted validation, the temporary CLM
cannot be discovered or pinned. Successful publication preserves the validated
directory's device/inode; failure leaves neither a destination nor staging residue.
Inputs remain unchanged. Nested staging is on the output filesystem in a private,
configuration-free container. The canonical validator's explicit root admits
confined shared shards and rejects external package/shard targets; its default CLI
root remains the package parent. This boundary's regression belongs to
`backend/tests/test_model_defined_service.py`.

Fresh checks on the corrected validator/staging layout: full backend suite, all
74 CLM/model-owned lifecycle cases, backend and exporter Ruff/format, strict mypy,
wheel/sdist build and installed imports, UI aggregate check, full-reference
re-export and installed CLI validation, production TCP cold/warm checks and both
fixture/reference browser paths. After retaining the exporter's previous private
publication mode (`0700`), the 74 affected cases reran on the final exporter and
the reference was regenerated. Backend runtime/tests/dependencies were unchanged
by that final permission-preservation adjustment.

| Repository gate | Result |
| --- | --- |
| Backend full suite | 1,897 passed / 20 optional CUDA skips; native `python -m pytest` |
| Backend Ruff / formatting / mypy | Passed; 105 typed files including exporter |
| Backend wheel build / installed smoke | Passed; installed CLI also validated the full reference from outside the source tree |
| UI aggregate check | Passed: generated API bindings, types, lint, 57 files / 1,019 unit tests, production build |
| CLM production browser | Fixture and complete reference each passed, no retries |

Prior aggregate evidence for the pinned integration commit
`f08ef0cc71ad817b270d43aac31974c00821ad97` is recorded in the
[final epic audit](https://github.com/murillo128/llm-model-explorer/pull/263#issuecomment-5927987026):
394 UI browser passes; independent contract/bindings/wire checks; 35 TCP passes /
10 optional skips; 47 production-browser passes / 16 optional-reference skips.
Those results are prior integration evidence, not fresh results for this repair.
UI/API/acceptance source and dependencies remain identical to that pin. Existing
branch filters defer child-PR remote component/application checks when targeting
`codex/epic-issue-260`; the corrected aggregate epic PR retains mandatory fresh CI.
No workflow or expectation was weakened.

The first broad attempt had 1,896 passes / 20 skips and one environment failure:
a temporary unguarded pytest launcher was re-imported by multiprocessing spawn,
starting child suites and timing out the unchanged cross-process cache test.
That test passed using the native `python -m pytest` entry point (6.08 seconds),
then the complete native rerun passed as recorded above. Headed browser checks
used the working host Xvfb outside the filesystem sandbox, whose display crash
was established during the original acceptance, with isolated ports and external
evidence directories.

Environment: Linux x86_64, Python 3.12.14, CPU PyTorch 2.14.0, Transformers 4.57.6,
Node 24.14.0, Playwright 1.63.0 and Chromium with SwiftShader. Startup times are
local observations, not performance guarantees.

## Supported scope and verification limit

Supported: the complete pinned native reference through explicit offline export,
exact base/head tensor inspection, pinned tokenizer/input-embedding lookup and
expandable model-owned decision structure. The original `.pt` directory alone
is not an admitted model. Other encoder families, quantized variants, arbitrary
checkpoint objects and unsupported head configurations are rejected. Runtime
performs no downloads or checkpoint conversion.

No encoder forward pass, inference endpoint, generation, candidate-quality
benchmark or upstream quality reproduction was run or added. Successful import,
numerical head examples and browser rendering do not establish a verified full
forward computation; the graph's model-supplied notice preserves that limit.
