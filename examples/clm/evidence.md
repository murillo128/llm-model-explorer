# CLM acceptance evidence

Initially observed on 2026-09-30; reference and focused checks refreshed on
2026-10-01 after the encoder-binding correction, using the immutable inputs in
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
`1ffc15f2881bcc8474b248a7ec7bdb47f7db755572d641e2f5dee20b4edcba78`.
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
| Cold / warm startup | 46.818 / 23.216 seconds, with native Qwen and CLM together |
| Warm cache | Cache retrieval logged; graph identity unchanged |

Graph identity:
`4e3954175f3bb7ecbbeb74496cac66ac7d3984df847f2a56e6134bab41325187`.

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

The production Chromium DPR1 reference test passed without retries (62.776
seconds): select the complete CLM model, expand Action head, open the native
`clm.action_head.out.weight` matrix and observe no UI error. Its attached summary
reported the same graph identity and [512, 1536] logical shape. The rendered
matrix screenshot was also inspected.

## Deterministic fixtures and repository gates

The 27 CLM backend cases passed, including independent float64 scalar numerical
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
The separate reduced-fixture production-browser test passed without retries.
These fixtures do not substitute for the reference results above.

Fresh repair checks: all 27 CLM cases, backend Ruff/format/mypy, exporter/reference
script Ruff/format, full-reference re-export, installed CLI import, production
TCP cold/warm checks, production UI build and both fixture/reference browser paths.
The broader component gates below were run on the original implementation at
`883868493eb48ad1c018cd0b546beeaad30c8400`. Backend runtime, API, UI source/tests and
dependency files remain identical; those broad suites were not rerun for the
offline admission/provenance correction. The affected exporter and CLM consumer
paths were rerun as described above.

| Repository gate | Result |
| --- | --- |
| Backend Ruff / formatting / mypy | Passed; exporter included in focused type checking |
| Backend wheel build / installed smoke | Passed; installed CLI also validated the full reference from outside the source tree |
| UI aggregate check | Passed: generated API bindings, types, lint, 57 files / 1,019 unit tests, production build |
| UI desktop + native-scrollbar browser matrix | 394 passed, no retries |
| Independent API validation | 365 references, 118 instance cases, 262 architecture cases, 132 reproducible wire fixtures |
| Native integration gate | Passed: fixture generation/reproducibility, 35 TCP passed / 10 optional skips, 46 DPR1 browser passed / 16 optional skips |

Integration skips concern existing unrelated local reference models and optional
CUDA. This issue's actual CLM reference checks passed separately. Existing CI
branch filters defer child-PR remote component/application checks when targeting
`codex/epic-issue-260`; the aggregate epic PR retains those gates. The corresponding
local checks above ran without changing workflows or waiving validation.

The full backend run initially had one environment failure: its reused dependency
environment lacked the installed validator command. The wheel was built and
installed into an isolated environment; the unchanged failing installed-command
case and all 20 CLM cases then passed (21 passed). The earlier broad run had
1,830 passed and 20 optional CUDA skips. No test expectation was weakened.
The sandboxed host Xvfb crashed before browser launch; the same headed checks
passed using the working host display outside that filesystem sandbox, with
isolated ports and external evidence directories.

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
