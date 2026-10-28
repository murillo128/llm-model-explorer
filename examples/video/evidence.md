# Video inspection evidence

Validation recorded on 2026-10-28 on Linux, Python 3.12.14 and CPU PyTorch.
These results cover static inspection, local tokenization and logical weight
streaming. No video inference was run.

## Reference coverage

| Profile | Native profile/graph conformance | Complete local checkpoint acceptance |
| --- | --- | --- |
| LTX 2B Distilled 0.9.6 | Passed | Not run; 0.9.8 is the complete LTX integration reference permitted by issue #331. |
| LTX 2B Distilled 0.9.8 | Passed | Passed: export, preparation, tokenizer, independent graph validation, two complete matrix streams and production browser. |
| Wan 2.2 TI2V-5B FP16 | Passed | Passed: export, preparation, UMT5 tokenizer, independent graph validation, native/FP8 matrix streams and production browser. |
| Wan 2.2 T2V-A14B FP8 scaled | Passed, including both experts | Pending. Actual selected bytes from both experts passed the numeric comparison below. |
| Wan 2.2 I2V-A14B FP8 scaled | Passed, including both experts | Pending. Actual selected bytes from both experts passed the numeric comparison below. |

Complete A14B sources plus the byte-preserving package copy require about 71.2 GB
for either pair with its VAE and UMT5 assets. The available space before reference
preparation was about 56.4 GB. No complete A14B pair was downloaded or certified;
metadata and partial bytes do not substitute for that gate. Only task-created
scratch copies were removed between complete-reference runs.

The pinned artifact identities, original hashes, configurations, inventories and
license declarations are in the [reference register](../../backend/src/llm_model_explorer/video_reference.json).
The exporter checked complete original file hashes before and after copying.
No weights, caches or complete generated graphs are repository artifacts.

## Complete-reference measurements

The [LTX report](evidence/ltx-0.9.8.json) and
[Wan report](evidence/wan-ti2v-5b.json) record source sizes/hashes, graph identities,
token IDs, selected matrix identities, complete logical-stream SHA-256 hashes and
process high-water RSS for descriptor revision 1. The LTX positional correction
below supersedes its graph identity/count; the original numerical measurements
are retained as recorded. The matrix oracle uses bounded slices of the original
Safetensors files and the source loader's weight-scale rule; it does not call the
application decoder or instantiate an upstream model.

| Measurement | LTX 0.9.8 + T5 | Wan TI2V-5B + UMT5 |
| --- | ---: | ---: |
| Package bytes | 25,391,240,842 | 18,162,324,592 |
| Graph response bytes | 5,741,164 | 5,665,767 |
| Graph nodes | 2,295 | 2,271 |
| Graph parameters | 1,160 | 1,263 |
| Logical tensors | 1,163 | 1,263 |
| Backend peak RSS, KiB | 475,248 | 764,700 |
| TCP reference check, seconds | 85.183 | 40.722 |
| Export peak RSS, KiB | 242,032 | 251,468 |
| Export wall time, seconds | 88.83 | 56.21 |

These are observations for the named package and selected operations, not a
constant-memory guarantee for every possible tensor operation. Export uses 1 MiB
copy blocks without dtype conversion; the test oracle slices 64 rows at a time.
The complete selected denoiser and text-encoder Q matrices matched byte-for-byte
as canonical float32, including scaled FP8 UMT5 weights.

[Selected A14B byte evidence](evidence/wan-a14b-selected-bytes.json) covers all four
requested FP8 expert files at immutable URLs: columns 0–7 of rows 0 and 5119 of
`blocks.1.self_attn.q.weight`, plus its scalar weight/input scales. The native CPU
reader with three-element chunks matched an independent E4M3FN bit-decoding
oracle followed by float32 weight scaling. The report's full-file SHA-256 fields
are **declared source identities**, not hashes verified from complete downloads.
The independent small fixture additionally uses a nonunit activation scale to
prove it is not multiplied into static weights.

## Browser observations

Real production-browser runs passed independently for LTX and Wan (one case per
complete package). Inspected captures show readable semantic prompt/image/video
ports and a native, noninitial parameter with completed values/statistics:

- [LTX overview](evidence/ltx-video-overview.png) and [block 1 modulation weight](evidence/ltx-video-weight.png).
- [Wan overview](evidence/wan-video-overview.png) and [block 1 Q matrix](evidence/wan-video-weight.png).
- [Synthetic low-noise expert](evidence/synthetic-a14b-low-noise.png): the focused
  UI fixture switches high/low expert block 1, checks distinct tensor requests and
  exact displayed values, then restores the retained graph view. This is browser
  conformance, not a full A14B checkpoint capture.

The existing real importer refresh browser test also passed with regenerated IDs.
The fixture update was checked to change only analyzer-derived identifiers and
revision, preserving its structure and navigation correspondence.

## LTX positional dependency correction

LTX descriptor revision 3 carries frame rate from the semantic configuration to
temporal coordinates and the Q/K rotary path in every block. The reviewed
[pipeline](https://github.com/Lightricks/LTX-Video/blob/4b2d053057623ddd4d0a1d3e9cd28890e9ef487f/ltx_video/pipelines/pipeline_ltx_video.py#L1143-L1149),
[coordinate conversion](https://github.com/Lightricks/LTX-Video/blob/4b2d053057623ddd4d0a1d3e9cd28890e9ef487f/ltx_video/models/autoencoders/vae_encode.py#L176-L208)
and [transformer](https://github.com/Lightricks/LTX-Video/blob/4b2d053057623ddd4d0a1d3e9cd28890e9ef487f/ltx_video/models/transformers/transformer3d.py#L193-L254)
establish pixel-to-second conversion, positional normalization and the exponential
frequency grid. LTX rotates paired channels across the full projected width
before splitting attention heads. The 0.9.6 causal coordinate correction is
explicit; 0.9.8 retains the reviewed default without that correction.

Unpatchification receives a separate three-element `T,H,W` value derived from the
latent shape, following the pinned [pipeline call](https://github.com/Lightricks/LTX-Video/blob/4b2d053057623ddd4d0a1d3e9cd28890e9ef487f/ltx_video/pipelines/pipeline_ltx_video.py#L1297-L1306)
and [patchifier](https://github.com/Lightricks/LTX-Video/blob/4b2d053057623ddd4d0a1d3e9cd28890e9ef487f/ltx_video/models/transformers/symmetric_patchifier.py#L62-L78).
Rotary frequencies feed only the attention blocks, never the reconstruction grid.

The frame-rate and geometry regressions each failed for both LTX releases before
their respective repairs (two failed, three Wan controls passed). Both assertions
remain in the final passing five-profile test. The focused video/graph/service
suite passed 388 cases in 49.03 seconds;
Ruff, formatting and mypy passed. A fresh installed-wheel smoke passed 18 cases
in 69.34 seconds. Full serialized metadata graphs for all three Wan profiles
retained their exact SHA-256 hashes, while both LTX graph identities changed.
Only the LTX producer revision changed; unrelated analyzer identities are stable.

A fresh production build and complete-package LTX browser run passed in 66.7
seconds, including native block-1 weight inspection. The retained 25,391,240,842-byte
package prepared graph `28ac3eddde9b6b2c7872b99666b31b5ba60aab0bcd879a558d1bb8551be1cc92`
with 2,301 nodes and 1,160 parameters. Observed preparation was 40.146 seconds
(33.592 seconds hashing; 3.602 seconds analysis). The LTX captures above are from
this revision. Export and original-file numerical comparisons were not repeated
for this descriptor-only correction; their earlier reports remain separately
identified. No additional complete checkpoint acceptance is claimed.

Focused reproduction after the correction:

```sh
cd backend
uv run --locked pytest tests/test_video_architecture.py tests/test_video_export.py \
  tests/test_video_fp8.py tests/test_architecture_analysis.py tests/test_architecture_service.py -q
```

The production-browser command in the example README selects `video.spec.ts`
with `LMEX_VIDEO_REFERENCE_PACKAGE` pointing at the complete local LTX package.

## Repository checks before the positional correction

- API validation passed: 468 references, 130 instance cases, 296 architecture
  cases, 81 neuron cases, 8 neuron payloads and 138 reproducible wire fixtures.
- Backend Ruff, formatting and mypy passed. The complete 2,199-case suite exposed
  two failures: duplicate FP8 graph companions and the navigation fixture's old
  analyzer IDs. Both were repaired; the final affected run passed all 351 cases
  across video storage/export/graphs, independent graph conformance and that
  navigation fixture. The full run had 2,177 passing cases; 20 CUDA
  cases were skipped. The source-specific graph assertions first failed for all
  five profiles, then passed after fixing Wan context trimming/padding and LTX
  VAE posterior sampling/channel normalization and normalization epsilons.
- The isolated installed-wheel smoke passed 18 cases. Its built wheel contained
  the exact video reference register and native descriptor modules.
- `npm --prefix ui run check` passed (69 files, 1,315 tests, generated bindings,
  types, lint and build). Both focused browser cases passed. Types/lint were
  repeated after the optional-reference configuration update.
- CI command/selection tests passed: 50 cases and 124 subtests. The optional video
  file is extended-only; explicit package input promotes full reference checks.
- Final `acceptance/check-integration.sh`: 87 HTTP/control-plane cases passed,
  11 optional CUDA/reference cases skipped. Production browser: 37 passed, 19 skipped.
  Complete LTX/Wan reference browser evidence above was run separately with actual
  package inputs; the unconfigured optional video case is skipped in the aggregate.

Reproduction commands for complete references are in [README.md](README.md).
Routine checks use the repository entry points: `api/validate_contract.py`,
backend Ruff/mypy/pytest, `acceptance/check-wheel.sh`, `npm --prefix ui run check`,
focused `test:browser` and `acceptance/check-integration.sh`. Keep full reference
inputs and evidence roots outside Git and use isolated browser ports.

## Test ownership and independent expectations

- LTX positional conformance is owned by `test_video_architecture.py`. The pinned
  pipeline, `vae_encode.py` and `transformer3d.py` provide the oracle: latent-to-pixel
  stride `[8,32,32]`, the 0.9.6-only causal correction, frame-rate conversion,
  normalization by `[20,2048,2048]`, and full-width rotary frequencies. The regression
  checks the actual configuration-to-Q/K edges for both releases and the displayed
  expression with pixel frame 8 at 25/50 fps (0.32/0.16 seconds), preserving spatial
  coordinates. The unpatchify regression additionally checks its geometry producer
  and the three-element latent `T,H,W` shape, using the pinned pipeline's
  `unpatchify` call as its oracle. Frequencies may feed only the attention blocks;
  they cannot substitute for reconstruction dimensions. Existing profile/graph
  tests cover all Wan conventions separately.
- Backend storage owns FP8 admission and numerical correctness. Hand-written
  E4M3FN bytes for `[1, -2, 0.5]`, weight scale `0.25`, and a distinct input
  scale must produce `[0.25, -0.5, 0.125]`. Exercise partial/chunk/ordered-row
  reads, native tensors in the same package, both declared scale spellings,
  missing/invalid/orphan companions and mutation. Existing packed-format tests
  remain the regression owner for GPTQ/NVFP4/NF4/compressed-tensors.
- Backend export/profile tests own source confinement, component identity,
  geometry, namespace collisions, bounded byte-preserving copying, snapshot
  mutation and atomic publication. Pinned upstream source and artifact metadata
  supply profile expectations; generated output is not the expected-result oracle.
- Native analysis tests own the two LTX releases and three Wan variants, text/VAE
  operations, optional/required/absent image conditioning, timestep expert
  selection and independent bindings in noninitial block instances.
- Independent API fixtures own the new static scope and serialization. One real
  backend path covers export, discovery, session, prepared graph and weight
  streaming; a representative production-browser journey covers multimodal ports,
  expert/instance navigation and the existing weight modal.
- Final checks use the repository's contract/generator, backend lint/type/tests
  and installed-wheel checks, UI aggregate check, focused browser and integrated
  TCP/production-browser harness. Full local reference evidence remains separate
  from fixtures and must name the files, hashes and variants actually checked.
