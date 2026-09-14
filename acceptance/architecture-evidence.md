# Static architecture integrated evidence

The four complete operator-provisioned reference checkpoints passed the production
HTTP gate on the integrated implementation at
`eebccb66218680af0f46ee326480a1dfa7022462`. This acceptance change adds harnesses
and documentation; it does not alter architecture semantics or rendering code.
The [reproduction instructions](architecture.md) keep actual reference checks
separate from source review and synthetic fixtures.

## Actual local checkpoints

All selected asset sizes and SHA-256 checksums were rechecked with bounded 1 MiB
reads against the existing operator provisioning receipt. Its pinned revisions
match the source-review metadata from the implementation children. No model was
downloaded, replaced, or modified in this acceptance run.

| Reference | Revision | Runtime fingerprinted bytes | Physical tensors | Graph nodes / edges | Coverage |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen3 GPTQ Int4 | `b9d87006067b0c0c2dea836370d6288e14f112ab` | 557,224,424 | 898 | 963 / 1,350 | Complete language scope; 28 blocks |
| Qwen3.5 NVFP4 | `2ac1e750cda67cc8538d731f6216f77b9c3a6f72` | 1,051,242,308 | 1,046 | 948 / 1,329 | Complete language scope; 24 hybrid blocks |
| V-JEPA 2 | `b3c1679b7c34d3255ef3547f27c7b226aefab26f` | 1,303,949,947 | 587 | 782 / 1,079 | Complete visual encoder/predictor scope; 24 + 12 blocks |
| SmolLM2 Base | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | 272,437,465 | 272 | 971 / 1,386 | Complete language scope; 30 blocks |

Language counts include one tokenizer context node. Qwen3.5 full-attention
instances are exactly 3, 7, 11, 15, 19, and 23; the other 18 retain the reviewed
linear-attention interiors. Numeric Qwen inventories remain explicitly partial.
V-JEPA has both stacks and no tokenizer requirement. Higher-rank patch/mask weights
remain metadata with unsupported-rank inspection.

[references.json](evidence/architecture/references.json) contains full content
fingerprints, runtime-selected inventories, separately counted provisioned assets,
quantization declarations, analyzer/description/schema revisions, source revision
identities, and per-model hash/analysis/cache timings. Runtime fingerprints cover
the backend-selected assets; README and chat-template files account for the
additional provisioned bytes and are listed separately.

## Startup and memory

The identified environment is Linux x86-64, Python 3.12.14, locked PyTorch CPU,
Intel Core i7-11700K (16 logical CPUs) and approximately 62.7 GiB host RAM. No inference or GPU numerical
work was required. Each process prepared the parent root containing all four
references; process readiness and memory therefore cover that root.

| Measurement run | Fresh-cache readiness | Warm-cache readiness | Cold peak RSS | Warm peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 observation | 12.200 s | 5.454 s | 395,024 KiB | 380,904 KiB |
| Qwen3.5 observation | 9.970 s | 5.328 s | 394,884 KiB | 379,976 KiB |
| V-JEPA observation | 9.755 s | 5.276 s | 395,264 KiB | 379,268 KiB |
| SmolLM2 observation | 9.755 s | 5.325 s | 394,832 KiB | 382,888 KiB |

Cold means a fresh application artifact cache, not a flushed OS page cache.
Warm graphs were byte-identical with no regeneration, and warm startup still
hashed model content. Linux `VmHWM` captures backend process peak RSS through
retrieval. Browser heap observations measure JS heap only, not whole-browser RSS
or physical GPU memory. These are observations, not throughput guarantees.

## Validation

- Independent OpenAPI, fixture regeneration and generated UI binding checks passed.
  The strict missing-reference gate was also exercised: all five absent
  architecture/regression prerequisites failed instead of skipping.
- Backend Ruff, formatting and strict mypy passed; **966 tests passed, 19 optional
  CUDA skips**.
- UI lint, typecheck and production build passed; **557 unit tests passed**.
- Complete HTTP acceptance with all four local architecture references: **24 passed,
  1 optional CUDA skip**. This includes actual SmolLM2 tensor/tokenizer and exact
  ordered/duplicate embedding-row regression. The ordinary model-free run was
  **19 passed, 6 explicit capability skips**.
- Component browser suite: **342 passed**, including native scrollbars at both
  DPRs. The delivery PR records the final production-browser verdict and
  exact-head CI; the local run preserves successful unchanged regression cases
  while rerunning corrected architecture cases.

The complete local fixture run also verifies read-only files and unchanged cache
publication across restart; exact native tensor bytes; guarded startup with no
model construction/tracing/tokenization/full tensor access; partial predictor,
unsupported architecture, stale-session, missing-cache and restart-required states.
Existing component suites retain independent source-derived topology assertions
and socket-readiness/cancellation checks. Large graphs, caches, weights and traces
remain outside Git.

## Harness observations

The first browser launch was blocked by the sandbox's X connection; headed checks
used host Xvfb. The full script's initial component port was occupied by an
unrelated service; the component and production suites use isolated ports/displays.
Neither event was an application failure.

The initial cancellation fixture used F32 direct delivery, bypassing the intended
conversion-producer barrier. Dense synthetic native weights now use exact F16
values so that the test actually holds progressive conversion before publication.
This changes no product behavior or numeric oracle.

A native pointer placed within a quarter CSS pixel of the V-JEPA matrix origin
rounded into row 1 on a fractional canvas boundary. Its reported value matched
that row, not the intended row 0. The reference value oracle now uses native
keyboard focus at scroll origin; existing pixel/pointer gates continue to cover
hit testing. This is a test-coordinate correction, not a scalar/rendering change.

## Actual graph layout observations

[layout.json](evidence/architecture/layout.json) records the four complete local
graphs expanded at DPR 1 in Chromium 153.0.8010.12, 1440×1000, SwiftShader.
These measurements were taken before opening numeric modals. They remain graph
layout observations even when a later numeric pointer-coordinate assertion was
corrected. The final native-value verdict belongs to the delivery PR.

| Graph | Worker packing time | JS heap snapshot |
| --- | ---: | ---: |
| smollm2 | 2.000 ms | 33,259,544 bytes |
| qwen3 | 3.100 ms | 33,129,148 bytes |
| qwen35 | 14.600 ms | 37,764,340 bytes |
| vjepa2 | 1.400 ms | 41,560,892 bytes |

Packing time excludes worker transfer, React composition, paint and numeric
inspection. Both compact and expanded views retain the entire backend graph;
viewport culling limits rendered DOM elements. Large Qwen embedding inspection
scenarios took minutes on software WebGL during the initial run; graph packing
times are not claims about the cost of transferring or inspecting those weights.
