# Integrated explorer polish acceptance

This acceptance increment joins the completed Tensor/Tokenizer polish and model
coverage changes on the post-Architecture-presentation baseline. It changes test
fixtures, harnesses and evidence; product algorithms and scientific expectations
remain owned by `docs/spec/`.

## Reproduce

Use the locked environments described in [README.md](README.md). The normal
`acceptance/check.sh` covers backend quality gates, API conformance, UI unit and
component browser tests, real TCP tests, and the production browser at DPR 1/2.

Focused additions:

```sh
backend/.venv/bin/python -m pytest acceptance/test_polish.py -ra
(cd ui && npm run build && xvfb-run -a npm run test:acceptance -- --grep 'polish|expanded model coverage')
```

Set `UI_TEST_PORT` to an unused base port to isolate browser servers. Use a single
headed browser worker on a dedicated Xvfb display. On this host, Xvfb's automatic
EGL vendor selection crashes inside NVIDIA initialization in the sandbox. The
software path is reproducible with these environment settings:

```sh
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export LIBGL_ALWAYS_SOFTWARE=1
```

Chromium still uses the suite's SwiftShader WebGL2 backend. This is software
rendering evidence, not CUDA or physical GPU performance evidence.

The long-prompt case waits up to 45 seconds for complete embeddings: a captured
DPR 2 run finished its HTTP response in 1.3 seconds and completed all 630 UI row
uploads after about 22 seconds. Geometry cases disable the observer's unused
full-framebuffer readback, while existing pixel-oracle cases retain it. Completion,
row count, current-state and geometry assertions remain mandatory; this suite
does not impose a physical-GPU latency requirement on SwiftShader.

The component centering readiness check uses the same one-device-pixel outer
containment allowance as its final assertions. A 390×844 CSS viewport produced
an integer-height profile ending at 607 px beside a fractional pane ending at
606.796875 px. All matrix position terms were exact; the earlier readiness check
incorrectly applied its 0.03 px coordinate tolerance to that boundary too. Exact
centering, cell scale, profile alignment and final containment checks remain.

## Coverage map

| Required proof | Acceptance entry point |
| --- | --- |
| Compact stationary header, moved extrema, aligned truthful rulers, drawer retention | Existing `product.spec.ts` inventory/metadata and distribution cases; new `polish inventory` cases verify both requested widths and capture both drawer states |
| Centered underfilled geometry, attached profiles, linked matrix/range drag, Escape/right-click history | Existing production matrix-navigation and camera cases; new `polish magnifier` sweeps edges after scrolling, resizing and DPR change, then replaces the source |
| Prompt content sizing, bounded scrolling, accessible manual split/reset, matrix independence | New `polish real tokenizer` tests use actual tokenizer and embedding responses at 1178/1440 px and a constrained 640 px width |
| Stale token/embedding replacement and sequence-position linkage | Existing production A→B→A, progressive row, stale continuity and model/session fencing tests; new packed-checkpoint browser test links duplicate token positions |
| Native/GPTQ/NVFP4 logical bytes, statistics and distributions | Existing native TCP oracles plus `test_packed_inventory_data_analysis_and_embedding_rows_over_tcp`: cold/warm bytes, float64 statistics and exact uint32 bin counts |
| Honest complete/partial coverage | Packed fixtures split every physical companion into indexed shards; adding an unknown I32 record requires partial diagnostics and 404 for its numeric endpoints |
| Architecture-aware input tables | `test_text_families_and_non_text_capability_over_tcp`: all three text mappings, native and quantized checkpoints, first/last/order/duplicate/empty rows; V-JEPA rejects embeddings while retaining graph and tensors |
| Packed input tables | Both packed formats serve ordered embedding rows through the same public TCP endpoint, compared byte-for-byte with the independent scalar oracle |
| Architecture weight modal and graph compatibility | Existing production `architecture.spec.ts` opens native and decoded bindings, checks exact values and graph identity, and verifies focus/camera/resource restoration |

## Oracle boundaries

`polish_fixtures.py` reuses the reviewed synthetic packed format fixtures from
`backend/tests/quantized_oracles.py`. Their scalar Python/struct decoder does not
call production decoder helpers. Full tensor and ordered-row comparisons use
exact float32 bytes, preserving signed zero. Statistics use independent NumPy
float64 calculations with the existing numerical tolerance; distributions use
independent scalar bin assignment and exact little-endian uint32 bytes.

The tiny text tokenizer is an authored six-ID WordLevel fixture. It fits every
synthetic vocabulary and produces repeated IDs at distinct sequence positions.
Native-only Qwen fixtures intentionally contain just an input table; they prove
numeric capability, not complete architecture coverage. Full reduced graph
fixtures remain available for Qwen packed checkpoints and V-JEPA.

## Optional local references

Use the external manifest documented in [architecture.md](architecture.md).
`LMEX_REFERENCE_MODEL_DIR` selects SmolLM2 Base for its existing TCP/browser smoke.
`LMEX_ARCHITECTURE_REFERENCES` enables the four architecture reference checks and
the new `test_local_quantized_checkpoint_input_rows` cases. The latter compare
Qwen3/Qwen3.5 input rows with independent, bounded Safetensors slices, including
first/last/duplicate/empty rows. They do not infer packed-layer correctness from
the checkpoints' native embedding tables.

The separate `backend/scripts/check_quantized_reference.py` checks real packed
layers against scalar references with hash-pinned upstream provenance files.
Keep those files, manifests, checkpoints, caches, raw traces and logs outside Git.
Absent optional configuration is SKIP; supplied invalid references fail. Fixture
success does not establish actual-reference acceptance or CUDA support.

## Results and inspected captures

The 2026-09-15 run uses Python 3.12.14, PyTorch 2.14.0+cpu, Node 24.14.0 and
locked UI dependencies. Runtime source is unchanged from integration commit
`26871e1a36b4f5b3bb12b5e607d48ccaa64c6251`; acceptance additions run against
that same runtime. Raw logs, traces and local checkpoint manifests remain outside
Git.

| Gate | Observed result |
| --- | --- |
| Backend Ruff, format and mypy | PASS; unchanged backend validated in an isolated snapshot |
| Backend unit tests | 1,216 PASS; 19 CUDA SKIP |
| API validation and generated UI binding check | PASS; 283 references, 88 instance cases, 144 architecture cases, 76 wire fixtures |
| UI typecheck, lint and production build | PASS |
| UI unit tests | 681 PASS |
| Component browsers | 469 initial PASS and the readiness mismatch described above; all 30 affected centering cases PASS after correction |
| Production browser fixtures, DPR 1/2 | 66 PASS; 10 optional reference cases SKIP in this fixture-only invocation |
| Actual-reference production browser, DPR 1 | 5 PASS: four architecture/native-weight-modal cases and SmolLM2 Tensor/Tokenizer smoke |
| Acceptance Ruff and format | PASS |
| Full real TCP suite | 31 PASS: 24 deterministic cases and 7 actual-reference cases; 1 CUDA SKIP |

Actual-checkpoint evidence is separate in [local-references.json](evidence/polish/local-references.json).
All four approved checkpoints passed complete metadata/inventory and cold/warm
architecture checks. SmolLM2 passed its existing tensor/embedding/tokenizer smoke;
Qwen3 and Qwen3.5 passed the new exact ordered input-row checks. The separate
[packed comparison](evidence/polish/quantized-reference-comparison.json) compared
1,105,522 GPTQ and 68,722 NVFP4 logical float32 values with zero canonical bit
mismatches. It covers one complete projection per format plus selected ranges
across other layers, not whole-checkpoint inference equivalence. Its precision
notes distinguish canonical float32 bits from upstream inference rounding and
negative-zero LUT conventions.

Actual-reference browser checks were run at DPR 1; actual-reference DPR 2 and
CUDA were not validated. The mandatory deterministic browser run covers both
DPR settings. Full component-browser checks also run on the published PR head in
CI; the local corrected run isolates the 30 cases affected by the readiness fix.

Selected production captures use deterministic fixtures, DPR 1 and a 900 px
viewport height. All eight were visually inspected in addition to the automated
geometry, state-retention, resource, focus and overflow assertions.

| Width | Tensor Inventory | Tokenizer split |
| --- | --- | --- |
| 1178 px | [Expanded](evidence/polish/tensor-expanded-1178.png), [collapsed](evidence/polish/tensor-collapsed-1178.png) | [Automatic](evidence/polish/tokenizer-auto-1178.png), [manual](evidence/polish/tokenizer-manual-1178.png) |
| 1440 px | [Expanded](evidence/polish/tensor-expanded-1440.png), [collapsed](evidence/polish/tensor-collapsed-1440.png) | [Automatic](evidence/polish/tokenizer-auto-1440.png), [manual](evidence/polish/tokenizer-manual-1440.png) |

The Tensor captures show the compact breadcrumb header, selected inventory leaf,
40 px icon rail, reclaimed matrix space, and attached marginal profiles with
native scrollbar clearance. The Tokenizer captures show the short prompt in a
compact panel and the same eight-row embedding matrix after a 64 px splitter
drag. The thin, vertically centered matrix preserves square cells and fills the
available width; extra workspace does not stretch its eight logical rows.
