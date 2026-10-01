# Architecture hierarchy and operation-card evidence

Observed on 2026-10-01 with Python 3.12.14, Node 24.14.0 and the locked
Chromium build, using CPU and SwiftShader. Weights, caches, traces and screenshots
remain outside Git.

## Contracts and primary proof

| Contract / plausible defect | Independent oracle and owner |
| --- | --- |
| A targeted projection must remain one component, with exactly five children and real boundary forwarding | `backend/tests/test_lora_architecture.py`: explicit base/A/B/scale/add edges, q/v geometry and exact native/adapter storage names; both biased and unbiased bases |
| Bare projections and all original parameter bindings must survive | The same test compares bare/composed descriptors and untargeted leaves; existing dense topology, native/GPTQ parity and fail-closed LoRA cases remain active |
| Primitive cards must have a closed port/parameter/scalar vocabulary | `backend/tests/test_operation_cards.py`: explicit expected formulas on materialized dense graphs and signatures across dense, hybrid and visual fixtures; `CardSummary.test.tsx` verifies exposed scalar names |
| YARN scale must describe the configured computation | Existing DeepSeek graph regression checks the visible factor against a high-precision evaluation of the pinned source expression, `(1 + 0.1 * 0.707 * ln(40))² / sqrt(192)` |
| Ordinary browser expansion and matrix inspection must consume the actual graph | `ui/acceptance/lora-hierarchy.spec.ts`: production backend and built UI, collapsed/expanded projection, isolated component/Back, reshape/transpose/softmax cards, and real A/B binary requests |

On baseline `98a279fd44a01fadde33dbb8bfa8c6ac2986fb84`, the focused hierarchy
regression failed because the projection was a leaf, and the scale regression
failed on `y = (alpha / r) * x`. The nine primitive-card cases also failed on an
isolated baseline source snapshot: missing signatures and the `y`/`W` vocabulary.
Before its correction, the DeepSeek regression separately failed on the
unexplained `softmax_scale` formula. All pass with the final construction rules.

## Validation

- Backend Ruff and mypy passed. The full suite passed 1,911 tests with 20 CUDA
  skips before the final additional bias case and YARN metadata correction.
  After those changes, all 54 focused LoRA, primitive, grouping, template and
  DeepSeek tests passed, including the existing full graph/serialization budgets.
- `npm run check` passed: generated API bindings, TypeScript, lint, 1,023 unit
  tests and production build. TypeScript and lint were rerun after browser-test
  refinements.
- API validation passed: 365 references, 118 instance cases, 262 architecture
  cases and 132 reproducible wire fixtures. No API schema/binding changes.
- The production browser passed the deterministic adapted-projection case and
  the existing SmolLM2 binding, port-clearance, modal-lifetime, nested-return,
  session-replacement and Shared cases. One existing diagnostic case encountered
  a Chromium response-body transport error before assertions; its isolated rerun
  passed without changing assertions or adding retries.
- The same adapted-projection browser case passed against the existing local
  `HuggingFaceTB/SmolLM2-135M` and `hfm8tr/smollm2-135m-smoltalk-lora` pair:
  1,331 graph nodes, 2,016 edges, and actual q-projection A/B inspection. This is
  a focused presentation/binding smoke, not a new inference or full QLoRA
  reference campaign. Existing reference evidence remains authoritative.

Shared primitives use analyzer revision `static-graph-core-4`; dense bare
descriptions use revision 4, dense LoRA revision 2, and DeepSeek revision 3.
Numeric artifact keys and tensor identities are unchanged. Model-owned definitions
retain their authored formulas; unknown/composite operations retain source-specific
descriptions.

To reproduce the focused browser path, build `ui/`, then run from `ui/`:

```sh
UI_TEST_PORT=42690 xvfb-run -a npm run test:acceptance -- lora-hierarchy.spec.ts --project=dpr1
```

The optional `LMEX_LORA_REFERENCE_MODEL_ROOT` selects the existing local reference
pair instead of the deterministic fixture. It reads local files only.
