# Issue #320: neuron inspector evidence

Source revision: `16d0dd34cabd28a7f78e58f67bb1d8ba28e8580c`. Screenshots were captured on 2026-10-27 from
the production Vite build with the real FastAPI backend, Chromium/SwiftShader,
DPR 1, and the complete local `HuggingFaceTB/SmolLM2-135M` Base checkpoint at
revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. No model artifacts are included.

| Viewport | Inspector | Native down weight |
| --- | --- | --- |
| 1440 × 960 | [Inspector](neuron-reference-inspector-1440.png) | [Column 427](neuron-reference-weight-1440.png) |
| 760 × 960 | [Scrolled inspector](neuron-reference-inspector-760.png) | [Column 427](neuron-reference-weight-760.png) |

The journey selects the concrete layer-7 MLP and neuron 427, opens gate/up row
427 and down column 427 through the existing native matrix modal, and returns
without changing graph camera or layout. All three real tensor, statistics and
distribution streams complete. The narrow inspector screenshot deliberately
shows internal scrolling. The matrix remains at native scale with normal scroll;
the orange column guide does not change scalar storage or authoritative values.

## Comparison with the issue mockup

The [accepted visual reference](https://raw.githubusercontent.com/murillo128/llm-model-explorer/48fc9f029b705c1a082f873a97f3a6761e8aee90/docs/design/neuron-explorer/reference.svg)
is illustrative. The implementation retains the live architecture graph and
browser, the existing 52 px header / 28 px footer, light tokens and focus styling.
The inspector sits beside the graph and stacks beneath it at narrow widths;
its content scrolls internally. Opening it does not recenter the graph, so
resizing the available canvas can clip the unchanged camera's visible region.
The formula is `a[j] = SiLU(g[j]) * u[j]`, before down. The lower experiment area
is a stable typed composition slot for #321; no illustrative activation results,
prompts or execution buttons are presented as working features here.

This backend revision omits the optional `graph_binding`. The real checkpoint
therefore uses the explicit concrete-module selector. The product does not infer
a graph join from names. The contract-fixture browser journey separately proves
the direct **Explore neurons** action for an exact typed graph/node binding on a
nonzero layer, and rejects structural/ambiguous mappings in unit tests.

## Validation ownership and outcomes

- API tests use the generated neuron schema and contract fixture, with independent
  wrong-session/model/revision and bounded-JSON cases.
- Selection tests own exact binding, actual index bounds, inventory shape and
  gate/up/down orientation. Component tests own invalid editing, unsupported
  capability/tokenizer explanations, same-module reselection and stale replies.
  The reselection regression failed with row 0 while the input displayed 2 before
  the fix, then passed with both the displayed coordinate and bound row equal to 2.
- Browser tests own actual keyboard focus, internal layout, scalar identity,
  row/column overlays, native matrix geometry and graph/tab/model transitions.
  Existing matrix and tokenizer tests retain their independent rendering oracles.
- One integrated checkpoint journey owns the real HTTP/inventory/production UI
  boundary at both viewports. It does not certify activation/intervention math or
  complete reference acceptance for the whole Neuron Explorer epic.

| Check | Observed result |
| --- | --- |
| API validator | OpenAPI valid; 81 neuron cases, 8 neuron payloads and 138 wire fixtures validated, alongside the existing contract cases |
| Generated bindings, typecheck, lint, build | Passed; build retains the existing large-chunk advisory |
| Complete UI unit suite | 66 files, 1,267 tests passed |
| Focused component-browser regressions | 55 passed |
| Selector's complete component-browser portfolio, desktop/narrow/native-scrollbar projects | 451 passed in the full run; seven architecture-shell fixture failures were corrected and all seven passed in a focused rerun (458 covered) |
| Real SmolLM2 production-browser journey | One case passed, covering both 1440 × 960 and 760 × 960 |

The architecture-shell mock previously answered every unknown endpoint with
`session_not_found`. Adding the new capability request exposed that incorrect
fixture response and correctly triggered the application's existing session
recovery. The fixture now returns the contract-owned unavailable capability for
its live sessions; its assertions were unchanged. No other component-browser
failure occurred. Typecheck and lint passed again after this fixture correction.

The full browser command was the native selector with the complete changed-path
array: `python3 .github/scripts/validation_selector.py --plan PLAN --run browser
--main`. The focused repair command was `npm run test:browser --
tests/architecture-shell.spec.ts`. Unit/fast validation used `npm run api:check`,
`npm run typecheck`, `npm run lint`, `npm test`, and `npm run build`. These are
scoped child results, not a claim that the full backend/integration or all
reference portfolios have run. The aggregate epic retains those CI gates.

Integration freshness: rebased onto `22373118f4664d0cae188bff87430b75172576a1`
(#319). The UI tree is unchanged by that integration update. On the rebased
source, `npm run check` passed again (1,267 unit tests), both neuron component
browser cases passed, and the real checkpoint journey passed again; the four
screenshots above were recaptured against that backend.

## Reproduction

Use the repository's locked dependencies and standard Node 24 toolchain. For the
real checkpoint journey, configure a private architecture-reference manifest as
documented in `acceptance/architecture.md`, build the UI, then run from `ui/`:

```sh
LMEX_ARCHITECTURE_REFERENCES=/path/to/private/manifest.json \
LMEX_REQUIRE_ARCHITECTURE_REFERENCES=1 \
npm run test:acceptance -- architecture.spec.ts --project=dpr1 \
  --grep 'complete local reference.*smollm2.*neuron selection'
```

The local run used an isolated checkpoint copy. Its local `_name_or_path` metadata
was removed to restore the exact reviewed upstream configuration required by the
reference validator; the copy's directory name supplies the public model identity.
Weights and tokenizer files were unchanged, and the original checkpoint was not
modified. The existing backend environment was reused with `PYTHONPATH` pointing
to this worktree's `backend/src`. Xvfb used the Mesa EGL vendor and software GL on
this host. These environment details do not alter the application contract.

Epic-child product CI is explicitly deferred by the four product workflows for
`codex/epic-issue-*` bases. The executor-routing workflow's path filters do not
match this change. Local results above are the child evidence; aggregate epic CI
and the final independent audit remain separate gates.
