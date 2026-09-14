# Live input-embedding UI evidence

Controlling contract: issue #44; accepted behavior is recorded in
`docs/spec/ui/tokenizer-explorer.md`. Synthetic fixtures establish UI association
and rendering behavior, not a claim about a particular model's tokenization.
Backend lookup correctness remains covered by the accepted embedding API/backend
fixtures and `backend/evidence/input-embeddings.md`.

## Frozen prompt baseline

The four small PNG snapshots under
`ui/tests/tokenizer-prompt-regression.spec.ts-snapshots/` were captured before
implementation at integration pin `8166d94bfae6b0f51da571dff614d900cf5f4375`.
`git diff a65025c0004fca179b874efe8a2353e91ae31f57 8166d94 -- ui/src/tokenizer`
was empty: this is the issue's pre-improvement prompt source in the accepted
compact shell. Both the empty and annotated prompt are compared at 1440×900 and
390×844. The annotated fixture includes an inserted special, overlapping Unicode
spans, and a duplicate token ID. The final comparison waits for the downstream
matrix to complete, preserves the 260px editor height, and asserts no document
overflow. The original prompt stylesheet, helper copy, and editor document/history
extensions remain unchanged.

## Reproduction

Use the locked Node version in `ui/.nvmrc`, install with `npm ci` in `ui`, then:

```sh
npm run check
UI_TEST_PORT=4310 npm run test:browser -- --project=desktop --project=narrow
UI_TEST_PORT=4310 xvfb-run -a npm run test:browser -- --project=native-scrollbars
```

Run browser commands sequentially because Playwright owns its report/artifact
directories. Native-scrollbar tests need an accessible X display. On this executor,
sandboxed Xvfb could not allocate WebGL; the native tests use host display access.
Do not regenerate the frozen prompt snapshots when checking this implementation.

## Covered observations

- Typed POST body and exact ordered ID echo validation, including duplicates,
  wrong order/length/identity, invalid input and empty response metadata.
- META and DATA in one network chunk, split float32 words, rendering before
  COMPLETE, exact per-cell values in asymmetric `[3,7]` fixtures at DPR 1 and 2.
- Token row `i` maps to sequence position `i`, including duplicate IDs, overlapping
  emoji byte tokens, and inserted specials. Existing IDs are used; no second
  token strip or source copy is introduced.
- Token hover/focus/activation highlights only the native row; GPU pixel readback
  verifies the other rows remain unchanged and no scalar upload occurs. Matrix
  arrow-key inspection links the existing ID and exposes exact coordinates/value.
- A→B→A source/options changes, delayed tokenization and embedding headers/data,
  late errors, session changes, and production-shell model A→B→A changes cannot
  revive old matrices. Known superseded operation IDs receive DELETE.
- Empty sequences, HTTP unsupported capability, backend cancellation, invalid
  identity, and partial terminal failures remain local to the embedding region.
- `[400,2048]` pending data retains native internal scroll coordinates. Editing
  releases all tested renderer CPU storage and disposes the previous renderer.
- Existing tokenizer tests retain clipboard, Unicode, selection, undo/redo, IME,
  and delayed-response coverage. The standalone test harness uses the compact
  shell's workspace track, rather than placing its prompt in the footer track.

The endpoint supplies no authoritative statistics/distribution artifact for these
matrices. Matrix-only rendering uses the shared provisional transfer and green
palette, with amber inspection guides. No browser model computation is added.
