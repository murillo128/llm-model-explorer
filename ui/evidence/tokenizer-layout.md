# Adaptive Tokenizer Explorer panels

Issue #112 replaces fixed panel heights with content-aware allocation and a
keyboard/pointer divider. The editor and shared Matrix Explorer stay mounted
while the split changes. CodeMirror measures wrapped content; the prompt grows
to its workspace-relative cap and then scrolls internally. Manual allocation is
local to the mounted workspace and double activation restores automatic sizing.

## Reproduction and checks

Use the locked Node/npm dependencies and Chromium specified by `ui/package.json`
and `ui/playwright.config.ts`, then run from `ui/`:

```sh
npm ci
npm run check
npx playwright test tokenizer-layout.spec.ts tokenizer.spec.ts \
  tokenizer-embeddings.spec.ts tokenizer-model-switch.spec.ts \
  tokenizer-prompt-regression.spec.ts shell-viewport.spec.ts \
  --project=desktop --project=narrow
```

The layout tests load the production build in the real application shell with
deterministic API fixtures. They assert content growth/capping/internal scrolling,
remaining embeddings height, pointer and keyboard bounds, accessible divider
semantics, reset with current content, browser resize in automatic/manual modes,
mount-local preference lifetime, stable stale/current geometry, token↔row linkage,
matrix camera independence, and no document overflow. Existing tokenizer checks
retain native editing/clipboard/IME, generation fencing, streaming and resource
lifetime coverage. Screenshot comparison also covers the compact inline
source/bracket/ID surface. Geometry and accessibility assertions are the gate.

## Built UI references

Local checks passed: the UI check (600 unit tests plus API generation,
typecheck, lint and build), all 12 new layout scenarios, and the 60 existing
tokenizer/shell scenarios. The offscreen camera fixtures explicitly reserve a
small embeddings viewport so their scroll/reveal assertions remain meaningful
with the larger automatic embeddings allocation.
The targeted real backend/browser prompt acceptance also passed at DPR 1 and 2:
embedding completion preserves prompt pixels, selection, history and composition
inside the adaptive allocation.

These small screenshots show the production workspace at 1440×900 and 390×844.
The word spans, IDs and embeddings are synthetic test fixtures, not model results.
Short matrices retain the shared Matrix Explorer geometry at this issue's base;
this change does not implement the sibling underfilled-content camera work.

| Viewport | Short, automatic | Short, manual | Long, automatic | Long, manual |
| --- | --- | --- | --- | --- |
| Desktop | [Reference](tokenizer-layout/desktop-short-auto.png) | [Reference](tokenizer-layout/desktop-short-manual.png) | [Reference](tokenizer-layout/desktop-long-auto.png) | [Reference](tokenizer-layout/desktop-long-manual.png) |
| Narrow | [Reference](tokenizer-layout/narrow-short-auto.png) | [Reference](tokenizer-layout/narrow-short-manual.png) | [Reference](tokenizer-layout/narrow-long-auto.png) | [Reference](tokenizer-layout/narrow-long-manual.png) |

The automatic long-prompt references show the editor scrolled internally. Manual
references use the divider; the long-prompt example leaves the minimum useful
embeddings area. Playwright regenerates these as per-test attachments.
