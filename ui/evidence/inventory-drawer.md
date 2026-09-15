# Tensor inventory drawer and rail

The expanded inventory stays in its own navigation track. Collapsing leaves a
40 CSS-pixel rail containing only the restore icon. The tooltip appears on hover
or keyboard focus; the matrix receives the reclaimed space. Expanded narrow
layouts retain the existing stacked panes.

## References

- Production matrix at 1440 × 1000: [expanded](inventory-matrix-expanded-dpr1.png),
  [collapsed](inventory-matrix-collapsed-dpr1.png). These use the integrated
  local acceptance backend and its deterministic 32 × 32 tensor.
- Built application, deep logical-path inventory at 1440 × 900:
  [expanded](inventory-expanded-1440.png), [collapsed](inventory-collapsed-1440.png).
- Same navigation fixture at 390 × 844:
  [expanded](inventory-expanded-390.png), [collapsed](inventory-collapsed-390.png).
  The navigation fixture intentionally selects a descriptor-only rank-3 tensor.

Screenshots are supporting references. DOM, accessibility, geometry, and resource
assertions are the gates; these fixtures do not establish local model acceptance.

## Validation

- API binding freshness, TypeScript, ESLint, all 665 unit tests, and production build pass.
- Inventory browser checks cover desktop/narrow layouts, an exact 40 px icon-only
  rail without persistent text or chevron, hover/focus tooltip, Enter/Space and
  focus transfer, saved width and branch restoration after reload, pointer and
  keyboard resizing, temporary width clamping, and document overflow down to
  280 × 400 with a mounted matrix in both drawer states.
- The active-stream test preserves the canvas, manual zoom extent, scroll origin,
  scalar allocation, selected tensor, and three original requests; new values
  continue arriving while collapsed without cancellation or restart.
- Related shell, tensor navigation/header, and native-scrollbar browser checks
  cover responsive bounds and scientific geometry.
- Integrated production inventory acceptance passes at DPR 1 and 2, preserving
  streamed panel geometry, GPU resources, saved width, and branch preferences.

Reproduce from `ui/` with Node 24.14 and locked dependencies:

```sh
npm run check
UI_TEST_PORT=18430 xvfb-run -a npm run test:browser -- \
  tests/tensor-inventory.spec.ts tests/tensor-navigation.spec.ts \
  tests/tensor-header.spec.ts tests/shell-viewport.spec.ts \
  tests/tensor-explorer-scrollbars.spec.ts
UI_TEST_PORT=18530 xvfb-run -a npm run test:acceptance -- \
  --grep 'integrated inventory preferences' --output /tmp/inventory-acceptance
```

The integrated test requires the repository's backend environment. Headed
Chromium needs a working Xvfb display; host execution was used because sandboxed
Xvfb/WebGL allocation was unavailable. Separate output directories avoid trace
cleanup collisions when component and integrated suites run concurrently.

Set `CAPTURE_INVENTORY_EVIDENCE=1` for either inventory browser suite to refresh
its references. Retain only the DPR-1 production screenshots; DPR-2 is checked
with the same geometry/resource assertions.
