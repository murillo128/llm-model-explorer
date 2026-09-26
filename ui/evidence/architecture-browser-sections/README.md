# Architecture Browser section and row evidence

Captured from the authored, contract-valid `browser-vjepa` fixture with Node
24.14.0, npm 11.9.0, Playwright 1.63.0, and the repository's deterministic
SwiftShader configuration.

- `desktop.png` uses the 1440 × 900 desktop project and shows the collapsed
  Model section, both repetition rows (24 encoder and 12 predictor instances),
  and the four distinct encoder/predictor attention/MLP Shared families.
- `narrow.png` uses the 390 × 844 narrow project and records the real bounded
  upper-pane viewport without scaling or widening the browser.

The focused browser test asserts equal section typography/disclosure geometry,
common row gutter/icon/minimum-height metrics, consistent count metadata, and
the distinct accessible Model selection, Explore stack, family-selection, and
component-selection actions. It also verifies that section disclosure preserves
selection, projection, layout invocation count, camera, scope, and HTTP numeric
consumer activity; search restores the prior section/tree presentation.

Reproduce from `ui/` after `npm ci`:

```sh
npm run build
UI_TEST_PORT=4373 npx playwright test tests/architecture-browser.spec.ts \
  --project=desktop --project=narrow
```
