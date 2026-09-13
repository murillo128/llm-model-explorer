# Progressive Tensor Explorer evidence

Validated on Linux with Node 24.14.0, npm 11.9.0, and Playwright's Chromium 1243.
The browser fixture uses the actual React app, typed API client, LMEX decoder,
and WebGL2 renderer with controlled stream timing. It does not require model
weights or claim real-backend integration.

Reproduce from `ui/` with `npm ci`, `npm run check`, then
`npx playwright install --with-deps chromium` and
`xvfb-run -a npm run test:browser` on Linux without a display.
The focused suite is `npm run test:browser -- tensor-explorer.spec.ts`.
Local execution used `UI_TEST_PORT=4314` and an already installed matching browser
via `PLAYWRIGHT_BROWSERS_PATH=/tmp/issue-3-playwright`. Due to host disk exhaustion,
this issue’s installed dependencies, generated caches/build output and browser temporary files were held under
`/dev/shm/issue-14-browser`; these local paths do not change the tests.

| Evidence | Observed result |
| --- | --- |
| API generation, strict types, lint, unit tests, production build | Passed; 178 unit/component cases |
| Complete browser suite | 100 passed, including 18 Tensor Explorer desktop/narrow cases and 4 headed native-scrollbar cases |
| Asymmetric `[2,3]` matrix | Values and framebuffer pixels preserve native C order; split words decode progressively |
| Distribution sections | Cross-section chunks preserve `[rows,100]` and `[100,columns]`; pending suffix differs from zero |
| Delayed statistics | Matrix displays first; successful statistics update without additional scalar allocation/upload |
| GPU storage | One R32F allocation for the small tensor, two R32UI profile allocations; large uint32 counts preserved |
| Reference `[576,1536]` | Logical data geometry is 1536×576, 100×576 and 1536×100, independent of viewport/DPR |
| Scrolling/bands | Coordinate and GPU-pixel assertions at top, middle, band transitions and bottom/right; 128-pixel texture ceiling forces both band axes |
| Outcomes | Auxiliary failure leaves completed tensor inspectable; failed/cancelled prefixes stay incomplete; private error text stays hidden |
| Generations and disposal | A→B→A stale tensor/statistics/distribution callbacks ignored; exactly owned operations cancelled and all textures released |
| Special cases | Rank-1 strip requests only tensor/statistics; empty/unsupported descriptors start no work; allocation failure is explicit and cleans resources |
| Density | Fixed log1p(count)/log1p(axis length), including zero-axis behavior; no client histogram computation |

The screenshot references below were inspected alongside programmatic geometry
and pixel assertions. Their synthetic matrix pattern is test data, not a claim
about model structure. The reference screenshots deliberately constrain the
visible viewport to exercise scrolling while preserving the full data extent.

- [Desktop reference](tensor-explorer-desktop.png)
- [Narrow reference](tensor-explorer-narrow.png)

CI retains the complete browser report and per-test screenshots as its
`ui-browser-evidence` artifact. Hover, chroma selection and magnifier interactions
remain separate work. Backend integration is outside this fixture-based screen
validation.

## Audit regressions

The catalogue-refresh regression first reproduced disposal of a populated
renderer on both desktop and narrow layouts. It now holds catalogue loading open
and checks both loading and completion while the tensor prefix is streaming and
after cancellation. The same three operation handles, scalar/count allocations,
upload totals, renderer instances and received values survive all four shell
updates. Controller identity depends on the actual client/session/selection
inputs, not the props wrapper created by React.

The headed `native-scrollbars` project first reproduced zero visible rows for a
complete `[1536]` strip at DPR 1 and 2. It checks that native scrollbar thickness
is positive, the exact scientific height is visible, and scrolling reaches the
last value for `[1536]` and `[2,1536]`. It also removes/restores horizontal overflow
by resizing, without scalar reallocation. The main scroller uses intrinsic auto
height so the browser adds native scrollbar chrome outside the data extent,
with a responsive viewport ceiling. CI runs the suite under Xvfb and retains
per-test screenshots plus measured scrollbar/data geometry as attachments.
