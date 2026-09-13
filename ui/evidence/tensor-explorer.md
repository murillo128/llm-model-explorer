# Progressive Tensor Explorer evidence

Validated on Linux with Node 24.14.0, npm 11.9.0, and Playwright's Chromium 1243.
The browser fixture uses the actual React app, typed API client, LMEX decoder,
and WebGL2 renderer with controlled stream timing. It does not require model
weights or claim real-backend integration.

Reproduce from `ui/` with `npm ci`, `npm run check`, then
`npx playwright install chromium` and `npm run test:browser`.
The focused suite is `npm run test:browser -- tensor-explorer.spec.ts`.
Local execution used `UI_TEST_PORT=4314` and an already installed matching browser
via `PLAYWRIGHT_BROWSERS_PATH=/tmp/issue-3-playwright`; neither changes the tests.

| Evidence | Observed result |
| --- | --- |
| API generation, strict types, lint, unit tests, production build | Passed; 175 unit/component cases |
| Focused browser suite | 16 passed across desktop/narrow projects, with explicit DPR 1/2 contexts |
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
