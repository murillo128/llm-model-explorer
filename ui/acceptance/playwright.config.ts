import { defineConfig } from '@playwright/test';
import { resolve } from 'node:path';

const requestedPortfolio = process.env.LMEX_TEST_PORTFOLIO ?? 'routine';
// Explicit reference inputs (including invalid ones) and required mode must not
// disappear behind routine filtering. Full/reference commands validate them.
const referencesRequested = Boolean(process.env.LMEX_REFERENCE_MODEL_DIR ||
  process.env.LMEX_ARCHITECTURE_REFERENCES || process.env.LMEX_LORA_REFERENCE_MODEL_ROOT ||
  process.env.LMEX_REQUIRE_ARCHITECTURE_REFERENCES === '1');
const portfolio = requestedPortfolio === 'routine' && referencesRequested ? 'full' : requestedPortfolio;
if (!['routine', 'extended', 'full'].includes(portfolio)) throw new Error(`Invalid portfolio: ${portfolio}`);
const traceRequested = process.argv.some((argument, i) =>
  argument.startsWith('--trace=') ? argument !== '--trace=off' :
    argument === '--trace' && process.argv[i + 1] !== 'off');
const phase = traceRequested ? 'diagnostic' : process.env.LMEX_TEST_PHASE ?? portfolio;
if (!['routine', 'extended', 'full', 'diagnostic'].includes(phase)) throw new Error(`Invalid evidence phase: ${phase}`);
if (phase !== 'diagnostic' && phase !== portfolio) throw new Error(`Evidence phase ${phase} disagrees with portfolio ${portfolio}`);
const evidenceRoot = process.env.LMEX_EVIDENCE_DIR
  ? resolve(process.env.LMEX_EVIDENCE_DIR, 'browser')
  : new URL('../test-results/acceptance/', import.meta.url).pathname;
const phaseRoot = resolve(evidenceRoot, phase);

const componentPort = Number(process.env.UI_TEST_PORT ?? 4173);
const isolatedPorts = Boolean(process.env.UI_TEST_PORT);
const dpr1 = {
  ui: isolatedPorts ? componentPort + 2 : 4175,
  backend: isolatedPorts ? componentPort + 3 : 8765,
};
const dpr2 = {
  ui: isolatedPorts ? componentPort + 4 : 4177,
  backend: isolatedPorts ? componentPort + 5 : 8767,
};
// Isolate native window/pointer interactions on the shared X display. Parallel
// headed runs showed intermittent loss of transient inspection/drag state.
const workers = process.env.PLAYWRIGHT_ACCEPTANCE_WORKERS
  ? Number(process.env.PLAYWRIGHT_ACCEPTANCE_WORKERS)
  : 1;

export default defineConfig({
  testDir: '.',
  testMatch: '**/*.spec.ts',
  workers,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  ...(portfolio === 'extended' ? { grep: /@extended/ } : {}),
  ...(portfolio === 'routine' ? { grepInvert: /@extended/ } : {}),
  outputDir: resolve(phaseRoot, 'artifacts'),
  reporter: [['list'], ['json', { outputFile: resolve(phaseRoot, 'report.json') }]],
  use: {
    headless: false,
    viewport: { width: 1440, height: 1000 },
    // --trace on is an explicit focused diagnostic; passing routine runs are trace-free.
    trace: 'off', screenshot: 'only-on-failure',
    launchOptions: { args: [`--use-angle=${process.env.LMEX_WEBGL_BACKEND ?? 'swiftshader'}`, '--enable-unsafe-swiftshader'] },
  },
  projects: [
    { name: 'dpr1', use: { baseURL: `http://127.0.0.1:${dpr1.ui}`, deviceScaleFactor: 1 } },
    // DPR 1 owns all cases, including newly added untagged tests. Only explicit
    // physical-pixel/interaction contracts need a second density invocation.
    { name: 'dpr2', grep: /@density/, use: { baseURL: `http://127.0.0.1:${dpr2.ui}`, deviceScaleFactor: 2 } },
  ],
  webServer: [{
    command: `LMEX_STATIC_PORT=${dpr1.ui} LMEX_BACKEND_PORT=${dpr1.backend} node acceptance/static.mjs`,
    cwd: new URL('..', import.meta.url).pathname,
    url: `http://127.0.0.1:${dpr1.ui}`,
    reuseExistingServer: false,
  }, {
    command: `LMEX_STATIC_PORT=${dpr2.ui} LMEX_BACKEND_PORT=${dpr2.backend} node acceptance/static.mjs`,
    cwd: new URL('..', import.meta.url).pathname,
    url: `http://127.0.0.1:${dpr2.ui}`,
    reuseExistingServer: false,
  }],
});
