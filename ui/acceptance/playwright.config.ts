import { defineConfig } from '@playwright/test';

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
const workers = process.env.PLAYWRIGHT_ACCEPTANCE_WORKERS
  ? Number(process.env.PLAYWRIGHT_ACCEPTANCE_WORKERS)
  : 1;

export default defineConfig({
  testDir: '.',
  testMatch: 'product.spec.ts',
  workers,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  outputDir: '../test-results/acceptance',
  reporter: [['list'], ['json', { outputFile: '../test-results/acceptance.json' }]],
  use: {
    headless: false,
    viewport: { width: 1440, height: 1000 },
    trace: 'retain-on-failure', screenshot: 'only-on-failure',
    launchOptions: { args: [`--use-angle=${process.env.LMEX_WEBGL_BACKEND ?? 'swiftshader'}`, '--enable-unsafe-swiftshader'] },
  },
  projects: [
    { name: 'dpr1', use: { baseURL: `http://127.0.0.1:${dpr1.ui}`, deviceScaleFactor: 1 } },
    { name: 'dpr2', use: { baseURL: `http://127.0.0.1:${dpr2.ui}`, deviceScaleFactor: 2 } },
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
