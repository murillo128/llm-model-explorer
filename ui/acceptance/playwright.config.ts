import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  testMatch: 'product.spec.ts',
  workers: 1,
  timeout: 90_000,
  expect: { timeout: 15_000 },
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  outputDir: '../test-results/acceptance',
  reporter: [['list'], ['json', { outputFile: '../test-results/acceptance.json' }]],
  use: {
    baseURL: 'http://127.0.0.1:4175',
    headless: false,
    viewport: { width: 1440, height: 1000 },
    trace: 'retain-on-failure', screenshot: 'only-on-failure',
    launchOptions: { args: [`--use-angle=${process.env.LMEX_WEBGL_BACKEND ?? 'swiftshader'}`, '--enable-unsafe-swiftshader'] },
  },
  projects: [{ name: 'dpr1', use: { deviceScaleFactor: 1 } },
    { name: 'dpr2', use: { deviceScaleFactor: 2 } }],
  webServer: {
    command: 'node acceptance/static.mjs',
    cwd: new URL('..', import.meta.url).pathname,
    url: 'http://127.0.0.1:4175',
    reuseExistingServer: false,
  },
});
