import { defineConfig, devices } from '@playwright/test';

const port = Number(process.env.UI_TEST_PORT ?? 4173);
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: './tests',
  // A deployment test edits dist/runtime-config.json and restores it afterward.
  workers: 1,
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'desktop', testIgnore: '**/tensor-explorer-scrollbars.spec.ts', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
    { name: 'narrow', testIgnore: '**/tensor-explorer-scrollbars.spec.ts', use: { ...devices['Desktop Chrome'], viewport: { width: 390, height: 844 } } },
    { name: 'native-scrollbars', testMatch: '**/tensor-explorer-scrollbars.spec.ts', use: { ...devices['Desktop Chrome'], headless: false, viewport: { width: 390, height: 844 } } },
  ],
  webServer: [{
    command: `npm run preview -- --port ${port} --strictPort`,
    url: baseURL,
    reuseExistingServer: false,
  }, {
    command: `npm run dev -- --port ${port + 1} --strictPort`,
    url: `http://127.0.0.1:${port + 1}`,
    reuseExistingServer: false,
  }],
});
