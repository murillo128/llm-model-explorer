import { readFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

const configPath = new URL('../dist/runtime-config.json', import.meta.url);
async function assetHashes() {
  const root = new URL('../dist/assets/', import.meta.url);
  const files = (await readdir(root)).sort();
  return Promise.all(files.map(async (file) => [file,
    createHash('sha256').update(await readFile(new URL(file, root))).digest('hex'),
  ]));
}
async function mockConfig(page: Page, body = '{"backend_base_url":"https://backend.example/"}', status = 200) {
  await page.route('**/models', (route) => route.fulfill({ json: { models: [], diagnostics: [] } }));
  await page.route('**/runtime-config.json', (route) => route.fulfill({ status, contentType: 'application/json', body }));
}

// Both browser viewports verify runtime deployment values against the same production assets.
test('one production build accepts two deployed backend URLs', async ({ page }, testInfo) => {
  const deployedConfig = JSON.parse(await readFile(configPath, 'utf8')) as { backend_base_url?: unknown };
  expect(deployedConfig.backend_base_url).toEqual(expect.any(String));
  const before = await assetHashes();
  const backendRequests: string[] = [];
  let deployedBackend = 'https://models-a.example/api/';
  page.on('request', (request) => {
    if (new URL(request.url()).origin !== new URL(testInfo.project.use.baseURL!).origin) backendRequests.push(request.url());
  });
  await page.route('**/models', (route) => route.fulfill({ json: { models: [], diagnostics: [] } }));
  await page.route('**/runtime-config.json', (route) => route.fulfill({
    contentType: 'application/json', body: JSON.stringify({ backend_base_url: deployedBackend }),
  }));
  for (const backend of ['https://models-a.example/api/', 'http://192.0.2.10:9000///']) {
    deployedBackend = backend;
    await page.goto('/');
    await expect(page.getByTestId('backend-url')).toHaveText(backend.replace(/\/+$/, ''));
    await expect(page.getByText('No models available on this backend.')).toBeVisible();
  }
  expect(await assetHashes()).toEqual(before);
  expect(backendRequests).toEqual(['https://models-a.example/api/models', 'http://192.0.2.10:9000/models']);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('neutral-shell.png'), fullPage: true });
  await testInfo.attach('neutral shell', { path: testInfo.outputPath('neutral-shell.png'), contentType: 'image/png' });
});

test('keyboard navigation has visible focus and switches explorer slots', async ({ page }) => {
  await mockConfig(page);
  await page.goto('/');
  await expect(page.getByRole('navigation')).toBeVisible();
  await page.keyboard.press('Tab');
  const skip = page.getByRole('link', { name: 'Skip to workspace' });
  await expect(skip).toBeFocused();
  await expect(skip).toBeInViewport();
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: 'Tensor Explorer', exact: true })).toBeFocused();
  await page.keyboard.press('Tab');
  const tokenizer = page.getByRole('button', { name: 'Tokenizer Explorer', exact: true });
  await expect(tokenizer).toBeFocused();
  expect(await tokenizer.evaluate((element) => getComputedStyle(element).outlineStyle)).toBe('solid');
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: 'Tokenizer Explorer', exact: true })).toHaveAttribute('aria-current', 'page');
  await expect(page.getByText('Open a model session to use this explorer.')).toBeVisible();
  await expect(tokenizer).toHaveAttribute('aria-current', 'page');
  await page.reload();
  await expect(page.getByRole('navigation')).toBeVisible();
  await page.keyboard.press('Tab');
  await page.keyboard.press('Enter');
  await expect(page.getByRole('main')).toBeFocused();
});

for (const config of [
  { name: 'missing file', status: 404, body: '', message: 'Publish the file alongside the UI' },
  { name: 'missing URL', status: 200, body: '{}', message: 'Set backend_base_url' },
  { name: 'invalid URL', status: 200, body: '{"backend_base_url":"file:///private/models"}', message: 'Set backend_base_url' },
  { name: 'malformed JSON', status: 200, body: '<html>fallback</html>', message: 'not valid JSON' },
]) {
  test(`actionable configuration error: ${config.name}`, async ({ page }) => {
    await mockConfig(page, config.body, config.status);
    await page.goto('/');
    await expect(page.getByRole('alert')).toContainText(config.message);
    await expect(page.getByRole('navigation')).toHaveCount(0);
    await expect(page.getByText('/private/models', { exact: false })).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.unroute('**/runtime-config.json');
    await mockConfig(page);
    await page.getByRole('button', { name: 'Retry configuration' }).click();
    await expect(page.getByRole('button', { name: 'Tensor Explorer', exact: true })).toHaveAttribute('aria-current', 'page');
  });
}

test('configuration loading is visible before the explorer can initialize', async ({ page }) => {
  let release!: () => void;
  const pending = new Promise<void>((resolve) => { release = resolve; });
  await page.route('**/runtime-config.json', async (route) => {
    await pending;
    await route.fulfill({ contentType: 'application/json', body: '{"backend_base_url":"https://backend.example"}' });
  });
  await page.goto('/');
  await expect(page.getByRole('status')).toContainText('Loading application configuration');
  await expect(page.getByRole('navigation')).toHaveCount(0);
  release();
  await expect(page.getByRole('button', { name: 'Tensor Explorer', exact: true })).toHaveAttribute('aria-current', 'page');
});
