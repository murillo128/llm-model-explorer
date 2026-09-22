import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import { models, sessionA, tensors } from '../src/test/shell-fixtures';

const footer = (page: Page) => page.getByRole('contentinfo', { name: 'Application status' });
async function geometry(page: Page) {
  return page.evaluate(() => {
    const rect = (selector: string) => {
      const { x, y, width, height } = document.querySelector(selector)!.getBoundingClientRect();
      return { x, y, width, height };
    };
    return { app: rect('.app-bar'), workspace: rect('.workspace-frame'), footer: rect('.app-status-bar'),
      session: rect('.session-status'), card: rect('.working-surface'),
      document: [document.documentElement.scrollWidth, document.documentElement.scrollHeight],
      body: [document.body.scrollWidth, document.body.scrollHeight], viewport: [innerWidth, innerHeight] };
  });
}

test('footer observes real attempts, accessible retry and HTTP errors without connection toasts', async ({ page }) => {
  let release!: () => void;
  let pending = new Promise<void>(resolve => { release = resolve; });
  let result: 'transport' | 'http' | 'success' = 'transport';
  let requests = 0;
  await page.route('**/runtime-config.json', route => route.fulfill({ json: { backend_base_url: 'https://backend.example' } }));
  await page.route('**/models', async route => {
    requests++;
    await pending;
    if (result === 'transport') return route.abort('connectionrefused');
    if (result === 'http') return route.fulfill({ status: 503, json: { code: 'resource_exhausted', message: 'private' } });
    return route.fulfill({ json: { models } });
  });
  await page.goto('/');
  await expect(footer(page).getByRole('status').first()).toHaveText('○Connecting…');
  const initial = await geometry(page);
  expect(initial.app.height).toBe(52); expect(initial.footer.height).toBe(28);
  expect(initial.document).toEqual(initial.viewport); expect(initial.body).toEqual(initial.viewport);
  release();
  await expect(footer(page)).toContainText('Disconnected');
  for (let attempt = 0; attempt < 2; attempt++) {
    pending = new Promise<void>(resolve => { release = resolve; });
    const retry = page.getByRole('button', { name: 'Retry connection' });
    await retry.focus(); await page.keyboard.press('Enter');
    await expect(footer(page)).toContainText('Reconnecting…');
    await expect(retry).toBeDisabled();
    await page.keyboard.press('Enter');
    expect(requests).toBe(attempt + 2);
    expect(await geometry(page)).toMatchObject({ app: initial.app, workspace: initial.workspace, footer: initial.footer, session: initial.session, document: initial.viewport, body: initial.viewport });
    release();
    await expect(footer(page)).toContainText('Disconnected');
  }
  result = 'http';
  await page.getByRole('button', { name: 'Retry connection' }).click();
  await expect(footer(page)).toContainText('Connected');
  await expect(page.locator('.app-toast')).toHaveCount(0);
  result = 'success';
  await page.getByRole('button', { name: 'Refresh models' }).click();
  await expect(page.getByRole('combobox')).toBeEnabled();
  await expect(page.locator('.app-toast')).toHaveCount(0);
  expect(await geometry(page)).toMatchObject({ app: initial.app, workspace: initial.workspace, footer: initial.footer, session: initial.session, document: initial.viewport, body: initial.viewport });
});

test('operation toast and catalogue recovery preserve numeric consumer, geometry and camera', async ({ page }, testInfo) => {
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html`);
  await page.getByRole('combobox').selectOption('lab/alpha');
  await page.getByRole('button', { name: /^reference \[/ }).click();
  await expect.poll(() => page.evaluate(() => window.explorerFixture.requests.length)).toBe(3);
  const canvas = await page.locator('.matrix-scroll canvas').elementHandle();
  const box = (await page.locator('.matrix-scroll canvas').boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel(0, -180);
  const camera = () => page.evaluate(() => {
    const f = window.explorerFixture;
    return { view: f.renderers[0]!.view, renderers: f.renderers.length, requests: f.requests.length,
      cancelled: f.cancelled, allocations: f.metrics.scalarAllocations };
  });
  await expect.poll(async () => (await camera()).view!.scaleX).toBeGreaterThan(1);
  await page.mouse.move(0, 0);
  const before = await camera();
  const bounds = await geometry(page);
  const metadata = await page.locator('.status-metadata').boundingBox();
  await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).focus();
  await page.evaluate(() => window.explorerFixture.requests[1]!.stream.error(new TypeError('private transport error')));
  await expect(page.locator('.app-toast')).toHaveCount(1);
  await expect(page.locator('.app-toast').getByRole('alert')).toContainText('connection was lost');
  await expect(page.getByRole('button', { name: 'Tensor Explorer', exact: true })).toBeFocused();
  await expect(footer(page)).toContainText('Disconnected');
  await expect(page.locator('[data-result="statistics"]')).toHaveAttribute('data-state', 'failed');
  expect(await camera()).toEqual(before);
  expect(await geometry(page)).toEqual(bounds);
  expect(await page.locator('.status-metadata').boundingBox()).toEqual(metadata);
  await page.screenshot({ path: testInfo.outputPath('operation-feedback.png') });
  for (let attempt = 0; attempt < 2; attempt++) {
    await page.evaluate(() => { window.explorerFixture.catalogue.failed = true; window.explorerFixture.catalogue.paused = true; });
    await page.getByRole('button', { name: 'Retry connection' }).click();
    await expect(footer(page)).toContainText('Reconnecting…');
    await page.evaluate(() => window.explorerFixture.catalogue.resume());
    await expect(footer(page)).toContainText('Disconnected');
    await expect(page.locator('.app-toast')).toHaveCount(1);
  }
  await page.evaluate(() => { window.explorerFixture.catalogue.failed = false; });
  await page.getByRole('button', { name: 'Retry connection' }).click();
  await expect(footer(page)).toContainText('Connected');
  await page.getByRole('button', { name: 'Dismiss notification' }).focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('.app-toast')).toHaveCount(0);
  await expect(page.locator('[data-result="statistics"]')).toHaveAttribute('data-state', 'failed');
  expect(await camera()).toEqual(before);
  expect(await geometry(page)).toEqual(bounds);
  expect(await canvas!.evaluate(node => node === document.querySelector('.matrix-scroll canvas'))).toBe(true);
  await page.getByRole('button', { name: 'Cancel loading' }).click();
  await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'cancelled');
  await expect(page.locator('.app-toast')).toHaveCount(0);
  await expect(footer(page)).toContainText('Connected');
});

test('failed close remains retryable after dismissal and success timer never reaches a replacement model', async ({ page }) => {
  let fails = true;
  let deletes = 0;
  await page.clock.install();
  await page.route('**/runtime-config.json', route => route.fulfill({ json: { backend_base_url: 'https://backend.example' } }));
  await page.route('https://backend.example/**', route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === '/models') return route.fulfill({ json: { models } });
    if (request.method() === 'DELETE') { deletes++; return fails ? route.abort('connectionrefused') : route.fulfill({ status: 204 }); }
    if (path === '/sessions') return route.fulfill({ status: 201, json: sessionA });
    if (path.endsWith('/tensors')) return route.fulfill({ json: { tensors, coverage: 'complete', diagnostics: [] } });
    return route.fulfill({ json: sessionA });
  });
  await page.goto('/');
  await page.getByRole('combobox').selectOption(models[0]!.id);
  await expect(page.getByText('Session active.')).toBeVisible();
  await page.getByRole('button', { name: 'Session options', exact: true }).click();
  await page.getByRole('button', { name: 'Close session', exact: true }).click();
  await expect(page.locator('.app-toast')).toHaveCount(1);
  await page.clock.runFor(20000);
  await expect(page.getByRole('button', { name: 'Retry close' })).toBeVisible();
  await page.getByRole('button', { name: 'Dismiss notification' }).click();
  await expect(footer(page)).toContainText('Disconnected');
  await expect(page.getByText('Session active.')).toBeVisible();
  await page.getByRole('button', { name: 'Session options', exact: true }).click();
  await page.getByRole('button', { name: 'Close session', exact: true }).click();
  await expect(page.locator('.app-toast')).toHaveCount(1);
  fails = false;
  await page.getByRole('button', { name: 'Retry close' }).focus(); await page.keyboard.press('Enter');
  await expect(page.getByText('Session closed.')).toBeVisible();
  expect(deletes).toBe(3);
  await page.getByRole('combobox').selectOption(models[0]!.id);
  await expect(page.getByText('Session active.')).toBeVisible();
  await page.clock.runFor(20000);
  await expect(page.locator('.app-toast')).toHaveCount(0);
  await expect(footer(page)).toContainText('Connected');
});
