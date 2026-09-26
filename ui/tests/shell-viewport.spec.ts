import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import { models, sessionA, tensors } from '../src/test/shell-fixtures';

async function geometry(page: Page) {
  const bounds = await page.evaluate(() => {
    const rect = (selector: string) => {
      const { top, bottom, height, left, right } = document.querySelector(selector)!.getBoundingClientRect();
      return { top, bottom, height, left, right };
    };
    return { top: rect('.app-bar'), bottom: rect('.app-status-bar'), workspace: rect('.workspace-frame'),
      width: innerWidth, height: innerHeight,
      document: [document.documentElement.scrollWidth, document.documentElement.scrollHeight],
      body: [document.body.scrollWidth, document.body.scrollHeight], scroll: [scrollX, scrollY] };
  });
  expect(bounds.document).toEqual([bounds.width, bounds.height]);
  expect(bounds.body).toEqual([bounds.width, bounds.height]);
  expect(bounds.scroll).toEqual([0, 0]);
  expect(bounds.top.top).toBe(0);
  expect(bounds.top.height).toBe(52);
  expect(bounds.bottom.height).toBe(28);
  expect(bounds.bottom.bottom).toBe(bounds.height);
  expect(bounds.workspace.top).toBe(bounds.top.bottom);
  expect(bounds.workspace.bottom).toBe(bounds.bottom.top);
  expect(bounds.workspace.height).toBe(bounds.height - 80);
  if (bounds.height >= 840) expect(bounds.workspace.height / bounds.height).toBeGreaterThanOrEqual(.85);
  for (const name of ['Architecture Explorer', 'Tokenizer Explorer', 'Tensor Explorer', 'Refresh models', 'Session options']) {
    await expect(page.getByRole('button', { name, exact: true })).toBeInViewport({ ratio: 1 });
  }
  await expect(page.getByRole('combobox', { name: 'Model' })).toBeInViewport({ ratio: 1 });
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 1000, height: 840 },
  { width: 390, height: 640 }, { width: 280, height: 400 }]) {
  test(`fixed shell contains normal states at ${viewport.width}×${viewport.height}`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);
    let releaseModels!: () => void;
    const pendingModels = new Promise<void>((resolve) => { releaseModels = resolve; });
    let catalogue: 'populated' | 'empty' | 'failed' = 'populated';
    let refreshes = 0;
    const deleted: string[] = [];
    await page.route('**/runtime-config.json', (route) => route.fulfill({ json: { backend_base_url: 'https://backend.example' } }));
    await page.route('https://backend.example/**', async (route) => {
      const request = route.request();
      const path = new URL(request.url()).pathname;
      if (path === '/models') {
        refreshes++;
        await pendingModels;
        if (catalogue === 'failed') return route.abort('connectionrefused');
        return route.fulfill({ json: { models: catalogue === 'empty' ? [] : [{ ...models[0], size_bytes: 272400000 }], diagnostics: [] } });
      }
      if (request.method() === 'DELETE') { deleted.push(path); return route.fulfill({ status: 204 }); }
      if (path === '/sessions') {
        expect(request.postDataJSON()).toEqual({ model_id: models[0]!.id });
        return route.fulfill({ status: 201, json: sessionA });
      }
      if (path.endsWith('/tensors')) return route.fulfill({ json: { coverage: 'complete', diagnostics: [], tensors: Array.from({ length: 40 }, (_, i) =>
        ({ ...tensors[2], id: `cube-${i}`, name: `layer-${i}.cube`, path: [`layer-${i}`, 'cube'] })) } });
      if (path.endsWith('/tokenize')) return route.fulfill({ json: { ...request.postDataJSON(), tokens: [] } });
      return route.fulfill({ json: sessionA });
    });
    await page.goto('/');
    await expect(page.getByText('Loading models…')).toBeVisible();
    await geometry(page);
    releaseModels();
    const model = page.getByRole('combobox', { name: 'Model' });
    await expect(model).toBeEnabled();
    await model.focus();
    expect(await model.evaluate((element) => getComputedStyle(element).outlineStyle)).toBe('solid');
    await model.selectOption(models[0]!.id);
    await expect(page.getByText('Session active.')).toBeVisible();
    await page.getByRole('button', { name: /layer-0.cube/ }).click();
    await expect(page.getByRole('heading', { name: 'layer-0 › cube' })).toBeVisible();
    await geometry(page);
    const inventory = page.getByRole('complementary', { name: 'Tensor inventory' });
    expect(await inventory.evaluate((element) => element.scrollHeight > element.clientHeight)).toBe(true);
    await inventory.evaluate((element) => { element.scrollTop = element.scrollHeight; });
    await geometry(page);
    await expect(page.getByRole('heading', { level: 1 })).toHaveCount(0);
    await expect(page.locator('.surface-label')).toHaveCount(0);
    await expect(page.locator('.app-status-bar')).not.toContainText(models[0]!.id);
    if (viewport.width === 1440) {
      await page.screenshot({ path: testInfo.outputPath('compact-active-shell.png') });
      await testInfo.attach('compact active shell', { path: testInfo.outputPath('compact-active-shell.png'), contentType: 'image/png' });
    }
    await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Tokenizer Explorer', exact: true })).toHaveAttribute('aria-current', 'page');
    await expect(page.getByRole('region', { name: 'Tokenizer Explorer workspace', exact: true })).toBeVisible();
    await geometry(page);

    const refresh = page.getByRole('button', { name: 'Refresh models' });
    await refresh.focus(); await page.keyboard.press('Enter');
    await expect.poll(() => refreshes).toBe(2);
    const options = page.getByRole('button', { name: 'Session options', exact: true });
    await options.focus(); await page.keyboard.press('Enter');
    await expect(options).toHaveAttribute('aria-expanded', 'true');
    await expect(page.getByText('272,400,000', { exact: true })).toBeVisible();
    await geometry(page);
    await page.keyboard.press('Escape');
    await expect(options).toBeFocused();
    await expect(options).toHaveAttribute('aria-expanded', 'false');
    await page.keyboard.press('Enter'); await page.keyboard.press('Tab');
    await expect(page.getByRole('button', { name: 'Close session' })).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(page.getByText('Session closed.')).toBeVisible();
    expect(deleted).toEqual([`/sessions/${sessionA.id}`]);
    await expect(options).toBeFocused();
    await geometry(page);
    for (const result of ['empty', 'failed'] as const) {
      catalogue = result;
      await refresh.click();
      await expect(page.getByText(result === 'empty' ? 'No models available on this backend.' :
        'Could not load models. The backend is unreachable or returned an invalid response.')).toBeVisible();
      await geometry(page);
    }
  });
}
