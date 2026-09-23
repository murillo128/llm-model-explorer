import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import { contractInventory, contractResponse } from './architecture-fixtures';
import { models, sessionA, sessionB, tensors } from '../src/test/shell-fixtures';

const backend = 'https://pane-loading.example';
const preference = (page: Page, key: string, value: unknown) => page.addInitScript(({ key, value }) => {
  localStorage.setItem(key, JSON.stringify(value));
}, { key, value });
function gate() {
  let release!: () => void;
  const wait = new Promise<void>((resolve) => { release = resolve; });
  return { wait, release };
}
async function watchBounds(page: Page, selector: string) {
  await page.evaluate((target) => {
    const node = document.querySelector<HTMLElement>(target)!;
    const bounds = () => {
      const box = node.getBoundingClientRect();
      return [box.x, box.y, box.width, box.height];
    };
    const samples = [bounds()];
    new ResizeObserver(() => samples.push(bounds())).observe(node);
    Object.assign(window, { paneBoundsSamples: samples });
  }, selector);
}
async function unchangedBounds(page: Page, expected: { x: number; y: number; width: number; height: number }) {
  const samples = await page.evaluate(() => (window as typeof window & { paneBoundsSamples: number[][] }).paneBoundsSamples);
  expect(samples.length).toBeGreaterThan(0);
  expect(samples.every((box) => box.every((value, index) => Math.abs(value - [expected.x, expected.y, expected.width, expected.height][index]!) < 1))).toBe(true);
}
async function configure(page: Page, respond: (path: string, method: string, modelId?: string) => Promise<{ status?: number; json: unknown }>) {
  await page.route('**/runtime-config.json', (route) => route.fulfill({ json: { backend_base_url: backend } }));
  await page.route(`${backend}/**`, async (route) => {
    const request = route.request();
    const result = await respond(new URL(request.url()).pathname, request.method(), request.method() === 'POST' ? request.postDataJSON()?.model_id : undefined);
    await route.fulfill({ status: result.status ?? 200, json: result.json });
  });
}

test('inventory shell survives delayed session and inventory changes without replacing user choices', async ({ page }) => {
  const session = gate(), inventory = gate();
  let inventoryCalls = 0;
  await configure(page, async (path, method) => {
    if (path === '/models') return { json: { models } };
    if (path === '/sessions' && method === 'POST') { await session.wait; return { status: 201, json: sessionA }; }
    if (path.endsWith('/tensors')) { inventoryCalls++; await inventory.wait; return { json: { tensors, coverage: 'complete', diagnostics: [] } }; }
    return { json: sessionA };
  });
  await page.goto('/');
  const pane = page.locator('#tensor-inventory');
  const divider = page.locator('.tensor-layout > .inventory-resizer');
  await expect(pane).toBeVisible();
  await expect(divider).toHaveAttribute('aria-valuenow', page.viewportSize()!.width > 760 ? '280' : '200');
  await pane.evaluate((node) => { node.setAttribute('data-shell-marker', 'original'); });
  await page.getByRole('combobox', { name: 'Model' }).selectOption(models[0]!.id);
  await expect(pane.getByText('Loading model session…')).toBeVisible();
  await expect(pane).toHaveAttribute('data-shell-marker', 'original');
  session.release();
  await expect(pane.getByText('Loading tensor inventory…')).toBeVisible();
  if (page.viewportSize()!.width > 760) {
    await divider.focus(); await divider.press('ArrowRight');
    await expect(divider).toHaveAttribute('aria-valuenow', '296');
  }
  await pane.getByRole('button', { name: 'Collapse inventory' }).click();
  const restore = page.getByRole('button', { name: 'Expand inventory' });
  await expect(restore).toBeFocused();
  const center = page.locator('.tensor-layout > .working-surface');
  const bounds = (await center.boundingBox())!;
  await watchBounds(page, '.tensor-layout > .working-surface');
  inventory.release();
  await expect.poll(() => inventoryCalls).toBe(1);
  await expect(page.getByRole('button', { name: /left.weight/, includeHidden: true })).toBeAttached();
  await expect(restore).toBeVisible();
  await expect(restore).toBeFocused();
  await expect(pane).toHaveAttribute('data-shell-marker', 'original');
  expect(await center.boundingBox()).toEqual(bounds);
  await unchangedBounds(page, bounds);
  await restore.click();
  await expect(pane.getByRole('button', { name: /left.weight/ })).toBeVisible();
  if (page.viewportSize()!.width > 760) await expect(divider).toHaveAttribute('aria-valuenow', '296');
});

test('saved collapsed browser stays collapsed through graph retrieval and keeps its width on explorer return', async ({ page }) => {
  await preference(page, 'lmex.architecture-browser.pane', { visible: false, width: 344 });
  const graph = gate();
  let architectureCalls = 0;
  await configure(page, async (path, method) => {
    if (path === '/models') return { json: { models } };
    if (path === '/sessions' && method === 'POST') return { status: 201, json: sessionA };
    if (path.endsWith('/tensors')) return { json: contractInventory };
    if (path.endsWith('/architecture')) {
      architectureCalls++;
      await graph.wait;
      return { json: { ...contractResponse, model_id: sessionA.model_id } };
    }
    return { json: sessionA };
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  const pane = page.locator('#architecture-browser');
  const restore = page.getByRole('button', { name: 'Expand browser' });
  await expect(restore).toBeVisible();
  await pane.evaluate((node) => { node.setAttribute('data-shell-marker', 'original'); });
  await page.getByRole('combobox', { name: 'Model' }).selectOption(models[0]!.id);
  await expect.poll(() => architectureCalls).toBe(1);
  await restore.click();
  if (page.viewportSize()!.width > 760) {
    const divider = page.getByRole('separator', { name: 'Resize architecture browser' });
    await divider.focus(); await divider.press('ArrowRight');
    await expect(divider).toHaveAttribute('aria-valuenow', '360');
  }
  await pane.getByRole('button', { name: 'Collapse browser' }).click();
  await expect(restore).toBeVisible();
  const center = page.locator('.architecture-workspace > .working-surface');
  const bounds = (await center.boundingBox())!;
  await watchBounds(page, '.architecture-workspace > .working-surface');
  graph.release();
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  await expect(canvas).toHaveAttribute('aria-busy', 'false');
  await expect(restore).toBeVisible();
  await expect(restore).toBeFocused();
  await expect(pane).toHaveAttribute('data-shell-marker', 'original');
  expect(await center.boundingBox()).toEqual(bounds);
  await unchangedBounds(page, bounds);
  expect(architectureCalls).toBe(1);
  const layouts = await canvas.getAttribute('data-layout-count');
  await restore.click();
  await expect(pane.getByRole('searchbox', { name: 'Search components' })).toBeVisible();
  if (page.viewportSize()!.width > 760) {
    const divider = page.getByRole('separator', { name: 'Resize architecture browser' });
    await expect(divider).toHaveAttribute('aria-valuenow', '360');
    await divider.focus(); await divider.press('ArrowRight');
    await expect(divider).toHaveAttribute('aria-valuenow', '376');
  }
  await expect(canvas).toHaveAttribute('data-layout-count', layouts!);
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await expect(pane).toBeVisible();
  if (page.viewportSize()!.width > 760) await expect(page.getByRole('separator', { name: 'Resize architecture browser' })).toHaveAttribute('aria-valuenow', '376');
});

test('unavailable architecture and model replacement keep the browser shell without stale rows', async ({ page }) => {
  const modelB = gate();
  await configure(page, async (path, method, modelId) => {
    if (path === '/models') return { json: { models } };
    if (path === '/sessions' && method === 'POST') return { status: 201, json: modelId === sessionB.model_id ? sessionB : sessionA };
    if (path.endsWith('/tensors')) return { json: contractInventory };
    if (path.endsWith('/architecture')) {
      if (path.includes(sessionB.id)) { await modelB.wait; return { json: { status: 'unavailable', model_id: sessionB.model_id, reason: 'unsupported_architecture', requires_restart: false, diagnostics: [] } }; }
      return { json: { ...contractResponse, model_id: sessionA.model_id } };
    }
    return { json: sessionA };
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await page.getByRole('combobox', { name: 'Model' }).selectOption(models[0]!.id);
  await expect(page.getByRole('searchbox', { name: 'Search components' })).toBeVisible();
  const pane = page.getByRole('complementary', { name: 'Architecture browser' });
  await pane.getByRole('button', { name: 'Collapse browser' }).click();
  await expect(page.getByRole('button', { name: 'Expand browser' })).toBeVisible();
  await page.getByRole('combobox', { name: 'Model' }).selectOption(models[1]!.id);
  await expect(page.getByRole('button', { name: 'Expand browser' })).toBeVisible();
  await expect(pane.locator('.architecture-browser-row')).toHaveCount(0);
  modelB.release();
  await expect(page.locator('.architecture-capability-state')).toContainText('Architecture is not supported for this model.');
  await expect(page.getByRole('button', { name: 'Expand browser' })).toBeVisible();
});

test('architecture retrieval error and retry retain the open browser and pending user collapse', async ({ page }) => {
  await preference(page, 'lmex.architecture-browser.pane', { visible: true, width: 344 });
  const retry = gate();
  let architectureCalls = 0;
  await configure(page, async (path, method) => {
    if (path === '/models') return { json: { models } };
    if (path === '/sessions' && method === 'POST') return { status: 201, json: sessionA };
    if (path.endsWith('/tensors')) return { json: contractInventory };
    if (path.endsWith('/architecture')) {
      architectureCalls++;
      if (architectureCalls === 1) return { status: 503, json: { code: 'unavailable', message: 'temporarily unavailable' } };
      await retry.wait;
      return { json: { ...contractResponse, model_id: sessionA.model_id } };
    }
    return { json: sessionA };
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await page.getByRole('combobox', { name: 'Model' }).selectOption(models[0]!.id);
  const pane = page.locator('#architecture-browser');
  await expect(pane).toBeVisible();
  if (page.viewportSize()!.width > 760) await expect(page.getByRole('separator', { name: 'Resize architecture browser' })).toHaveAttribute('aria-valuenow', '344');
  await expect(pane.locator('.architecture-browser-state')).toContainText('Architecture browser unavailable. Retry retrieval.');
  await page.getByRole('button', { name: 'Retry retrieval' }).click();
  await expect.poll(() => architectureCalls).toBe(2);
  await pane.getByRole('button', { name: 'Collapse browser' }).click();
  await expect(page.getByRole('button', { name: 'Expand browser' })).toBeVisible();
  retry.release();
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('aria-busy', 'false');
  await expect(page.getByRole('button', { name: 'Expand browser' })).toBeVisible();
  expect(architectureCalls).toBe(2);
});

test('late graph for a replaced model cannot restore obsolete browser content or pane state', async ({ page }) => {
  const oldGraph = gate();
  let requestedA = false, deliveredA = false;
  await configure(page, async (path, method, modelId) => {
    if (path === '/models') return { json: { models } };
    if (path === '/sessions' && method === 'POST') return { status: 201, json: modelId === sessionB.model_id ? sessionB : sessionA };
    if (path.endsWith('/tensors')) return { json: contractInventory };
    if (path.endsWith('/architecture')) {
      if (path.includes(sessionA.id)) {
        requestedA = true;
        await oldGraph.wait;
        deliveredA = true;
        return { json: { ...contractResponse, model_id: sessionA.model_id } };
      }
      return { json: { ...contractResponse, model_id: sessionB.model_id, graph: { ...contractResponse.graph, graph_id: 'graph-b' } } };
    }
    return { json: sessionA };
  });
  await page.goto('/');
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await page.getByRole('combobox', { name: 'Model' }).selectOption(models[0]!.id);
  await expect.poll(() => requestedA).toBe(true);
  const pane = page.locator('#architecture-browser');
  await pane.getByRole('button', { name: 'Collapse browser' }).click();
  await page.getByRole('combobox', { name: 'Model' }).selectOption(models[1]!.id);
  const canvas = page.getByLabel('Architecture graph', { exact: true });
  await expect(canvas).toHaveAttribute('data-graph-id', 'graph-b');
  await expect(canvas).toHaveAttribute('aria-busy', 'false');
  oldGraph.release();
  await expect.poll(() => deliveredA).toBe(true);
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => resolve())));
  await expect(canvas).toHaveAttribute('data-graph-id', 'graph-b');
  await expect(page.getByRole('button', { name: 'Expand browser' })).toBeVisible();
  await expect(pane.locator('.architecture-browser-search')).toHaveCount(1);
});
