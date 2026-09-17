import { expect, test } from '@playwright/test';
import { viewOptions, graphAction, graphPreference, findComponent } from './architecture-controls';

const harness = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/architecture.html`;
const panel = '[aria-label="Architecture graph"]';
const ready = async (page: import('@playwright/test').Page) => {
  await expect(page.locator(panel)).toHaveAttribute('aria-busy', 'false');
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
};
const camera = (page: import('@playwright/test').Page) => page.locator('.react-flow__viewport').getAttribute('style');

for (const kind of ['dense', 'hybrid', 'visual'] as const) test(`${kind}: declarations stay as ports across browser, dimensions, collapse and restore`, async ({ page }, info) => {
  const calls: string[] = []; page.on('request', (r) => { if (/\/sessions\//.test(r.url())) calls.push(r.url()); });
  await page.goto(`${harness}?fixture=interface-${kind}`); await ready(page);
  const removed = kind === 'visual' ? ['video', 'context_indices', 'target_indices', 'representations'] : ['Token IDs', 'positions', 'mask', 'current_mask', 'logits', 'tokenizer'];
  const absent = async () => { for (const id of removed) {
    await expect(page.locator(`.react-flow__node[data-id=${JSON.stringify(id)}]`)).toHaveCount(0);
    await expect(page.locator(`.architecture-browser-row[data-node-id=${JSON.stringify(id)}]`)).toHaveCount(0);
  } };
  await absent();
  await info.attach(`${kind}-interfaces`, { body: await page.screenshot(), contentType: 'image/png' });
  const name = kind === 'visual' ? 'video' : 'positions';
  const search = page.getByRole('searchbox', { name: 'Search components' });
  const before = await camera(page), layouts = await page.locator(panel).getAttribute('data-layout-count');
  await search.fill(name); await search.press('Enter');
  expect(await camera(page)).toBe(before);
  await expect(page.locator(panel)).toHaveAttribute('data-layout-count', layouts!);
  await absent();
  await page.getByRole('button', { name: 'Inspect selected', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Interface inspection' })).toContainText(name);
  await expect(page.getByRole('dialog')).toContainText('independent.fixture');
  await page.getByRole('button', { name: 'Close inspection' }).click();
  await search.fill('');
  for (const dimensions of [true, false]) {
    await graphPreference(page, 'Show dimensions', dimensions); await ready(page);
    await expect(page.locator('.architecture-port-label').filter({ hasText: name }).first()).toBeAttached();
    await absent();
  }
  await graphAction(page, 'Show all operations'); await ready(page); await absent();
  await expect(page.locator(`.architecture-browser-row[data-node-id=${JSON.stringify(kind === 'visual' ? 'predictor' : 'LM head')}]`)).toHaveCount(1);
  await graphAction(page, 'Collapse all'); await ready(page); await absent();
  await page.getByRole('button', { name: 'Toggle explorer' }).click();
  await page.getByRole('button', { name: 'Toggle explorer' }).click(); await ready(page); await absent();
  expect(calls).toEqual([]);
});

test('boundary hover/focus/pinning preserves layout and source connections; explicit isolation retains ports', async ({ page }) => {
  await page.goto(`${harness}?fixture=interface-hybrid`); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  const port = page.locator('.architecture-port[data-node-id="presentation:model"]').filter({ has: page.locator('.architecture-port-dot') }).nth(1);
  const before = await camera(page), layouts = await page.locator(panel).getAttribute('data-layout-count');
  await port.hover();
  await expect(page.locator('.architecture-connection[data-emphasized="true"]')).toHaveCount(2);
  await port.focus(); await port.press('Enter');
  await page.mouse.move(0, 0);
  await expect(page.locator('.architecture-connection[data-emphasized="true"]')).toHaveCount(2);
  expect(await camera(page)).toBe(before);
  await expect(page.locator(panel)).toHaveAttribute('data-layout-count', layouts!);
  await findComponent(page, 'language');
  await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
  await expect(page.locator('.architecture-port[data-node-id="language"][data-port-id="positions"]')).toHaveCount(1);
  await expect(page.locator('.architecture-node[data-presentation="external"]')).toHaveCount(1); // Real LM head.
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
});

test('many interfaces stay targetable without document overflow', async ({ page }) => {
  await page.goto(`${harness}?fixture=interface-many`); await ready(page);
  await viewOptions(page); await page.keyboard.press('Escape');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight)).toBe(true);
  const labels = page.locator('.react-flow__node[data-id="model"] .architecture-port-label');
  await expect(labels.filter({ hasText: 'auxiliary_17' })).toHaveCount(1);
  const unconnected = page.locator('.architecture-port[data-node-id="model"][aria-label*="auxiliary_17"]');
  await unconnected.focus(); await unconnected.press('Enter');
  await expect(unconnected).toHaveAttribute('data-emphasized', 'true');
  await expect(page.locator('.architecture-connection[data-emphasized="true"]')).toHaveCount(0);
  const ports = await page.locator('.architecture-port').evaluateAll((items) => items.map((e) => ({ label: e.getAttribute('aria-label'), top: (e as HTMLElement).style.top })));
  expect(ports.length).toBeGreaterThanOrEqual(23);
  expect(new Set(ports.map((p) => p.label)).size).toBe(ports.length);
});
