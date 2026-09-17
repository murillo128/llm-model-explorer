import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import { openShared, findComponent, graphAction } from './architecture-controls';
import { makeProjectionFixture } from './architecture-projection-fixture';

const harness = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/architecture.html`;
const panel = (page: Page) => page.getByLabel('Architecture graph', { exact: true });
const card = (page: Page, id: string) => page.locator(`.react-flow__node[data-id=${JSON.stringify(id)}]`);
async function ready(page: Page) {
  await expect(panel(page)).toHaveAttribute('aria-busy', 'false');
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
}
async function state(page: Page) {
  await ready(page);
  return page.evaluate(() => ({
    count: document.querySelector('[data-layout-count]')?.getAttribute('data-layout-count'),
    camera: document.querySelector('.react-flow__viewport')?.getAttribute('style'),
    scope: document.querySelector('[data-scope-id]')?.getAttribute('data-scope-id'),
    focus: document.querySelector('[aria-label="Architecture focus"]')?.textContent,
    expanded: [...document.querySelectorAll('.architecture-node[data-expanded="true"]')].map((e) => e.parentElement?.getAttribute('data-id')),
    inspection: document.querySelector('output')?.textContent,
  }));
}
async function selectOnly(page: Page, id: string) {
  const before = await state(page), node = card(page, id);
  for (const target of ['.architecture-node-label', '.architecture-node-heading']) {
    await node.locator(target).click(target === '.architecture-node-heading' ? { position: { x: 2, y: (await node.locator(target).boundingBox())!.height - 2 } } : {});
    await expect(node.locator('.architecture-node')).toHaveAttribute('data-selected', 'true');
    await expect(page.locator('.architecture-node[data-selected="true"]')).toHaveCount(1);
    await expect(page.getByRole('dialog')).toHaveCount(0);
    expect(await state(page)).toEqual(before);
  }
}
test.beforeEach(async ({ page }) => { await page.goto(harness); await ready(page); });

test('label and body select only for source, expanded, repetition and derived cards in both modes', async ({ page }) => {
  await selectOnly(page, 'repeat:layers:0:1');
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click(); await ready(page);
  await expect(card(page, 'repeat:layers:0:1').locator('.architecture-node')).toHaveAttribute('data-selected', 'true');
  await findComponent(page, 'layer1'); await ready(page);
  await selectOnly(page, 'layer1');
  await card(page, 'layer1').locator('.architecture-expand').click(); await ready(page);
  await selectOnly(page, 'layer1');
  await selectOnly(page, 'linear1');
  await card(page, 'layer1').locator('.architecture-navigate').click(); await ready(page);
  await selectOnly(page, 'layer1'); await selectOnly(page, 'linear1');
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('connections'); await ready(page);
  await findComponent(page, 'layer-3.gate'); await ready(page);
  await graphAction(page, 'Focus MLP'); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  await selectOnly(page, 'mlp:layer-3.gate');
  await card(page, 'mlp:layer-3.gate').locator('.architecture-expand').click(); await ready(page);
  await selectOnly(page, 'mlp:layer-3.gate');
  await card(page, 'mlp:layer-3.gate').locator('.architecture-navigate').click(); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  await selectOnly(page, 'mlp:layer-3.gate'); await selectOnly(page, 'layer-3.gate');
});

test('real double-clicks toggle once, preserve zoom and leave leaf inspection explicit', async ({ page }) => {
  const before = await state(page);
  await card(page, 'repeat:layers:0:1').locator('.architecture-node-label').dblclick(); await ready(page);
  expect(Number((await state(page)).count)).toBe(Number(before.count) + 1);
  await expect(page.locator('output')).toBeEmpty();
  await findComponent(page, 'layer1'); await ready(page);
  const collapsed = await state(page);
  await card(page, 'layer1').locator('.architecture-node-heading').dblclick({ position: { x: 2, y: (await card(page, 'layer1').locator('.architecture-node-heading').boundingBox())!.height - 2 } }); await ready(page);
  await expect(card(page, 'layer1').locator('.architecture-expand')).toHaveAttribute('aria-expanded', 'true');
  const expanded = await state(page);
  expect(Number(expanded.count)).toBe(Number(collapsed.count) + 1);
  expect(expanded.camera?.match(/scale\(([^)]+)\)/)?.[1]).toBe(collapsed.camera?.match(/scale\(([^)]+)\)/)?.[1]);
  await expect(page.locator('output')).toBeEmpty();
  await card(page, 'linear1').locator('.architecture-node-heading').dblclick({ position: { x: 2, y: (await card(page, 'linear1').locator('.architecture-node-heading').boundingBox())!.height - 2 } }); await ready(page);
  expect(await state(page)).toEqual(expanded);
  await card(page, 'layer1').locator('.architecture-node-label').dblclick(); await ready(page);
  await expect(card(page, 'layer1').locator('.architecture-expand')).toHaveAttribute('aria-expanded', 'false');
  const contracted = await state(page);
  expect(Number(contracted.count)).toBe(Number(expanded.count) + 1);
  expect(contracted.camera?.match(/scale\(([^)]+)\)/)?.[1]).toBe(expanded.camera?.match(/scale\(([^)]+)\)/)?.[1]);
  await expect(page.locator('output')).toBeEmpty();
  await expect(page.getByRole('dialog')).toHaveCount(0);
});

test('nested expansion and exact hidden selection survive parent controls, explorer switching and Back', async ({ page }) => {
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('components'); await ready(page);
  await findComponent(page, 'layer-3'); await ready(page);
  await card(page, 'layer-3').locator('.architecture-navigate').click(); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  await card(page, 'layer-3.attention').locator('.architecture-node-label').dblclick(); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  await card(page, 'layer-3.attention.Q').locator('.architecture-node-label').click();
  const before = await state(page);
  const position = (await card(page, 'layer-3').boundingBox())!;
  await card(page, 'layer-3').locator('.architecture-expand').dblclick(); await ready(page);
  await expect(card(page, 'layer-3.attention')).toHaveCount(0);
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-3.attention.Q');
  await expect(card(page, 'layer-3').locator('.architecture-node')).toHaveAttribute('data-selected', 'false');
  const contracted = await state(page), at = (await card(page, 'layer-3').boundingBox())!;
  expect(Number(contracted.count)).toBe(Number(before.count) + 1);
  expect(Math.abs(position.x - at.x)).toBeLessThan(1);
  expect(Math.abs(position.y - at.y)).toBeLessThan(1);
  expect(contracted.camera?.match(/scale\(([^)]+)\)/)?.[1]).toBe(before.camera?.match(/scale\(([^)]+)\)/)?.[1]);
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Toggle explorer', exact: true }).click(); await ready(page);
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-3.attention.Q');
  // Preserve the retained overview camera, where header controls can be smaller
  // than a device pixel on narrow screens; reopen through the keyboard control.
  await card(page, 'layer-3').locator('.architecture-expand').focus();
  await page.keyboard.press('Enter'); await ready(page);
  await expect(card(page, 'layer-3.attention').locator('.architecture-expand')).toHaveAttribute('aria-expanded', 'true');
  await expect(card(page, 'layer-3.attention.Q').locator('.architecture-node')).toHaveAttribute('data-selected', 'true');
  const restored = await state(page);
  await card(page, 'layer-3.attention').locator('.architecture-navigate').click(); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  await card(page, 'layer-3.attention').locator('.architecture-node-label').dblclick(); await ready(page);
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
  const back = await state(page);
  expect(back).toEqual({ ...restored, count: back.count });
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-3.attention.Q');
  await expect(page.locator('output')).toBeEmpty();
});

test('empty groups have no toggle, remain selectable and inspect only explicitly by keyboard', async ({ page }) => {
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('empty-group'); await ready(page);
  const empty = card(page, 'empty'), before = await state(page);
  await expect(empty.locator('.architecture-expand')).toHaveCount(0);
  await selectOnly(page, 'empty');
  await empty.locator('.architecture-node-label').dblclick();
  await empty.locator('.architecture-node-heading').dblclick({ position: { x: 2, y: (await empty.locator('.architecture-node-heading').boundingBox())!.height - 2 } });
  await empty.locator('.architecture-node-label').focus();
  await page.keyboard.press('ArrowRight'); await page.keyboard.press('ArrowLeft');
  expect(await state(page)).toEqual(before);
  await empty.locator('.architecture-info').focus(); await page.keyboard.press('Enter');
  await expect(page.locator('output')).toHaveText('empty-group: empty');
});

test('contracting after Show all operations leaves sibling detail visible and reopens completely', async ({ page }) => {
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('connections'); await ready(page);
  await graphAction(page, 'Show all operations'); await ready(page);
  await findComponent(page, 'layer-3.attention'); await ready(page);
  const graph = makeProjectionFixture({ count: 4 });
  const children = graph.nodes.filter((n) => n.parent_id === 'layer-3.attention');
  const before = await state(page);
  await expect(panel(page)).toHaveAttribute('data-visible-nodes', String(graph.nodes.length - 4));
  await graphAction(page, 'Toggle selected group'); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-visible-nodes', String(graph.nodes.length - 4 - children.length));
  expect(Number((await state(page)).count)).toBe(Number(before.count) + 1);
  await graphAction(page, 'Toggle selected group'); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-visible-nodes', String(graph.nodes.length - 4));
  expect(Number((await state(page)).count)).toBe(Number(before.count) + 2);
  await expect(page.locator('output')).toBeEmpty();
  await expect(page.getByRole('dialog')).toHaveCount(0);
});

test('header navigation targets the pressed card and explores its nested operation; controls stay ordered and focused', async ({ page }, info) => {
  await findComponent(page, 'linear1'); await ready(page);
  await card(page, 'layer1').locator('.architecture-node-label').click();
  const operation = card(page, 'linear1'), nav = operation.locator('.architecture-navigate');
  await expect(nav).toHaveAccessibleName('Explore component: linear1 (linear1)');
  await expect(nav).toHaveAttribute('title', 'Explore component: linear1 (linear1)');
  await expect(nav).toBeEnabled();
  for (const id of ['linear1', 'layer1']) {
    const geometry = await card(page, id).locator('.architecture-node-heading > button').evaluateAll((buttons) =>
      buttons.map((button) => { const r = button.getBoundingClientRect(); return { x: r.x, right: r.right, height: r.height }; }));
    for (let i = 1; i < geometry.length; i++) expect(geometry[i]!.x).toBeGreaterThanOrEqual(geometry[i - 1]!.right);
    expect(geometry.every((r) => r.height > 0)).toBe(true);
  }
  expect(await operation.locator('.architecture-node-heading > button').evaluateAll((buttons) => buttons.map((b) => b.classList.item(2))))
    .toEqual(['architecture-node-label', 'architecture-navigate', 'architecture-info']);
  await nav.focus(); await page.keyboard.press('Tab'); await page.keyboard.press('Shift+Tab');
  expect(await nav.evaluate((e) => getComputedStyle(e).outlineStyle)).toBe('solid');
  const modelHeader = info.outputPath('model-header.png');
  await operation.locator('.architecture-node-heading').screenshot({ path: modelHeader });
  await info.attach('model-header', { path: modelHeader, contentType: 'image/png' });
  await page.keyboard.press('Enter'); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', 'linear1');
  await expect(page.locator('output')).toBeEmpty();
  await expect(nav).toHaveAccessibleName('View in model: linear1 (linear1)');
  const isolatedHeader = info.outputPath('isolated-header.png');
  await operation.locator('.architecture-node-heading').screenshot({ path: isolatedHeader });
  await info.attach('isolated-header', { path: isolatedHeader, contentType: 'image/png' });
  await nav.click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', '');
  await card(page, 'layer1').locator('.architecture-navigate').click(); await ready(page);
  expect(await card(page, 'layer1').locator('.architecture-node-heading > button').evaluateAll((buttons) => buttons.map((b) => b.classList.item(2))))
    .toEqual(['architecture-node-label', 'architecture-expand', 'architecture-navigate', 'architecture-info']);
  await card(page, 'layer1').locator('.architecture-node-label').click();
  await nav.dblclick(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', 'linear1');
  await expect(nav).toHaveAccessibleName('View in model: linear1 (linear1)');
  await nav.click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', '');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'linear1');
  await expect(page.locator('.architecture-node[data-selected="true"]')).toHaveCount(1);
  await expect(operation.locator('.architecture-node')).toHaveAttribute('data-selected', 'true');
  const bounds = (await operation.boundingBox())!, flow = (await page.locator('.architecture-flow').boundingBox())!;
  expect(Math.abs(bounds.x + bounds.width / 2 - flow.x - flow.width / 2)).toBeLessThan(2);
  await expect(page.locator('output')).toBeEmpty();
});

test('independent controls and ports do not bubble; keyboard expansion and disabled targets remain truthful', async ({ page }) => {
  const range = card(page, 'repeat:layers:0:1'), navigation = range.locator('.architecture-navigate');
  await expect(navigation).toHaveAttribute('aria-disabled', 'true');
  await expect(navigation).toHaveAttribute('title', /no component navigation target/);
  const before = await state(page);
  // Send real pointer events despite Playwright's aria-disabled actionability guard.
  await navigation.dblclick({ force: true }); expect(await state(page)).toEqual(before);
  await findComponent(page, 'layer1'); await ready(page);
  const group = card(page, 'layer1'), label = group.locator('.architecture-node-label');
  await label.focus(); await page.keyboard.press('Enter');
  await page.keyboard.press('ArrowRight'); await ready(page);
  await expect(group.locator('.architecture-expand')).toHaveAttribute('aria-expanded', 'true');
  const expanded = await state(page);
  await card(page, 'linear1').locator('.architecture-port').first().dblclick();
  expect(await state(page)).toEqual(expanded);
  await card(page, 'linear1').locator('.architecture-info').dblclick();
  await expect(page.locator('output')).toContainText(': linear1');
  expect((await state(page)).count).toBe(expanded.count); expect((await state(page)).camera).toBe(expanded.camera);
  await group.locator('.architecture-expand').dblclick(); await ready(page);
  await expect(group.locator('.architecture-expand')).toHaveAttribute('aria-expanded', 'false');
  expect(Number((await state(page)).count)).toBe(Number(expanded.count) + 1);
  await label.focus(); await page.keyboard.press('ArrowRight'); await ready(page);
  await expect(group.locator('.architecture-expand')).toHaveAttribute('aria-expanded', 'true');
  await page.keyboard.press('ArrowLeft'); await ready(page);
  await expect(group.locator('.architecture-expand')).toHaveAttribute('aria-expanded', 'false');
});

test('nested derived double-click contracts after expansion and shared structure navigation requires a concrete instance', async ({ page }) => {
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('connections'); await ready(page);
  await findComponent(page, 'layer-3'); await ready(page);
  await card(page, 'layer-3').locator('.architecture-node-label').dblclick(); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  const derived = card(page, 'mlp:layer-3.gate');
  await derived.locator('.architecture-node-label').dblclick(); await ready(page);
  await expect(derived.locator('.architecture-expand')).toHaveAttribute('aria-expanded', 'true');
  const expanded = await state(page);
  await derived.locator('.architecture-node-label').dblclick(); await ready(page);
  await expect(derived.locator('.architecture-expand')).toHaveAttribute('aria-expanded', 'false');
  expect(Number((await state(page)).count)).toBe(Number(expanded.count) + 1);
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await derived.locator('.architecture-info').click();
  await expect(page.getByRole('dialog', { name: 'Group inspection' })).toBeVisible();
  await page.keyboard.press('Escape');
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('templates'); await ready(page);
  await openShared(page, 'shared-full-attention'); await ready(page);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  const controls = page.locator('.architecture-navigate');
  for (const control of await controls.all()) {
    await expect(control).toHaveAttribute('aria-disabled', 'true');
    await expect(control).toHaveAttribute('title', /Choose a concrete instance/);
  }
  const structure = await state(page);
  await controls.first().dblclick({ force: true }); expect(await state(page)).toEqual(structure);
  await page.getByLabel('Shared structure instance', { exact: true }).selectOption('layer-2.attention'); await ready(page);
  // Geometry still uses the anchor instance's ID, while navigation uses the bound source.
  const rebound = card(page, 'layer-0.attention.Q').locator('.architecture-navigate');
  await expect(rebound).toHaveAccessibleName('Explore component: Q projection (layer-2.attention.Q)');
  await expect(card(page, 'layer-0.attention').locator('.architecture-navigate')).toHaveAccessibleName(/^View in model:.*layer-2.attention\)/);
  await page.getByRole('button', { name: 'Fit view', exact: true }).click(); await ready(page);
  const shared = await state(page);
  await rebound.click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', 'layer-2.attention.Q');
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
  const restored = await state(page);
  expect(restored).toEqual({ ...shared, count: restored.count });
  await expect(rebound).toHaveAccessibleName('Explore component: Q projection (layer-2.attention.Q)');
  await rebound.click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-template-id', '');
  await expect(panel(page)).toHaveAttribute('data-scope-id', 'layer-2.attention.Q');
  await card(page, 'layer-2.attention.Q').locator('.architecture-navigate').click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', '');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-2.attention.Q');
  await expect(card(page, 'layer-2.attention.Q').locator('.architecture-node')).toHaveAttribute('data-selected', 'true');
});
