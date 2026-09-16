import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import { findComponent, openShared, selectComponent } from './architecture-controls';

const harness = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/architecture.html`;
const panel = (page: Page) => page.getByLabel('Architecture graph', { exact: true });
const browser = (page: Page) => page.getByRole('complementary', { name: 'Architecture browser' });
const row = (page: Page, id: string) => browser(page).locator(`[data-node-id=${JSON.stringify(id)}]`).first();
const search = (page: Page) => page.getByRole('searchbox', { name: 'Search components' });
async function ready(page: Page) { await expect(panel(page)).toHaveAttribute('aria-busy', 'false'); }
async function state(page: Page) {
  await ready(page);
  return page.evaluate(() => ({ camera: document.querySelector('.react-flow__viewport')?.getAttribute('style'),
    layout: document.querySelector('[data-layout-count]')?.getAttribute('data-layout-count'),
    source: document.querySelector('[data-source-node-ids]')?.getAttribute('data-source-node-ids'),
    scope: document.querySelector('[data-scope-id]')?.getAttribute('data-scope-id') }));
}

test('selection and search preserve the camera/layout; exact hidden selection reveals only on Center', async ({ page }) => {
  await page.goto(`${harness}?fixture=components`); await ready(page);
  const before = await state(page);
  await selectComponent(page, 'layer-3.attention');
  expect(await state(page)).toEqual(before);
  await expect(row(page, 'layer-3.attention')).toHaveAttribute('data-selected', 'true');
  await expect(page.locator('output')).toHaveText('');
  await search(page).fill('absent');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-3.attention');
  expect(await state(page)).toEqual(before);
  await page.getByRole('button', { name: 'Center selected', exact: true }).click(); await ready(page);
  await expect(search(page)).toHaveValue('absent');
  await expect(page.locator('[data-id="layer-3.attention"] .architecture-node')).toHaveAttribute('data-expanded', 'false');
  const source = JSON.parse((await panel(page).getAttribute('data-source-node-ids'))!) as string[];
  expect(source).toContain('layer-3.attention'); expect(source).not.toContain('layer-3.attention.Q');
  expect(source).not.toContain('layer-0.attention');
  await page.getByRole('button', { name: 'Clear node selection' }).click();
  await expect(search(page)).toHaveValue('absent');
  await search(page).fill('layer-3.attention.Q'); await search(page).press('Enter');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-3.attention.Q');
  const leaf = row(page, 'layer-3.attention.Q');
  await expect(leaf.locator('[data-icon="block"]')).toHaveCount(1);
  await expect(leaf.locator('.architecture-browser-disclosure')).toHaveCount(0);
  const leafState = await state(page); await leaf.locator('[data-browser-name]').dblclick();
  expect(await state(page)).toEqual(leafState);
  await page.getByRole('button', { name: 'Collapse browser' }).click();
  await page.locator('[data-id="layer-3.attention"] .architecture-node-label').click();
  await expect(page.getByRole('button', { name: 'Expand browser' })).toBeVisible();
  await page.getByRole('button', { name: 'Expand browser' }).click();
  await expect(search(page)).toHaveValue('layer-3.attention.Q');
  expect(await state(page)).toEqual(leafState);
  await expect(page.getByRole('button', { name: 'Find component', exact: true })).toHaveCount(0);
  await expect(page.getByRole('combobox', { name: 'Shared structures', exact: true })).toHaveCount(0);
  await expect(page.locator('.architecture-context-navigation[aria-label="Model components"]')).toHaveCount(0);
});

test('browser and canvas toggles share nested expansion, independent siblings and exact targets', async ({ page }) => {
  await page.goto(`${harness}?fixture=components`); await ready(page);
  await findComponent(page, 'layer-3.attention'); await ready(page);
  const attention = row(page, 'layer-3.attention');
  await expect(attention.locator('[data-icon="component"]')).toHaveCount(1);
  await attention.locator('.architecture-browser-disclosure').click(); await ready(page);
  await expect(attention.locator('.architecture-browser-disclosure')).toHaveAttribute('aria-expanded', 'true');
  await expect(page.locator('[data-id="layer-3.attention"] .architecture-node')).toHaveAttribute('data-expanded', 'true');
  await row(page, 'layer-3').locator('.architecture-browser-disclosure').click(); await ready(page);
  await expect(row(page, 'layer-3.attention')).toHaveCount(0);
  await row(page, 'layer-3').locator('.architecture-browser-disclosure').click(); await ready(page);
  await expect(attention.locator('.architecture-browser-disclosure')).toHaveAttribute('aria-expanded', 'true');
  await findComponent(page, 'layer-3.attention'); await ready(page);
  await page.locator('[data-id="layer-3.attention"] .architecture-expand').click(); await ready(page);
  await expect(attention.locator('.architecture-browser-disclosure')).toHaveAttribute('aria-expanded', 'false');
  await attention.locator('[data-browser-name]').dblclick(); await ready(page);
  await expect(page.locator('[data-id="layer-3.attention"] .architecture-node')).toHaveAttribute('data-expanded', 'true');
  await page.locator('[data-id="layer-3.attention"] .architecture-node-label').dblclick(); await ready(page);
  await expect(attention.locator('.architecture-browser-disclosure')).toHaveAttribute('aria-expanded', 'false');
  await expect(row(page, 'layer-3').locator('.architecture-browser-disclosure')).toHaveAttribute('aria-expanded', 'true');
  await attention.locator('[data-browser-name]').focus(); await page.keyboard.press('ArrowRight'); await ready(page);
  await expect(attention.locator('.architecture-browser-disclosure')).toHaveAttribute('aria-expanded', 'true');
  await page.keyboard.press('Escape'); await expect(search(page)).toBeFocused();
});

test('outside isolation selection is inert; Center and expansion return explicitly with Back restoration', async ({ page }) => {
  await page.goto(`${harness}?fixture=components`); await ready(page);
  await findComponent(page, 'layer-3.attention'); await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
  const isolated = await state(page);
  await selectComponent(page, 'layer-1.mlp'); expect(await state(page)).toEqual(isolated);
  await page.getByRole('button', { name: 'Center selected', exact: true }).click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', '');
  await expect(page.locator('[data-id="layer-1.mlp"] .architecture-node')).toHaveAttribute('data-expanded', 'false');
  await page.getByRole('button', { name: 'Back', exact: true }).click();
  const restored = await state(page); expect(restored.camera).toBe(isolated.camera); expect(restored.source).toBe(isolated.source);
  await expect(panel(page)).toHaveAttribute('data-scope-id', 'layer-3.attention');
  await row(page, 'layer-1.mlp').locator('.architecture-browser-disclosure').click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', '');
  await expect(page.locator('[data-id="layer-1.mlp"] .architecture-node')).toHaveAttribute('data-expanded', 'true');
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-1.mlp');
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-scope-id', 'layer-3.attention');
});

test('family disclosure has no graph effect, uses declared counts/order and selects concrete source identity', async ({ page }) => {
  await page.goto(`${harness}?fixture=templates`); await ready(page);
  const family = page.locator('[data-family-id="shared-full-attention"]'), before = await state(page);
  await expect(family.locator('[data-icon="family"]')).toHaveCount(1); await expect(family).toContainText('2 instances');
  await family.getByRole('button', { name: 'Show instances of Full attention' }).click();
  const familyName = family.getByRole('button', { name: 'Select shared family Full attention' });
  await familyName.focus(); await page.keyboard.press('ArrowLeft');
  await expect(family.getByRole('button', { name: 'Show instances of Full attention' })).toHaveAttribute('aria-expanded', 'false');
  await page.keyboard.press('ArrowRight'); await page.keyboard.press('Escape');
  await expect(search(page)).toBeFocused();
  expect(await state(page)).toEqual(before);
  expect(await family.locator('[data-node-id]').evaluateAll((rows) => rows.map((r) => (r as HTMLElement).dataset.nodeId))).toEqual(['layer-0.attention', 'layer-2.attention']);
  await family.locator('[data-node-id="layer-2.attention"] [data-browser-name]').click();
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-2.attention');
  expect(await state(page)).toEqual(before);
  await openShared(page, 'shared-full-attention'); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-template-instance-id', '');
  await expect(browser(page).locator('[data-node-id][data-selected="true"]')).toHaveCount(0);
  await selectComponent(page, 'layer-0.attention');
  await expect(row(page, 'layer-0.attention')).toHaveAttribute('data-selected', 'true');
  await expect(page.getByRole('button', { name: 'Inspect selected', exact: true })).toHaveCount(0);
  await expect(panel(page)).toHaveAttribute('data-template-instance-id', '');
  await search(page).fill('');
  await expect(page.getByRole('button', { name: /^Inspect matrix/ })).toHaveCount(0);
  await page.getByLabel('Shared structure instance', { exact: true }).selectOption('layer-2.attention'); await ready(page);
  await family.locator('[data-node-id="layer-2.attention"] .architecture-browser-disclosure').click(); await ready(page);
  await expect(family.locator('[data-node-id="layer-2.attention"] .architecture-browser-disclosure')).toHaveAttribute('aria-expanded', 'false');
  await selectComponent(page, 'layer-2.attention.Q');
  await page.getByRole('button', { name: 'Center selected', exact: true }).click(); await ready(page);
  await expect(panel(page)).toHaveAttribute('data-template-instance-id', 'layer-2.attention');
  expect(JSON.parse((await panel(page).getAttribute('data-source-node-ids'))!)).toContain('layer-2.attention.Q');
  await page.getByRole('combobox', { name: 'Fixture', exact: true }).selectOption('templates-absent'); await ready(page);
  await expect(browser(page)).toContainText('No verified shared structures'); await expect(page.getByLabel('Graph selection', { exact: true })).toHaveCount(0);
});

test('pane collapse/restore and bounded resize preserve mounted canvas, filter and scroll; storage failure is safe', async ({ page }) => {
  await page.addInitScript(() => { Storage.prototype.setItem = () => { throw new Error('disabled'); }; });
  await page.goto(`${harness}?fixture=components-large`); await ready(page);
  const canvas = await page.locator('.react-flow').elementHandle(), before = await state(page);
  const separator = page.getByRole('separator', { name: 'Resize architecture browser' });
  if (page.viewportSize()!.width > 760) {
    await separator.focus(); await page.keyboard.press('End'); await expect(separator).toHaveAttribute('aria-valuenow', '480');
    await page.keyboard.press('Home'); await expect(separator).toHaveAttribute('aria-valuenow', '200');
    await page.keyboard.press('ArrowRight'); await expect(separator).toHaveAttribute('aria-valuenow', '216');
    const box = (await separator.boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + 40); await page.mouse.down(); await page.mouse.move(box.x + 80, box.y + 40); await page.mouse.up();
    expect(Number(await separator.getAttribute('aria-valuenow'))).toBeGreaterThan(216);
  }
  await search(page).fill('attention');
  const scroller = page.locator('.architecture-browser-scroll'); await scroller.evaluate((node) => { node.scrollTop = 220; });
  const scroll = await scroller.evaluate((node) => node.scrollTop);
  await page.getByRole('button', { name: 'Collapse browser' }).click();
  await expect(page.getByRole('button', { name: 'Expand browser' })).toBeFocused();
  expect((await page.getByRole('navigation', { name: 'Browser navigation' }).boundingBox())!.width).toBe(40);
  await page.getByRole('button', { name: 'Expand browser' }).click();
  await expect(page.getByRole('button', { name: 'Collapse browser' })).toBeFocused();
  await expect(search(page)).toHaveValue('attention'); expect(await scroller.evaluate((node) => node.scrollTop)).toBe(scroll);
  expect(await canvas!.evaluate((element) => element === document.querySelector('.react-flow'))).toBe(true);
  expect(await state(page)).toEqual(before);
  expect(await page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.scrollHeight])).toEqual([page.viewportSize()!.width, page.viewportSize()!.height]);
});

test('captures final browser, minimal header, shared instances and nested isolation', async ({ page }, info) => {
  await page.goto(`${harness}?fixture=templates`); await ready(page);
  await page.screenshot({ path: info.outputPath('browser-expanded.png') });
  await page.getByRole('button', { name: 'Collapse browser' }).click();
  await page.screenshot({ path: info.outputPath('browser-collapsed.png') });
  await page.getByRole('button', { name: 'Expand browser' }).click();
  await selectComponent(page, 'layer-2.attention');
  await page.screenshot({ path: info.outputPath('browser-filtered-selection.png') });
  await search(page).fill('');
  await page.locator('[data-family-id]').getByRole('button', { name: 'Show instances of Full attention' }).click();
  await page.locator('[data-family-id]').scrollIntoViewIfNeeded();
  await page.screenshot({ path: info.outputPath('browser-shared-instances.png') });
  await findComponent(page, 'layer-2');
  await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
  await selectComponent(page, 'layer-2.attention');
  await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
  await search(page).fill('');
  await page.screenshot({ path: info.outputPath('browser-nested-isolation.png') });
});

 test('clearing search restores tree scroll and retains explicit filtered expansion', async ({ page }) => {
  await page.goto(`${harness}?fixture=components-large`); await ready(page);
  const scroller = page.locator('.architecture-browser-scroll');
  await scroller.evaluate((node) => { node.scrollTop = 190; });
  await expect.poll(() => scroller.evaluate((node) => node.scrollTop)).toBe(190);
  await search(page).fill('layer-30.attention');
  await row(page, 'layer-30.attention').locator('.architecture-browser-disclosure').click(); await ready(page);
  await search(page).fill('');
  await expect.poll(() => scroller.evaluate((node) => node.scrollTop)).toBe(190);
  await expect(row(page, 'layer-30.attention').locator('.architecture-browser-disclosure')).toHaveAttribute('aria-expanded', 'true');
});

test('family selection clears a prior connection inspector without changing graph detail or camera', async ({ page }) => {
  await page.goto(`${harness}?fixture=templates`); await ready(page);
  const edge = page.locator('.architecture-connection').first();
  await edge.focus(); await page.keyboard.press('Enter');
  await expect(page.getByRole('dialog', { name: 'Connection inspection', exact: true })).toBeVisible();
  const before = await state(page);
  await browser(page).getByRole('button', { name: 'Select shared family Full attention' }).click();
  await expect(page.getByRole('dialog', { name: 'Connection inspection', exact: true })).toHaveCount(0);
  expect(await state(page)).toEqual(before);
  await expect(page.getByRole('button', { name: 'Explore structure', exact: true })).toBeVisible();
  await findComponent(page, 'layer-2.attention');
  await page.getByRole('button', { name: 'Explore component', exact: true }).click(); await ready(page);
  await browser(page).getByRole('button', { name: 'Select shared family Full attention' }).click();
  await page.getByRole('button', { name: 'Back', exact: true }).click(); await ready(page);
  await expect(page.getByLabel('Graph selection', { exact: true })).toHaveAttribute('data-node-id', 'layer-2.attention');
  await expect(browser(page).getByRole('button', { name: 'Select shared family Full attention' })).toHaveAttribute('aria-pressed', 'false');
});
