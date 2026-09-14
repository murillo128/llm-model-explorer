import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

async function open(page: Page, name = 'A', longPath = false) {
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html`);
  if (longPath) await page.evaluate(() => { window.explorerFixture.tensors.find((tensor) => tensor.id === 'A')!.path = ['A'.repeat(200)]; });
  await page.getByRole('combobox').selectOption('lab/alpha');
  await page.getByRole('button', { name: new RegExp(`^${longPath ? 'A'.repeat(200) : name} \\[` ) }).click();
  await expect.poll(() => page.evaluate(() => window.explorerFixture.requests.length)).toBe(3);
}
const geometry = (page: Page) => page.evaluate(() => {
  const header = document.querySelector('.matrix-panel-header')!.getBoundingClientRect();
  const matrix = document.querySelector('.matrix-surfaces')!.getBoundingClientRect();
  return { headerTop: header.top, height: header.height, matrixTop: matrix.top, matrixWidth: matrix.width };
});

for (const constant of [false, true]) test(`header and matrix geometry stay fixed through streaming, success, cancellation, and auxiliary failure (${constant ? 'constant' : 'varied'} data)`, async ({ page }) => {
  await open(page);
  const initial = await geometry(page);
  expect(initial.height).toBe(40);
  const range = page.getByRole('region', { name: 'Distribution range' });
  expect((await range.boundingBox())!.height).toBe(32);
  await page.evaluate((constant) => {
    const f = window.explorerFixture;
    f.emit(0, 1, f.metadata(0)); f.data(0, constant ? [0, 0, 0] : [-2, 0, 2]);
  }, constant);
  await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'streaming');
  expect(await geometry(page)).toEqual(initial);
  await page.evaluate((constant) => {
    const f = window.explorerFixture;
    const statistics = f.metadata(1);
    const distribution = f.metadata(2);
    if (statistics.kind !== 'tensor_statistics' || distribution.kind !== 'tensor_distributions') throw new Error('Unexpected fixture');
    f.emit(1, 1, constant ? { ...statistics, minimum: 0, maximum: 0, mean: 0, stddev: 0,
      percentiles: { p01: 0, p05: 0, p50: 0, p95: 0, p99: 0 } } : statistics); f.end(1);
    f.emit(2, 1, constant ? { ...distribution, domain_minimum: 0, domain_maximum: 0 } : distribution);
    const counts = Array<number>(500).fill(0);
    if (constant) {
      counts[50] = counts[150] = 3;
      counts[350] = counts[351] = counts[352] = 2;
    } else {
      for (const index of [0, 50, 99, 175, 125, 150, 200, 351, 499, 425, 276, 352]) counts[index] = 1;
    }
    f.data(2, counts); f.end(2);
  }, constant);
  await expect(page.locator('[data-result="distributions"]')).toHaveCount(0);
  if (constant) await expect(range).toContainText('Constant · samples in bin 50; no value span');
  expect((await range.boundingBox())!.height).toBe(32);
  expect(await geometry(page)).toEqual(initial);
  await page.evaluate((constant) => {
    const f = window.explorerFixture; f.data(0, constant ? [0, 0, 0] : [1, -1, 0]); f.end(0);
  }, constant);
  await expect(page.locator('[data-result]')).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
  expect(await geometry(page)).toEqual(initial);
  await expect(page.locator('.matrix-panel-header')).not.toContainText(/Complete|Ready/);
  await page.getByRole('button', { name: /^B \[/ }).click();
  await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'loading');
  const next = await geometry(page);
  await page.getByRole('button', { name: 'Cancel loading' }).click();
  await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'cancelled');
  expect(await geometry(page)).toEqual(next);
  await page.getByRole('button', { name: /^A \[/ }).click();
  const failed = await geometry(page);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    f.emit(6, 1, f.metadata(6)); f.data(6, [-2, 0, 2, 1, -1, 0]); f.end(6);
    f.requests[7]!.stream.close();
  });
  await expect(page.locator('[data-result="statistics"]')).toHaveAttribute('data-state', 'failed');
  expect(await geometry(page)).toEqual(failed);
  expect(await page.evaluate(() => window.explorerFixture.renderers.at(-3)!.populatedPrefix)).toBe(6);
});

test('metadata previews, pins, anchors below the icon, and leaves scientific width intact', async ({ page }) => {
  await open(page, 'reference');
  const trigger = page.getByRole('button', { name: 'Tensor information', exact: true });
  const dialog = page.getByRole('dialog', { name: 'Tensor information' });
  const close = page.getByRole('button', { name: 'Close tensor information' });
  const initial = await geometry(page);
  await trigger.hover();
  await expect(dialog).toBeVisible();
  await expect(close).toHaveCount(0);
  expect((await dialog.boundingBox())!.y).toBe((await trigger.boundingBox())!.y + (await trigger.boundingBox())!.height);
  await page.mouse.move(0, 0);
  await expect(dialog).toHaveCount(0);
  await trigger.focus();
  await page.keyboard.press('Shift+Tab');
  await page.keyboard.press('Tab');
  await expect(dialog).toBeVisible();
  await expect(close).toHaveCount(0);
  await expect(trigger).toBeFocused();
  expect(await trigger.evaluate((node) => getComputedStyle(node).outlineStyle)).not.toBe('none');
  await page.keyboard.press('Enter');
  await expect(close).toBeFocused();
  await page.mouse.move(0, 0);
  await expect(dialog).toBeVisible();
  await expect(dialog.locator('dt')).toHaveText(['Logical path', 'Rank', 'Elements', 'Storage dtype', 'Logical dtype']);
  await expect(dialog).not.toContainText(/One value per device pixel|Focus the matrix/);
  expect(await geometry(page)).toEqual(initial);
  await page.locator('.matrix-scroll').evaluate((node) => { node.scrollTop = 200; node.scrollLeft = 200; });
  expect(await geometry(page)).toEqual(initial);
  await close.click();
  await expect(dialog).toHaveCount(0);
  await expect(trigger).toBeFocused();
  await page.keyboard.press('Space');
  await expect(close).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(dialog).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(trigger).toBeFocused();
  await trigger.click();
  await page.getByRole('heading', { name: 'Tensors', exact: true }).click();
  await expect(dialog).toHaveCount(0);
});

test('touch pins metadata and supports close and outside tap without hover', async ({ browser }, testInfo) => {
  const context = await browser.newContext({ hasTouch: true, viewport: testInfo.project.use.viewport ?? { width: 390, height: 844 } });
  const page = await context.newPage();
  await open(page);
  const trigger = page.getByRole('button', { name: 'Tensor information', exact: true });
  const dialog = page.getByRole('dialog', { name: 'Tensor information' });
  await trigger.tap();
  await expect(dialog).toBeVisible();
  await page.getByRole('button', { name: 'Close tensor information' }).tap();
  await expect(dialog).toHaveCount(0);
  await trigger.tap();
  await expect(dialog).toBeVisible();
  await page.getByRole('heading', { name: 'Tensors', exact: true }).tap();
  await expect(dialog).toHaveCount(0);
  await context.close();
});


test('pinned metadata follows its icon when operation status changes available identity width', async ({ page }) => {
  await open(page, 'A', true);
  const trigger = page.getByRole('button', { name: 'Tensor information', exact: true });
  const dialog = page.getByRole('dialog', { name: 'Tensor information' });
  await trigger.click();
  // Simulate producer cancellation without an outside pointer dismissal.
  await page.evaluate(() => { const f = window.explorerFixture; for (let i = 0; i < 3; i++) f.end(i, 6); });
  await expect(page.getByRole('button', { name: 'Cancel loading' })).toHaveCount(0);
  await expect.poll(() => dialog.evaluate((node) => {
    const rect = node.getBoundingClientRect();
    const icon = document.querySelector('.matrix-info-trigger')!.getBoundingClientRect();
    const pane = node.closest('.working-surface')!.getBoundingClientRect();
    return rect.top === icon.bottom && Math.abs(rect.left - Math.max(pane.left, Math.min(icon.left, pane.right - rect.width))) < 1;
  })).toBe(true);
  await expect(dialog).toBeVisible();
});

test('range metadata remains keyboard-scrollable in a very narrow scientific pane', async ({ page }) => {
  await page.setViewportSize({ width: 280, height: 844 });
  await open(page);
  const initial = await geometry(page);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    const metadata = f.metadata(2);
    if (metadata.kind !== 'tensor_distributions') throw new Error('Unexpected fixture');
    f.emit(2, 1, { ...metadata, domain_minimum: Math.fround(1e20), domain_maximum: Math.fround(1e20) });
    f.data(2, [0]);
  });
  const range = page.getByRole('region', { name: 'Distribution range' });
  await expect(range).toContainText('Constant · samples in bin 50; no value span');
  expect(await geometry(page)).toEqual(initial);
  expect(await range.evaluate(node => node.scrollHeight > node.clientHeight)).toBe(true);
  await range.focus();
  await range.press('End');
  await expect.poll(() => range.evaluate(node => node.scrollTop)).toBeGreaterThan(0);
  await expect(range).toBeFocused();
  await expect.poll(() => range.evaluate(node => {
    const description = node.lastElementChild!.getBoundingClientRect();
    const bounds = node.getBoundingClientRect();
    return description.top >= bounds.top && description.bottom <= bounds.bottom;
  })).toBe(true);
  expect(await geometry(page)).toEqual(initial);
});
