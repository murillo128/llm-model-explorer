import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

async function open(page: Page, name = 'A') {
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html`);
  await page.getByRole('combobox').selectOption('lab/alpha');
  await page.getByRole('button', { name: new RegExp(`^${name} \\[` ) }).click();
  await expect.poll(() => page.evaluate(() => window.explorerFixture.requests.length)).toBe(3);
}
const geometry = (page: Page) => page.evaluate(() => {
  const header = document.querySelector('.matrix-panel-header')!.getBoundingClientRect();
  const matrix = document.querySelector('.matrix-surfaces')!.getBoundingClientRect();
  return { headerTop: header.top, height: header.height, matrixTop: matrix.top, matrixWidth: matrix.width };
});

test('header and matrix geometry stay fixed through streaming, success, cancellation, and auxiliary failure', async ({ page }) => {
  await open(page);
  const initial = await geometry(page);
  expect(initial.height).toBe(40);
  await page.evaluate(() => { const f = window.explorerFixture; f.emit(0, 1, f.metadata(0)); f.data(0, [-2, 0, 2]); });
  await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'streaming');
  expect(await geometry(page)).toEqual(initial);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    f.data(0, [1, -1, 0]); f.end(0);
    f.emit(1, 1, f.metadata(1)); f.end(1);
    f.emit(2, 1, f.metadata(2));
    const counts = Array<number>(500).fill(0);
    for (const index of [0, 50, 99, 175, 125, 150, 200, 351, 499, 425, 276, 352]) counts[index] = 1;
    f.data(2, counts); f.end(2);
  });
  await expect(page.locator('[data-result]')).toHaveCount(0);
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
