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
async function compactHeader(page: Page) {
  await expect(page.locator('.matrix-panel-header')).toHaveCount(1);
  await expect(page.locator('.distribution-range')).toHaveCount(0);
  await expect(page.getByText('Full-range bins', { exact: true })).toHaveCount(0);
  const bounds = await geometry(page);
  expect(bounds.height).toBe(40);
  expect(bounds.matrixTop).toBe(bounds.headerTop + bounds.height);
  return bounds;
}

for (const constant of [false, true]) test(`header and matrix geometry stay fixed through streaming, success, cancellation, and auxiliary failure (${constant ? 'constant' : 'varied'} data)`, async ({ page }) => {
  await open(page);
  await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'loading');
  const initial = await compactHeader(page);
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
  expect(await compactHeader(page)).toEqual(initial);
  await page.getByRole('button', { name: 'Tensor information', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Tensor information' });
  await expect(dialog).toContainText('Full-range bins');
  if (constant) await expect(dialog).toContainText('Constant · samples in bin 50; no value span');
  await expect(dialog.locator('dt').filter({ hasText: /^True finite minimum$/ }).locator('+ dd')).toHaveText(constant ? '0' : '-2');
  await expect(dialog.locator('dt').filter({ hasText: /^True finite maximum$/ }).locator('+ dd')).toHaveText(constant ? '0' : '2');
  expect(await geometry(page)).toEqual(initial);
  await page.keyboard.press('Escape');
  await page.evaluate((constant) => {
    const f = window.explorerFixture; f.data(0, constant ? [0, 0, 0] : [1, -1, 0]); f.end(0);
  }, constant);
  await expect(page.locator('[data-result]')).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(0);
  expect(await compactHeader(page)).toEqual(initial);
  await expect(page.locator('.matrix-panel-header')).not.toContainText(/Complete|Ready/);
  await page.getByRole('button', { name: /^B \[/ }).click();
  await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'loading');
  const next = await geometry(page);
  await page.getByRole('button', { name: 'Cancel loading' }).click();
  await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'cancelled');
  expect(await compactHeader(page)).toEqual(next);
  await page.getByRole('button', { name: /^A \[/ }).click();
  const failed = await geometry(page);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    f.emit(6, 1, f.metadata(6)); f.data(6, [-2, 0, 2, 1, -1, 0]); f.end(6);
    f.requests[7]!.stream.close();
  });
  await expect(page.locator('[data-result="statistics"]')).toHaveAttribute('data-state', 'failed');
  expect(await compactHeader(page)).toEqual(failed);
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
  await expect(dialog.locator('dt')).toHaveText(['Logical path', 'Rank', 'Elements', 'Storage dtype', 'Logical dtype',
    'Distribution domain', 'True finite minimum', 'True finite maximum']);
  await expect(dialog).toContainText('Bin domain unavailable');
  await expect(dialog.locator('dt').filter({ hasText: /^True finite minimum$/ }).locator('+ dd')).toHaveText('Unavailable');
  await expect(dialog.locator('dt').filter({ hasText: /^True finite maximum$/ }).locator('+ dd')).toHaveText('Unavailable');
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

test('long tensor identity retains one row and reachable information, cancellation, and rightmost Fit width at 280px', async ({ page }) => {
  await page.setViewportSize({ width: 280, height: 844 });
  await open(page, 'A', true);
  const initial = await compactHeader(page);
  const header = page.locator('.matrix-panel-header');
  const info = page.getByRole('button', { name: 'Tensor information', exact: true });
  const fit = page.getByRole('button', { name: 'Fit width', exact: true });
  const cancel = page.getByRole('button', { name: 'Cancel loading' });
  const bounds = (await header.boundingBox())!;
  for (const button of [info, cancel, fit]) {
    await expect(button).toBeVisible();
    await button.click({ trial: true });
    const control = (await button.boundingBox())!;
    expect(control.x).toBeGreaterThanOrEqual(bounds.x);
    expect(control.x + control.width).toBeLessThanOrEqual(bounds.x + bounds.width);
    expect(control.y).toBeGreaterThanOrEqual(bounds.y);
    expect(control.y + control.height).toBeLessThanOrEqual(bounds.y + bounds.height);
  }
  expect((await fit.boundingBox())!.x).toBeGreaterThan((await cancel.boundingBox())!.x);
  await expect(page.locator('.tensor-identity')).toHaveAttribute('title', 'A'.repeat(200));
  expect(await page.locator('.tensor-identity').evaluate(node => node.scrollWidth > node.clientWidth)).toBe(true);
  await info.focus(); await info.press('Enter');
  const dialog = page.getByRole('dialog', { name: 'Tensor information' });
  await expect(dialog).toContainText('A'.repeat(200));
  await expect(dialog).toContainText('Bin domain unavailable');
  const popover = (await dialog.boundingBox())!;
  const pane = (await page.getByRole('region', { name: 'Tensor Explorer workspace', exact: true }).boundingBox())!;
  expect(popover.x).toBeGreaterThanOrEqual(pane.x);
  expect(popover.x + popover.width).toBeLessThanOrEqual(pane.x + pane.width);
  await page.keyboard.press('Escape');
  await expect(info).toBeFocused();
  await fit.focus(); await fit.press('Enter');
  await expect(fit).toBeFocused();
  expect(await compactHeader(page)).toEqual(initial);
  await cancel.focus(); await cancel.press('Enter');
  await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'cancelled');
  expect(await compactHeader(page)).toEqual(initial);
});

for (const input of ['keyboard', 'touch'] as const) test(`long path and constant distribution details scroll inside the narrow popover with ${input}`, async ({ browser }) => {
  const context = await browser.newContext({ hasTouch: input === 'touch', viewport: { width: 280, height: 640 } });
  const page = await context.newPage();
  await open(page, 'A', true);
  const initial = await compactHeader(page);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    const metadata = f.metadata(2);
    if (metadata.kind !== 'tensor_distributions') throw new Error('Unexpected fixture');
    f.emit(2, 1, { ...metadata, domain_minimum: Math.fround(1e20), domain_maximum: Math.fround(1e20) });
    f.data(2, [0]);
  });
  const info = page.getByRole('button', { name: 'Tensor information', exact: true });
  if (input === 'touch') await info.tap();
  else { await info.focus(); await info.press('Enter'); }
  const dialog = page.getByRole('dialog', { name: 'Tensor information' });
  const close = page.getByRole('button', { name: 'Close tensor information' });
  await expect(close).toBeFocused();
  await expect(dialog).toContainText('A'.repeat(200));
  await expect(dialog).toContainText('Constant · samples in bin 50; no value span');
  expect(await dialog.evaluate(node => node.scrollHeight > node.clientHeight)).toBe(true);
  const bounds = (await dialog.boundingBox())!;
  const pane = (await page.getByRole('region', { name: 'Tensor Explorer workspace', exact: true }).boundingBox())!;
  expect(bounds.x).toBeGreaterThanOrEqual(pane.x);
  expect(bounds.x + bounds.width).toBeLessThanOrEqual(pane.x + pane.width);
  expect(bounds.y + bounds.height).toBeLessThanOrEqual(pane.y + pane.height);
  if (input === 'keyboard') {
    await close.press('End');
    await expect.poll(() => dialog.evaluate(node => node.scrollTop)).toBeGreaterThan(0);
    await expect(close).toBeFocused();
  } else {
    const cdp = await context.newCDPSession(page);
    const x = bounds.x + bounds.width / 2, startY = bounds.y + bounds.height - 16;
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x, y: startY }] });
    for (let step = 1; step <= 5; step++) await cdp.send('Input.dispatchTouchEvent', {
      type: 'touchMove', touchPoints: [{ x, y: startY - step * (bounds.height - 48) / 5 }],
    });
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
    await expect.poll(() => dialog.evaluate(node => node.scrollTop)).toBeGreaterThan(0);
    await cdp.detach();
  }
  expect(await geometry(page)).toEqual(initial);
  expect(await page.evaluate(() => document.scrollingElement!.scrollTop)).toBe(0);
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(info).toBeFocused();
  await context.close();
});

test('tensor and unavailable auxiliary errors remain in the compact row with honest on-demand metadata', async ({ page }) => {
  await open(page);
  const initial = await compactHeader(page);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    for (const request of f.requests) request.stream.close();
  });
  for (const result of ['tensor', 'statistics', 'distributions'])
    await expect(page.locator(`[data-result="${result}"]`)).toHaveAttribute('data-state', 'failed');
  expect(await compactHeader(page)).toEqual(initial);
  await page.getByRole('button', { name: 'Tensor information', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Tensor information' });
  await expect(dialog).toContainText('Bin domain unavailable');
  await expect(dialog.locator('dt').filter({ hasText: /^True finite minimum$/ }).locator('+ dd')).toHaveText('Unavailable');
  await expect(dialog.locator('dt').filter({ hasText: /^True finite maximum$/ }).locator('+ dd')).toHaveText('Unavailable');
  expect(await geometry(page)).toEqual(initial);
});
