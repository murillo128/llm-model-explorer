import { findComponent, graphAction } from './architecture-controls';
import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import { nativeCamera } from './native-camera';

async function open(page: Page, model = 'lab/alpha', strict = false) {
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html?architecture${strict ? '&strict' : ''}`);
  await page.getByRole('combobox', { name: 'Model', exact: true }).selectOption(model);
  await expect(page.getByText('Partial tensor inventory:', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'Architecture Explorer', exact: true }).click();
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('data-visible-nodes', '3');
}
async function inspect(page: Page, node = 'linear1') {
  await findComponent(page, node);
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('aria-busy', 'false');
  const trigger = page.getByRole('button', { name: 'Inspect selected', exact: true });
  await trigger.focus(); await page.keyboard.press('Enter');
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.getByRole('dialog')).toHaveAccessibleName(node);
  await expect(page.getByRole('button', { name: 'Close inspection' })).toBeFocused();
}
const requests = (page: Page, count: number) => expect.poll(() => page.evaluate(() => window.explorerFixture.requests.length)).toBe(count);
const released = (page: Page) => expect.poll(() => page.evaluate(() => ({
  textures: window.explorerFixture.metrics.live.size, displays: window.explorerFixture.metrics.liveDisplays.size,
  liveRenderers: window.explorerFixture.renderers.filter((r) => r.state !== 'disposed').length,
  retainedScalars: window.explorerFixture.renderers.filter((r) => Reflect.get(r, 'values') !== null).length,
}))).toEqual({ textures: 0, displays: 0, liveRenderers: 0, retainedScalars: 0 });

test('concrete repeated weight preserves exact progressive values, independent profiles/statistics and native camera', async ({ page }, info) => {
  await open(page);
  await graphAction(page, 'Show all operations');
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('data-visible-nodes', '6');
  await inspect(page);
  const camera = await page.locator('.react-flow__viewport').getAttribute('style');
  const shell = await page.locator('dialog.architecture-inspection').boundingBox();
  await expect(page.getByText('Module: layers.1.linear')).toBeVisible();
  await page.getByLabel('Inspect parameter').selectOption('second');
  await requests(page, 3);
  await expect(page.locator('.matrix-panel-header')).toHaveCount(1);
  await expect(page.locator('.architecture-inspection-heading')).toHaveCount(1);
  const panel = (await page.locator('.viewer-panel').boundingBox())!;
  const title = (await page.locator('.matrix-panel-header').boundingBox())!;
  const body = (await page.locator('.viewer-panel-body').boundingBox())!;
  expect(title).toEqual({ x: panel.x, y: panel.y, width: panel.width, height: 40 });
  expect(body.y).toBe(title.y + title.height);
  expect(body.width).toBe(title.width);
  expect(await page.locator('dialog.architecture-inspection').boundingBox()).toEqual(shell);
  expect(await page.evaluate(() => window.explorerFixture.requests.map((r) => [r.tensor, r.kind]))).toEqual([['B', 'data'], ['B', 'statistics'], ['B', 'distributions']]);
  await page.evaluate(() => { const f = window.explorerFixture; f.emit(0, 1, f.metadata(0)); f.data(0, [-2, 0], 3); });
  await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'streaming');
  await nativeCamera(page);
  const prefix = await page.evaluate(() => {
    const f = window.explorerFixture, matrix = f.renderers[0]!;
    return { cells: [matrix.readCell(0, 0), matrix.readCell(0, 1), matrix.readCell(1, 0)], pixels: f.pixels(matrix), uploads: f.metrics.uploads, allocations: f.metrics.scalarAllocations };
  });
  expect(prefix.cells.map((c) => c && 'value' in c ? c.value : c?.state)).toEqual([-2, 0, 'pending']);
  expect(prefix.pixels[1]![0]).toEqual([46, 61, 76, 255]);
  await page.evaluate(() => { const f = window.explorerFixture; f.emit(1, 1, f.metadata(1)); });
  await expect(page.locator('[data-result="statistics"]')).toHaveAttribute('data-state', 'loading');
  await page.evaluate(() => window.explorerFixture.end(1));
  await expect(page.locator('[data-result="statistics"]')).toHaveCount(0);
  expect(await page.evaluate(() => window.explorerFixture.metrics.uploads)).toBe(prefix.uploads);
  expect(prefix.allocations).toBe(1);
  await page.evaluate(() => {
    const f = window.explorerFixture; f.data(0, [2, 1, -1, 0]); f.end(0);
    const rows = Array<number>(300).fill(0);
    for (const index of [0, 50, 199, 175, 225, 250]) rows[index] = 1;
    f.emit(2, 1, f.metadata(2)); f.data(2, [...rows, 1], 31);
  });
  await expect(page.locator('[data-result="tensor"]')).toHaveCount(0);
  await expect(page.locator('[data-result="distributions"]')).toHaveAttribute('data-state', 'streaming');
  expect(await page.evaluate(() => {
    const f = window.explorerFixture; return [f.renderers[1]!.readCell(2, 25), f.renderers[2]!.readCell(0, 0), f.renderers[2]!.readCell(0, 1)];
  })).toEqual([expect.objectContaining({ value: 1 }), expect.objectContaining({ value: 1 }), expect.objectContaining({ state: 'pending' })]);
  await page.evaluate(() => {
    const f = window.explorerFixture, columns = Array<number>(200).fill(0);
    for (const index of [0, 198, 50, 151]) columns[index] = 1;
    columns[101] = 2; f.data(2, columns.slice(1)); f.end(2);
  });
  await expect(page.locator('[data-result="distributions"]')).toHaveCount(0);
  await page.locator('.matrix-scroll').focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('dialog').getByLabel('Cell inspection')).toBeVisible();
  await expect(page.locator('.inspection-readout')).toContainText('row 0 · column 1');
  await expect(page.locator('.inspection-readout span').last()).toHaveText('0');
  const extent = page.locator('.matrix-scroll > div').first();
  expect(await extent.evaluate((e) => e.getBoundingClientRect().width)).toBe(2);
  await page.locator('.matrix-scroll').evaluate((host) => {
    const box = host.getBoundingClientRect(); host.dispatchEvent(new WheelEvent('wheel', { deltaY: -300, clientX: box.left + 1, clientY: box.top + 1, cancelable: true }));
  });
  expect(await extent.evaluate((e) => e.getBoundingClientRect().width)).toBeGreaterThan(2);
  expect(await page.evaluate(() => window.explorerFixture.metrics.scalarAllocations)).toBe(1);
  expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(camera);
  expect(await page.locator('dialog.architecture-inspection').boundingBox()).toEqual(shell);
  await page.screenshot({ path: info.outputPath('architecture-native-weight.png') });
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Inspect selected', exact: true })).toBeFocused();
  expect(await page.locator('.react-flow__viewport').getAttribute('style')).toBe(camera);
  await expect(page.getByLabel('Architecture graph', { exact: true })).toHaveAttribute('data-visible-nodes', '6');
  await released(page);
});

test('close, replace, and reopen fence callbacks and release all numeric resources', async ({ page }) => {
  await open(page); await inspect(page);
  await page.getByLabel('Inspect parameter').selectOption('second'); await requests(page, 3);
  await page.getByLabel('Inspect parameter').selectOption('vector-weight'); await requests(page, 5);
  await expect.poll(() => page.evaluate(() => window.explorerFixture.cancelled.length)).toBe(3);
  expect(await page.evaluate(() => window.explorerFixture.renderers.slice(0, 3).map((r) => r.state))).toEqual(['disposed', 'disposed', 'disposed']);
  await page.getByRole('button', { name: 'Close inspection' }).click(); await released(page);
  for (let i = 0; i < 3; i++) {
    await inspect(page, 'linear0');
    await page.getByLabel('Inspect parameter').selectOption('first'); await requests(page, 8 + i * 3);
    const uploads = await page.evaluate(() => window.explorerFixture.metrics.uploads);
    await page.evaluate(() => {
      const f = window.explorerFixture;
      for (const callback of f.callbacks.slice(0, 5)) callback.onData?.(new Uint8Array(new Float32Array([99]).buffer), 0);
    });
    expect(await page.evaluate(() => window.explorerFixture.metrics.uploads)).toBe(uploads);
    // Native modal contains keyboard focus, including backwards traversal.
    await page.getByRole('button', { name: 'Close inspection' }).focus();
    for (let j = 0; j < 8; j++) {
      await page.keyboard.press('Shift+Tab');
      expect(await page.evaluate(() => !!document.activeElement?.closest('dialog'))).toBe(true);
    }
    await page.keyboard.press('Escape'); await released(page);
  }
  await expect.poll(() => page.evaluate(() => window.explorerFixture.cancelled.length)).toBe(14);
});

test('every unavailable binding and V-JEPA patch shape remain metadata without packed-value requests', async ({ page }) => {
  await open(page, 'lab/beta'); await inspect(page);
  for (const [id, reason] of [['quantized', 'unsupported representation'], ['fused', 'requires view'], ['unresolved', 'unresolved binding'], ['volume', 'unsupported rank']]) {
    await page.getByLabel('Inspect parameter').selectOption(id!);
    await expect(page.getByRole('dialog').getByRole('status')).toContainText(reason!);
    await expect(page.locator('.matrix-scroll')).toHaveCount(0);
  }
  await expect(page.getByText(/encoder.patch_embed.proj.weight · native · logical shape/)).toContainText('1024 × 3 × 2 × 16 × 16');
  await requests(page, 0);
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'Tensor Explorer', exact: true }).click();
  await expect(page.getByText('Packed weights are metadata only.')).toBeVisible();
  await expect(page.getByRole('button', { name: /^unsupported \[/ })).toBeVisible();
});

test('renderer allocation failure remains local and retrying a fresh modal works', async ({ page }) => {
  await open(page); await inspect(page);
  await page.evaluate(() => { window.explorerFixture.metrics.failAllocation = true; });
  await page.getByLabel('Inspect parameter').selectOption('second');
  await expect(page.getByRole('alert')).toContainText('Exact rendering is unavailable');
  await requests(page, 0);
  await page.keyboard.press('Escape'); await released(page);
  await page.evaluate(() => { window.explorerFixture.metrics.failAllocation = false; });
  await inspect(page); await page.getByLabel('Inspect parameter').selectOption('second'); await requests(page, 3);
  await page.keyboard.press('Escape'); await released(page);
});

test('context loss during streaming leaves a closable graph-local error and releases resources', async ({ page }) => {
  await open(page); await inspect(page);
  await page.getByLabel('Inspect parameter').selectOption('second'); await requests(page, 3);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    f.emit(0, 1, f.metadata(0)); f.data(0, [-2, 0]);
  });
  await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'streaming');
  await page.locator('.matrix-scroll canvas').evaluate((canvas) => {
    (canvas as HTMLCanvasElement).getContext('webgl2')!.getExtension('WEBGL_lose_context')!.loseContext();
  });
  await expect(page.getByRole('alert')).toContainText('Exact rendering is unavailable');
  await page.keyboard.press('Escape'); await released(page);
  await expect.poll(() => page.evaluate(() => window.explorerFixture.cancelled.length)).toBe(3);
  await expect(page.getByLabel('Architecture graph', { exact: true })).toBeVisible();
});

test('model replacement during loading removes the modal and cancels its old session handles', async ({ page }) => {
  await open(page); await inspect(page);
  await page.getByLabel('Inspect parameter').selectOption('second'); await requests(page, 3);
  // Simulate external session/model replacement while the modal makes shell input inert.
  await page.getByRole('combobox', { name: 'Model', exact: true, includeHidden: true }).evaluate((element) => {
    const select = element as HTMLSelectElement;
    select.value = 'lab/beta'; select.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => window.explorerFixture.cancelled.length)).toBe(3);
  await released(page);
  await inspect(page); await page.getByLabel('Inspect parameter').selectOption('second'); await requests(page, 6);
  const uploads = await page.evaluate(() => window.explorerFixture.metrics.uploads);
  await page.evaluate(() => { for (const callback of window.explorerFixture.callbacks.slice(0, 3)) callback.onData?.(new Uint8Array(4), 0); });
  expect(await page.evaluate(() => window.explorerFixture.metrics.uploads)).toBe(uploads);
  await page.keyboard.press('Escape'); await released(page);
});

test('development StrictMode streams after replay, replaces weights, and releases every generation', async ({ page }) => {
  await open(page, 'lab/alpha', true);
  for (const parameter of ['second', 'vector-weight', 'second']) {
    if (!(await page.getByRole('dialog').count())) await inspect(page);
    await page.getByLabel('Inspect parameter').selectOption(parameter);
    const value = parameter === 'second' ? 2 : -2;
    // StrictMode can start and cancel replayed subscriptions. Deliver only to the
    // latest tensor operation and inspect the live renderer, never assume indices.
    await expect.poll(() => page.evaluate(() => window.explorerFixture.renderers.filter((r) => r.state === 'ready').length)).toBe(parameter === 'second' ? 3 : 1);
    await page.evaluate((value) => {
      const f = window.explorerFixture;
      const index = f.requests.findLastIndex((request) => request.kind === 'data');
      f.emit(index, 1, f.metadata(index)); f.data(index, [value], 3);
    }, value);
    await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'streaming');
    expect(await page.evaluate(() => window.explorerFixture.renderers.findLast((r) => r.state === 'ready' && r.geometry.columns !== 100 && r.geometry.rows !== 100)!.readCell(0, 0))).toMatchObject({ value });
    if (parameter === 'second') {
      await page.locator('.matrix-scroll').focus();
      await expect(page.locator('.inspection-readout span').last()).toHaveText(String(value));
    }
  }
  await page.keyboard.press('Escape'); await released(page);
  await expect(page.getByRole('button', { name: 'Inspect selected', exact: true })).toBeFocused();
  await inspect(page); await page.getByLabel('Inspect parameter').selectOption('second');
  await expect.poll(() => page.evaluate(() => window.explorerFixture.renderers.filter((r) => r.state === 'ready').length)).toBe(3);
  const uploads = await page.evaluate(() => window.explorerFixture.metrics.uploads);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    for (const callback of f.callbacks.slice(0, -3)) callback.onData?.(new Uint8Array(new Float32Array([99]).buffer), 0);
  });
  expect(await page.evaluate(() => window.explorerFixture.metrics.uploads)).toBe(uploads);
  await page.keyboard.press('Escape'); await released(page);
});
