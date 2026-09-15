import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import { nativeCamera } from './native-camera';
import type {} from './matrix-explorer-harness';

const url = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/matrix-explorer.html`;
async function open(page: Page, strict = false) {
  await page.goto(url + (strict ? '?strict' : ''));
  await expect(page.getByRole('region', { name: 'Synthetic result matrix' })).toBeVisible();
  // Card chrome is outside the numerical fixture budget, including at narrow widths.
  expect(await page.locator('.matrix-surfaces').boundingBox()).toEqual({
    x: 16, y: 56, width: Math.min(600, page.viewportSize()!.width - 32), height: 220,
  });
  await nativeCamera(page);
}
async function hover(page: Page, row: number, column: number) {
  await nativeCamera(page);
  const point = await page.evaluate(({ row, column }) => {
    const r = window.matrixFixture.renderers.filter((r) => r.state === 'ready')[0]!;
    const rect = r.canvas.getBoundingClientRect();
    return { x: rect.left + (column - r.view!.x + 0.5) / devicePixelRatio,
      y: rect.top + (row - r.view!.y + 0.5) / devicePixelRatio };
  }, { row, column });
  await page.mouse.move(point.x, point.y);
  await expect(page.locator('.inspection-readout')).toContainText(`row ${row} · column ${column}`);
}
for (const dpr of [1, 2]) test.describe(`standalone matrix at DPR ${dpr}`, () => {
  test.use({ deviceScaleFactor: dpr });
  test('progressive exact values, native orientation and linked callbacks without auxiliary artifacts', async ({ page }) => {
    await open(page);
    const start = await page.evaluate(() => {
      const f = window.matrixFixture;
      f.subscriptions[0]!.values(new Float32Array([-0, .1, NaN, Infinity, -Infinity]), 0);
      const r = f.renderers[0]!;
      return { geometry: r.geometry, css: r.canvas.getBoundingClientRect().toJSON(), cpu: r.diagnostics.cpuBytes,
        scalar: f.metrics.scalarAllocations, integer: f.metrics.integerAllocations, uploads: f.metrics.uploads };
    });
    expect(start.geometry).toEqual({ rows: 3, columns: 5, count: 15 });
    expect(start.css.width * dpr).toBe(5); expect(start.css.height * dpr).toBe(3);
    expect(start.cpu).toBe(60); expect(start.scalar).toBe(1); expect(start.integer).toBe(0);
    await expect(page.locator('.matrix-surfaces canvas')).toHaveCount(1);
    await hover(page, 0, 0); await expect(page.locator('.inspection-readout')).toContainText('-0');
    await hover(page, 0, 1); await expect(page.locator('.inspection-readout')).toContainText(String(Math.fround(.1)));
    await hover(page, 1, 2); await expect(page.locator('.inspection-readout')).toContainText('Unavailable — not received');
    await page.evaluate(() => window.matrixFixture.subscriptions[0]!.values(new Float32Array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]), 5));
    await expect(page.locator('.inspection-readout')).toContainText('7');
    await expect(page.getByRole('status', { name: 'Linked annotation' })).toHaveText('Row 1, cell 1:2');
    const before = await page.evaluate(() => window.matrixFixture.metrics.uploads);
    await page.evaluate(() => window.matrixFixture.subscriptions[0]!.transfer({ anchors: [0, 14], slope: 8 }));
    await hover(page, 2, 4);
    expect(await page.evaluate(() => ({ uploads: window.matrixFixture.metrics.uploads,
      scalar: window.matrixFixture.metrics.scalarAllocations, subscriptions: window.matrixFixture.subscriptions.length,
      cell: window.matrixFixture.cells.at(-1), row: window.matrixFixture.rows.at(-1), column: window.matrixFixture.columns.at(-1),
      cpu: window.matrixFixture.resources().cpuBytes }))).toEqual({ uploads: before, scalar: 1, subscriptions: 1,
      cell: { row: 2, column: 4 }, row: 2, column: 4, cpu: 60 });
    await page.mouse.move(350, 300);
    await expect(page.locator('.matrix-inspection')).toHaveCount(0);
    await expect(page.getByRole('status', { name: 'Linked annotation' })).toHaveText('No selection');
    await page.locator('.matrix-scroll').focus();
    await page.keyboard.press('ArrowDown'); await page.keyboard.press('ArrowRight');
    await expect(page.getByRole('status', { name: 'Linked annotation' })).toHaveText('Row 1, cell 1:1');
    await page.keyboard.press('Escape');
    expect(await page.evaluate(() => window.matrixFixture.renderers[0]!.view!.scaleX)).toBeGreaterThan(1);
    expect(await page.evaluate(() => [window.matrixFixture.cells.at(-1), window.matrixFixture.rows.at(-1), window.matrixFixture.columns.at(-1)]))
      .toEqual([{ row: 1, column: 1 }, 1, 1]);
    await page.locator('.matrix-scroll').evaluate((host) => (host as HTMLElement).blur());
    expect(await page.evaluate(() => [window.matrixFixture.cells.at(-1), window.matrixFixture.rows.at(-1), window.matrixFixture.columns.at(-1)])).toEqual([null, null, null]);
  });
  test('matrix-only scrolling keeps exact native indices', async ({ page }) => {
    await open(page);
    await page.evaluate(() => window.matrixFixture.render('large'));
    await expect.poll(() => page.evaluate(() => window.matrixFixture.subscriptions.length)).toBe(2);
    await page.evaluate(() => window.matrixFixture.subscriptions[1]!.values(Float32Array.from({ length: 576 * 1536 }, (_, i) => i), 0));
    await page.locator('.matrix-scroll').evaluate((host) => { host.scrollLeft = 100; host.scrollTop = 20; });
    await expect.poll(() => page.evaluate(() => window.matrixFixture.renderers.at(-1)!.view!.x)).toBe(100 * dpr);
    await hover(page, 20 * dpr + 2, 100 * dpr + 3);
    await expect(page.locator('.inspection-readout')).toContainText(String((20 * dpr + 2) * 1536 + 100 * dpr + 3));
    await expect(page.locator('.row-distributions, .column-distributions')).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  });
});

for (const strict of [false, true]) test(`A → B → A fences old updates and releases resources, StrictMode=${strict}`, async ({ page }) => {
  await open(page, strict);
  const count = strict ? 2 : 1;
  await expect.poll(() => page.evaluate(() => window.matrixFixture.subscriptions.length)).toBe(count);
  await page.evaluate(() => window.matrixFixture.subscriptions.at(-1)!.values(new Float32Array([1, 2, 3, 4, 5]), 0));
  await hover(page, 0, 1);
  for (const name of ['B', 'A'] as const) {
    const previous = await page.evaluate(() => window.matrixFixture.subscriptions.length);
    await page.evaluate((name) => window.matrixFixture.render(name), name);
    await expect.poll(() => page.evaluate(() => window.matrixFixture.subscriptions.length)).toBe(previous + 1);
    await expect(page.locator('.matrix-inspection')).toHaveCount(0);
    const result = await page.evaluate(() => {
      const f = window.matrixFixture;
      const uploads = f.metrics.uploads;
      for (const old of f.subscriptions.slice(0, -1)) {
        old.values(new Float32Array([99]), 0);
        old.transfer({ anchors: [2, -2] }); // Invalid too: stale callbacks must be no-ops.
        old.distribution('rows', new Uint32Array([99]), 0);
      }
      const current = f.renderers.at(-1)!;
      const prefix = current.populatedPrefix;
      f.subscriptions.at(-1)!.values(new Float32Array([42]), 0);
      return { staleUploads: f.metrics.uploads - uploads - 1, prefix, cell: current.readCell(0, 0), resources: f.resources() };
    });
    expect(result.staleUploads).toBe(0); expect(result.prefix).toBe(0);
    expect(result.cell).toMatchObject({ value: 42 }); expect(result.resources.textures).toBe(1); expect(result.resources.cpuBytes).toBe(60);
  }
  for (let i = 0; i < 3; i++) {
    await page.evaluate(() => window.matrixFixture.render(null));
    await expect(page.locator('.matrix-explorer')).toHaveCount(0);
    expect(await page.evaluate(() => window.matrixFixture.resources())).toEqual({ textures: 0, displays: 0, framebuffers: 0, buffers: 0, programs: 0, cpuBytes: 0 });
    if (i < 2) {
      await page.evaluate(() => window.matrixFixture.render('A'));
      await expect(page.locator('.matrix-explorer')).toHaveCount(1);
      await hover(page, 0, 0);
    }
  }
});

test('full and matrix-only compositions share scalar pixels and optional authoritative profiles', async ({ page }) => {
  await open(page);
  const pixels = () => page.evaluate(() => {
    const f = window.matrixFixture;
    const updates = f.subscriptions.at(-1)!;
    updates.values(Float32Array.from({ length: 15 }, (_, i) => i - 7), 0);
    updates.transfer({ anchors: [-7, 7] });
    const r = f.renderers.findLast((r) => r.geometry.columns === 5 && r.geometry.rows === 3)!;
    r.draw();
    const gl = r.canvas.getContext('webgl2')!;
    const result = new Uint8Array(15 * 4);
    gl.readPixels(0, 0, 5, 3, gl.RGBA, gl.UNSIGNED_BYTE, result);
    return Array.from(result);
  });
  const matrixOnly = await pixels();
  await page.evaluate(() => window.matrixFixture.render('full'));
  await expect(page.locator('.matrix-surfaces canvas')).toHaveCount(3);
  await nativeCamera(page);
  const full = await pixels();
  expect(full).toEqual(matrixOnly);
  await page.evaluate(() => {
    const updates = window.matrixFixture.subscriptions.at(-1)!;
    updates.distribution('rows', new Uint32Array([5]), 0);
    updates.distribution('columns', new Uint32Array([3]), 0);
  });
  const profiles = await page.evaluate(() => window.matrixFixture.renderers.filter((r) => r.state === 'ready' && r.geometry.count !== 15)
    .map((r) => ({ shape: [r.geometry.rows, r.geometry.columns], first: r.readCell(0, 0), pending: r.readCell(0, 1) })));
  expect(profiles).toMatchObject([
    { shape: [3, 100], first: { value: 5 }, pending: { state: 'pending' } },
    { shape: [100, 5], first: { value: 3 }, pending: { state: 'pending' } },
  ]);
  await hover(page, 0, 0);
  await page.evaluate(() => window.matrixFixture.render(null));
  await expect(page.locator('.matrix-explorer')).toHaveCount(0);
  expect(await page.evaluate(() => window.matrixFixture.resources())).toEqual({ textures: 0, displays: 0, framebuffers: 0, buffers: 0, programs: 0, cpuBytes: 0 });
});
