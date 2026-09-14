import { color, green } from './scalar-oracle';
import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

async function open(page: Page, name = 'A') {
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html`);
  await expect(page.getByRole('combobox')).toBeEnabled();
  await page.getByRole('combobox').selectOption('lab/alpha');
  await page.getByRole('button', { name: new RegExp(`^${name} \\[` ) }).click();
}
const status = (page: Page, result: string) => page.locator(`[data-result="${result}"]`);

for (const dpr of [1, 2]) test(`asymmetric progressive surfaces, independent results and immutable scalar allocation at DPR ${dpr}`, async ({ browser }, testInfo) => {
  const context = await browser.newContext({ viewport: testInfo.project.name === 'narrow' ? { width: 390, height: 844 } : { width: 1440, height: 900 }, deviceScaleFactor: dpr });
  const page = await context.newPage();
  await open(page);
  await expect.poll(() => page.evaluate(() => window.explorerFixture.requests.length)).toBe(3);
  const initial = await page.evaluate(() => {
    const f = window.explorerFixture;
    f.emit(0, 1, f.metadata(0));
    f.data(0, [-2, 0, 2], 3); // split actual network chunks inside words
    return f.metrics.scalarAllocations;
  });
  await expect(status(page, 'tensor')).toHaveAttribute('data-state', 'streaming');
  const partial = await page.evaluate(() => {
    const f = window.explorerFixture;
    const matrix = f.renderers.find((r) => r.geometry.columns === 3 && r.geometry.rows === 2)!;
    return { cells: [matrix.readCell(0, 0), matrix.readCell(0, 1), matrix.readCell(0, 2), matrix.readCell(1, 0)], pixels: f.pixels(matrix), uploads: f.metrics.uploads };
  });
  expect(partial.cells.map((cell) => cell && 'value' in cell ? cell.value : cell?.state)).toEqual([-2, 0, 2, 'pending']);
  expect(partial.pixels[1]![0]).toEqual([46, 61, 76, 255]);
  await expect(status(page, 'statistics')).toHaveAttribute('data-state', 'loading');
  await page.evaluate(() => { const f = window.explorerFixture; f.emit(1, 1, f.metadata(1)); });
  // Valid metadata alone does not claim a complete statistics result.
  await expect(status(page, 'statistics')).toHaveAttribute('data-state', 'loading');
  await page.evaluate(() => window.explorerFixture.end(1));
  await expect(status(page, 'statistics')).toHaveCount(0);
  expect(await page.evaluate(() => window.explorerFixture.metrics.uploads)).toBe(partial.uploads);
  expect(await page.evaluate(() => window.explorerFixture.metrics.scalarAllocations)).toBe(initial);
  expect(initial).toBe(1);
  expect(await page.evaluate(() => window.explorerFixture.metrics.integerAllocations)).toBe(2);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    f.data(0, [1, -1, 0]); f.end(0);
    f.emit(2, 1, f.metadata(2));
    const rows = Array<number>(200).fill(0);
    for (const index of [0, 50, 99, 175, 125, 150]) rows[index] = 1;
    const columns = Array<number>(300).fill(0);
    for (const index of [0, 151, 299, 225, 76, 152]) columns[index] = 1;
    // The callback crosses the metadata section offset, then leaves a pending suffix.
    f.data(2, [...rows, ...columns.slice(0, 153)], 31);
  });
  await expect(status(page, 'tensor')).toHaveCount(0);
  await expect(status(page, 'distributions')).toHaveAttribute('data-state', 'streaming');
  const distribution = await page.evaluate(() => {
    const f = window.explorerFixture;
    const row = f.renderers.find((r) => r.geometry.columns === 100)!;
    const column = f.renderers.find((r) => r.geometry.rows === 100)!;
    return { row: [row.readCell(0, 0), row.readCell(1, 25), row.readCell(1, 75)], column: [column.readCell(0, 0), column.readCell(25, 1), column.readCell(50, 2), column.readCell(75, 0)], rowPixel: f.pixels(row)[0]![0], columnPixel: f.pixels(column)[25]![1] };
  });
  expect(distribution.row.map((cell) => cell && 'value' in cell ? cell.value : null)).toEqual([1, 1, 1]);
  expect(distribution.column.map((cell) => cell && 'value' in cell ? cell.value : cell?.state)).toEqual([1, 1, 1, 'pending']);
  expect(distribution.rowPixel).toEqual([...color(Math.log1p(1) / Math.log1p(3)), 255]);
  expect(distribution.columnPixel).toEqual([...color(Math.log1p(1) / Math.log1p(2)), 255]);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    const tail = Array<number>(147).fill(0); tail[225 - 153] = 1; tail[299 - 153] = 1;
    f.data(2, tail); f.end(2);
  });
  await expect(status(page, 'distributions')).toHaveCount(0);
  await expect(page.locator('.matrix-surfaces canvas')).toHaveCount(3);
  await expect(page.locator('.tensor-results')).not.toBeVisible();
  await expect(page.getByRole('button', { name: 'Cancel loading' })).toHaveCount(0);
  await expect(page.getByText('Complete', { exact: true })).toHaveCount(0);
  await expect(page.getByText(/One value per device pixel|Focus the matrix/)).toHaveCount(0);
  const overhead = await page.evaluate(() => document.querySelector('.matrix-surfaces')!.getBoundingClientRect().top - document.querySelector('.working-surface')!.getBoundingClientRect().top);
  expect(overhead).toBeLessThanOrEqual(50);
  const path = testInfo.outputPath(`asymmetric-dpr-${dpr}.png`);
  await page.screenshot({ path, fullPage: true });
  await testInfo.attach('asymmetric scientific layout', { path, contentType: 'image/png' });
  await page.evaluate(() => window.explorerFixture.unmount());
  expect(await page.evaluate(() => ({ live: window.explorerFixture.metrics.live.size, states: window.explorerFixture.renderers.map((r) => r.state) }))).toEqual({ live: 0, states: ['disposed', 'disposed', 'disposed'] });
  await context.close();
});

for (const dpr of [1, 2]) test(`reference geometry and synchronized scrolling across texture bands at DPR ${dpr}`, async ({ browser }, testInfo) => {
  const context = await browser.newContext({ viewport: testInfo.project.name === 'narrow' ? { width: 390, height: 844 } : { width: 1440, height: 900 }, deviceScaleFactor: dpr });
  const page = await context.newPage();
  await open(page, 'unsupported');
  await page.addStyleTag({ content: '.matrix-surfaces { max-width: 600px; } .matrix-scroll { max-height: 180px; }' });
  await page.evaluate(() => { window.explorerFixture.metrics.bandLimit = 128; });
  await page.getByRole('button', { name: /^reference \[/ }).click();
  await expect.poll(() => page.evaluate(() => window.explorerFixture.renderers.length)).toBe(3);
  const geometry = await page.evaluate(() => window.explorerFixture.renderers.map((r) => [r.geometry.columns, r.geometry.rows]));
  expect(geometry).toEqual([[1536, 576], [100, 576], [1536, 100]]);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    f.emit(0, 1, f.metadata(0));
    f.data(0, Array.from({ length: 576 * 1536 }, (_, i) => (i % 19) - 9)); f.end(0);
    const metadata = f.metadata(2);
    if (metadata.kind !== 'tensor_distributions') throw new Error('Wrong fixture kind');
    f.emit(2, 1, { ...metadata, domain_minimum: -9, domain_maximum: 9 });
    const counts = Array<number>((576 + 1536) * 100).fill(0);
    for (let row = 0; row < 576; row++) for (let column = 0; column < 1536; column++) {
      const bin = Math.min(99, Math.floor(((row * 1536 + column) % 19) * 100 / 18));
      counts[row * 100 + bin]!++;
      counts[57600 + bin * 1536 + column]!++;
    }
    f.data(2, counts); f.end(2);
  });
  await expect(status(page, 'tensor')).toHaveCount(0);
  await expect(status(page, 'distributions')).toHaveCount(0);
  for (const [x, y] of [[0, 0], [127 / dpr, 127 / dpr], [128 / dpr, 128 / dpr], [256 / dpr, 256 / dpr], [1e6, 1e6]]) {
    await page.locator('.matrix-scroll').evaluate((host, point) => { host.scrollLeft = point[0]!; host.scrollTop = point[1]!; }, [x!, y!]);
    await expect.poll(() => page.evaluate(() => {
      const [matrix, row, column] = window.explorerFixture.renderers;
      const host = document.querySelector('.matrix-scroll')!;
      return matrix!.view!.x === Math.min(1536 - matrix!.view!.width, Math.round(host.scrollLeft * devicePixelRatio)) &&
        matrix!.view!.y === Math.min(576 - matrix!.view!.height, Math.round(host.scrollTop * devicePixelRatio)) &&
        matrix!.view!.y === row!.view!.y && matrix!.view!.x === column!.view!.x;
    })).toBe(true);
    const aligned = await page.evaluate(() => {
      const [matrix, row, column] = window.explorerFixture.renderers;
      const m = matrix!.canvas.getBoundingClientRect(); const r = row!.canvas.getBoundingClientRect(); const c = column!.canvas.getBoundingClientRect();
      const view = matrix!.view!;
      return { top: Math.abs(m.top - r.top) * devicePixelRatio, left: Math.abs(m.left - c.left) * devicePixelRatio,
        cell: matrix!.readCell(view.y, view.x), expected: ((view.y * 1536 + view.x) % 19) - 9,
        pixel: window.explorerFixture.pixels(matrix!)[0]![0]![0],
        rowCount: row!.readCell(view.y, 0), columnCount: column!.readCell(0, view.x),
        rowExpected: Array.from({ length: 1536 }, (_, column) => (view.y * 1536 + column) % 19 === 0).filter(Boolean).length,
        columnExpected: Array.from({ length: 576 }, (_, row) => (row * 1536 + view.x) % 19 === 0).filter(Boolean).length,
        rowPixel: window.explorerFixture.pixels(row!)[0]![0]![0], columnPixel: window.explorerFixture.pixels(column!)[0]![0]![0],
        right: m.right <= r.left, bottom: m.bottom <= c.top };
    });
    expect(aligned.top).toBeLessThan(0.01); expect(aligned.left).toBeLessThan(0.01);
    expect(aligned.cell).toMatchObject({ value: aligned.expected });
    expect(aligned.pixel).toBeCloseTo(color(1 / (1 + Math.exp(-12 * aligned.expected)))[0]!, 0);
    expect(aligned.rowCount).toMatchObject({ value: aligned.rowExpected });
    expect(aligned.columnCount).toMatchObject({ value: aligned.columnExpected });
    expect(aligned.rowPixel).toBe(color(Math.log1p(aligned.rowExpected) / Math.log1p(1536))[0]);
    expect(aligned.columnPixel).toBe(color(Math.log1p(aligned.columnExpected) / Math.log1p(576))[0]);
    expect(aligned.right).toBe(true); expect(aligned.bottom).toBe(true);
  }
  const path = testInfo.outputPath(`reference-dpr-${dpr}.png`);
  await page.screenshot({ path, fullPage: true });
  await testInfo.attach('reference aligned layout', { path, contentType: 'image/png' });
  await context.close();
});

test('auxiliary failure leaves matrix inspectable and failure/cancel prefixes visibly incomplete', async ({ page }) => {
  await open(page);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    f.emit(0, 1, f.metadata(0)); f.data(0, [-2, 0, 2, 1, -1, 0]); f.end(0);
    f.emit(1, 5, { code: 'operation_failed', message: '/private/backend' }); f.requests[1]!.stream.close();
    f.emit(2, 1, f.metadata(2)); f.data(2, [1, 0]);
  });
  await expect(status(page, 'statistics')).toHaveAttribute('data-state', 'failed');
  await expect(status(page, 'tensor')).toHaveCount(0);
  await page.getByRole('button', { name: 'Cancel loading' }).click();
  await expect(status(page, 'distributions')).toHaveAttribute('data-state', 'cancelled');
  await expect(status(page, 'tensor')).toHaveCount(0);
  await expect(page.locator('body')).not.toContainText('/private/backend');
  expect(await page.evaluate(() => window.explorerFixture.renderers[1]!.readCell(0, 2))).toMatchObject({ state: 'pending' });
  await page.getByRole('button', { name: /^B \[/ }).click();
  await page.evaluate(() => { const f = window.explorerFixture; f.emit(3, 1, f.metadata(3)); f.data(3, [0]); f.requests[3]!.stream.close(); });
  await expect(status(page, 'tensor')).toHaveAttribute('data-state', 'failed');
  await page.getByRole('button', { name: 'Cancel loading' }).click();
  expect(await page.evaluate(() => window.explorerFixture.renderers[3]!.populatedPrefix)).toBe(1);
});

test('A → B → A rejects queued callbacks of every kind and releases owned resources', async ({ page }) => {
  await open(page);
  await page.getByRole('button', { name: /^B \[/ }).click();
  await page.getByRole('button', { name: /^A \[/ }).click();
  await expect.poll(() => page.evaluate(() => window.explorerFixture.requests.length)).toBe(9);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    for (let i = 0; i < 6; i++) {
      f.callbacks[i]!.onMetadata?.(f.metadata(i));
      f.callbacks[i]!.onData?.(new Uint8Array(new Float32Array([99]).buffer), 0);
    }
  });
  expect(await page.evaluate(() => window.explorerFixture.renderers.slice(-3).map((r) => r.populatedPrefix))).toEqual([0, 0, 0]);
  expect(await page.evaluate(() => window.explorerFixture.cancelled.length)).toBe(6);
  await page.getByRole('button', { name: 'Cancel loading' }).click();
  await expect(status(page, 'tensor')).toHaveAttribute('data-state', 'cancelled');
  await page.getByRole('button', { name: 'Tokenizer Explorer', exact: true }).click();
  expect(await page.evaluate(() => window.explorerFixture.metrics.live.size)).toBe(0);
  expect(await page.evaluate(() => new Set(window.explorerFixture.cancelled).size)).toBe(9);
});

test('rank-1 strip, empty, unsupported and failed allocation have explicit states', async ({ page }) => {
  await open(page, 'vector');
  await expect(page.locator('.matrix-surfaces canvas')).toHaveCount(1);
  expect(await page.evaluate(() => window.explorerFixture.requests.map((r) => r.kind))).toEqual(['data', 'statistics']);
  await page.evaluate(() => { const f = window.explorerFixture; f.emit(0, 1, f.metadata(0)); f.data(0, [-2, -1, 0, 1, 2]); f.end(0); });
  await expect(status(page, 'tensor')).toHaveCount(0);
  expect(await page.evaluate(() => window.explorerFixture.renderers[0]!.geometry)).toEqual({ columns: 5, rows: 1, count: 5 });
  await page.getByRole('button', { name: /^empty \[/ }).click();
  await expect(page.getByText('Empty tensor — no values to render.')).toBeVisible();
  await page.getByRole('button', { name: /^unsupported \[/ }).click();
  await expect(page.getByText(/This rank-3 tensor/)).toBeVisible();
  expect(await page.evaluate(() => window.explorerFixture.requests.length)).toBe(2);
  await page.evaluate(() => { window.explorerFixture.metrics.failAllocation = true; });
  await page.getByRole('button', { name: /^A \[/ }).click();
  await expect(page.getByRole('alert')).toContainText('Exact rendering is unavailable');
  expect(await page.evaluate(() => window.explorerFixture.metrics.live.size)).toBe(0);
  expect(await page.evaluate(() => window.explorerFixture.requests.length)).toBe(2);
});

test('uint32 density storage preserves large counts and defines zero-axis intensity', async ({ page }) => {
  await open(page, 'unsupported');
  const result = await page.evaluate(() => {
    const f = window.explorerFixture;
    const renderer = new f.DistributionRenderer(document.createElement('canvas'), [1, 3], 0xffffffff);
    renderer.upload(new Uint32Array([0, 16777217, 0xffffffff]));
    renderer.setView(3, 1);
    const values = [renderer.readCell(0, 0), renderer.readCell(0, 1), renderer.readCell(0, 2)];
    const pixels = f.pixels(renderer);
    const zero = new f.DistributionRenderer(document.createElement('canvas'), [1, 1], 0);
    zero.upload(new Uint32Array([0])); zero.setView(1, 1);
    const zeroPixel = f.pixels(zero)[0]![0];
    renderer.dispose(); zero.dispose();
    return { values, pixels, zeroPixel, live: f.metrics.live.size, integer: f.metrics.integerAllocations, scalar: f.metrics.scalarAllocations };
  });
  expect(result.values.map((cell) => cell && 'value' in cell ? cell.value : null)).toEqual([0, 16777217, 0xffffffff]);
  expect(result.pixels[0]![0]).toEqual(green(-1));
  expect(result.pixels[0]![2]).toEqual(green(1));
  expect(result.zeroPixel).toEqual(green(-1));
  expect(result.live).toBe(0); expect(result.integer).toBe(2); expect(result.scalar).toBe(0);
});

test('catalogue refresh preserves the selected prefix and cancelled view without restarting operations', async ({ page }) => {
  await open(page);
  await page.evaluate(() => {
    const f = window.explorerFixture;
    f.emit(0, 1, f.metadata(0)); f.data(0, [-2, 0, 2]);
  });
  await expect(status(page, 'tensor')).toHaveAttribute('data-state', 'streaming');
  const snapshot = () => page.evaluate(() => {
    const f = window.explorerFixture;
    return { requests: f.requests.length, cancellations: f.cancelled.length,
      allocations: [f.metrics.scalarAllocations, f.metrics.integerAllocations], uploads: f.metrics.uploads,
      renderers: f.renderers.map((r) => ({ state: r.state, prefix: r.populatedPrefix })),
      first: f.renderers[0]!.readCell(0, 0), pending: f.renderers[0]!.readCell(1, 0) };
  });
  for (const cancelled of [false, true]) {
    if (cancelled) await page.getByRole('button', { name: 'Cancel loading' }).click();
    const before = await snapshot();
    expect(before.requests).toBe(3);
    expect(before.allocations).toEqual([1, 2]);
    expect(before.renderers[0]!.prefix).toBe(3);
    await page.evaluate(() => { window.explorerFixture.catalogue.paused = true; });
    await page.getByRole('button', { name: 'Refresh models' }).click();
    await expect(page.getByText('Loading models…')).toBeVisible();
    expect(await snapshot()).toEqual(before);
    await page.evaluate(() => window.explorerFixture.catalogue.resume());
    await expect(page.getByRole('combobox')).toBeEnabled();
    expect(await snapshot()).toEqual(before);
    await expect(status(page, 'tensor')).toHaveAttribute('data-state', cancelled ? 'cancelled' : 'streaming');
  }
});
