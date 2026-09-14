import { expect, test } from '@playwright/test';
import { nativeCamera } from './native-camera';
import { color } from './scalar-oracle';

const cases = [
  { name: 'symmetric', tensor: 'A', values: [-2, 0, 2, -1, 0, 1], bins: [0, 50, 99, 25, 50, 75], low: -2, high: 2, zero: 0.5 },
  { name: 'asymmetric', tensor: 'A', values: [-2, 0, 6, 0, 2, 4], bins: [0, 25, 99, 25, 50, 75], low: -2, high: 6, zero: 0.25 },
  { name: 'outliers', tensor: 'distribution-outliers', values: [-10000, 10000, ...Array.from({ length: 198 }, (_, i) => [-0.125, 0, 0.125][i % 3]!)],
    bins: [0, 99, ...Array.from({ length: 198 }, (_, i) => i % 3 === 0 ? 49 : 50)], low: -10000, high: 10000, zero: 0.5 },
  { name: 'constant zero', tensor: 'A', values: [0, 0, 0, 0, 0, 0], bins: [50, 50, 50, 50, 50, 50], low: 0, high: 0, zero: null },
  { name: 'constant positive', tensor: 'A', values: [3, 3, 3, 3, 3, 3], bins: [50, 50, 50, 50, 50, 50], low: 3, high: 3, zero: null },
  { name: 'nonfinite', tensor: 'A', values: [NaN, Infinity, -Infinity, NaN, NaN, NaN], bins: [-1, -1, -1, -1, -1, -1], low: null, high: null, zero: null },
];

for (const fixture of cases) test(`${fixture.name}: authoritative domain, bin placement and independent density`, async ({ page }) => {
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html`);
  await expect(page.getByRole('combobox')).toBeEnabled();
  await page.getByRole('combobox').selectOption('lab/alpha');
  await page.getByRole('button', { name: new RegExp(`^${fixture.tensor} \\[` ) }).click();
  await expect(page.getByLabel('Distribution range')).toHaveText('Bin domain unavailable');
  await nativeCamera(page);
  const position = () => page.locator('.matrix-surfaces').evaluate(node => node.getBoundingClientRect().top);
  const initialTop = await position();
  await page.evaluate((fixture) => {
    const f = window.explorerFixture;
    f.emit(0, 1, f.metadata(0)); f.data(0, fixture.values); f.end(0);
    const m = f.metadata(2);
    if (m.kind !== 'tensor_distributions') throw new Error('Unexpected fixture');
    f.emit(2, 1, { ...m, domain_minimum: fixture.low, domain_maximum: fixture.high });
    const counts = Array<number>(m.byte_length / 4).fill(0);
    fixture.bins.forEach((bin, i) => {
      if (bin < 0) return;
      counts[Math.floor(i / m.columns) * 100 + bin]!++;
      counts[m.rows * 100 + bin * m.columns + i % m.columns]!++;
    });
    f.data(2, counts, 17); f.end(2);
  }, fixture);
  await expect(page.locator('[data-result="distributions"]')).toHaveCount(0);
  expect(await position()).toBe(initialTop);
  const scales = page.locator('.distribution-scale');
  if (fixture.low === null) {
    await expect(page.getByLabel('Distribution range')).toContainText('No finite values');
    expect(await scales.locator('.distribution-endpoint').count()).toBe(0);
  } else {
    for (const orientation of ['Row', 'Column']) await expect(page.getByRole('img', { name: new RegExp(`^${orientation} bin domain: ${fixture.low} to ${fixture.high};`) })).toBeVisible();
    await expect(page.getByTitle(`True finite minimum: ${fixture.low}`)).toBeVisible();
    await expect(page.getByTitle(`True finite maximum: ${fixture.high}`)).toBeVisible();
  }
  const drawing = await page.evaluate(() => {
    const f = window.explorerFixture;
    const [matrix, rows, columns] = f.renderers;
    return {
      counts: [rows!, columns!].map(r => Array.from({ length: r.geometry.count }, (_, i) => r.readCell(Math.floor(i / r.geometry.columns), i % r.geometry.columns))),
      rowPixels: f.pixels(rows!), columnPixels: f.pixels(columns!),
      columns: matrix!.geometry.columns, uploads: f.metrics.uploads,
      guides: [...document.querySelectorAll<HTMLElement>('.distribution-zero')].map(e => {
        const panel = e.parentElement!.getBoundingClientRect(), guide = e.getBoundingClientRect();
        return { hidden: e.hidden, position: e.style.getPropertyValue('--distribution-zero'),
          fraction: e.classList.contains('distribution-zero-rows') ? (guide.left - panel.left) / panel.width : (guide.top - panel.top) / panel.height };
      }),
    };
  });
  const rowCounts = Array<number>(200).fill(0), columnCounts = Array<number>(100 * drawing.columns).fill(0);
  fixture.bins.forEach((bin, i) => {
    if (bin < 0) return;
    rowCounts[Math.floor(i / drawing.columns) * 100 + bin]!++;
    columnCounts[bin * drawing.columns + i % drawing.columns]!++;
  });
  expect(drawing.counts.map(counts => counts.map(cell => cell && 'value' in cell ? cell.value : null))).toEqual([rowCounts, columnCounts]);
  // Pixel oracle checks the density transfer across every bin, including empty bins.
  for (const [axis, counts, pixels, width, length] of [
    ['row', rowCounts, drawing.rowPixels, 100, drawing.columns],
    ['column', columnCounts, drawing.columnPixels, drawing.columns, 2],
  ] as const) for (let y = 0; y < pixels.length; y++) for (let x = 0; x < pixels[y]!.length; x++) {
    expect(pixels[y]![x], `${axis} ${y}:${x}`).toEqual([...color(Math.log1p(counts[y * width + x]!) / Math.log1p(length)), 255]);
  }
  expect(drawing.guides.every(g => fixture.zero === null ? g.hidden : !g.hidden && g.position === `${fixture.zero * 100}%`)).toBe(true);
  if (fixture.zero !== null) for (const guide of drawing.guides) expect(guide.fraction).toBeCloseTo(fixture.zero, 4);
  const before = await scales.evaluateAll(elements => elements.map(e => e.outerHTML));
  // Late robust statistics can alter weight luminosity, never the full bin domain.
  await page.evaluate((fixture) => {
    const f = window.explorerFixture;
    const m = f.metadata(1);
    if (m.kind !== 'tensor_statistics') throw new Error('Unexpected fixture');
    const lo = fixture.low, hi = fixture.high;
    const p01 = fixture.name === 'outliers' ? -0.125 : lo;
    const p99 = fixture.name === 'outliers' ? 0.125 : hi;
    f.emit(1, 1, { ...m, finite_count: lo === null ? 0 : m.count, non_finite_count: lo === null ? m.count : 0,
      minimum: lo, maximum: hi, mean: lo === null ? null : (lo + hi!) / 2, stddev: lo === null ? null : 0,
      percentiles: { p01, p05: p01, p50: lo === null ? null : (lo + hi!) / 2, p95: p99, p99 } });
    f.end(1);
  }, fixture);
  await expect(page.locator('[data-result="statistics"]')).toHaveCount(0);
  expect(await position()).toBe(initialTop);
  await page.locator('.matrix-scroll canvas').hover({ position: { x: 0.5, y: 0.5 } });
  await page.locator('.matrix-scroll').focus();
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('Escape');
  expect(await scales.evaluateAll(elements => elements.map(e => e.outerHTML))).toEqual(before);
  expect(await page.evaluate(() => window.explorerFixture.metrics.uploads)).toBe(drawing.uploads);
});
