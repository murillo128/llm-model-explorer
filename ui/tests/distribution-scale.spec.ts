import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import { nativeCamera } from './native-camera';
import { color } from './scalar-oracle';

const cases = [
  { name: 'symmetric', tensor: 'A', values: [-2, 0, 2, -1, 0, 1], bins: [0, 50, 99, 25, 50, 75], low: -2, high: 2, zero: 0.5 },
  { name: 'asymmetric', tensor: 'A', values: [-2, 0, 6, 0, 2, 4], bins: [0, 25, 99, 25, 50, 75], low: -2, high: 6, zero: 0.25 },
  { name: 'positive only', tensor: 'A', values: [2, 4, 10, 2, 6, 8], bins: [0, 25, 99, 0, 50, 75], low: 2, high: 10, zero: null },
  { name: 'negative only', tensor: 'A', values: [-10, -8, -2, -10, -6, -4], bins: [0, 25, 99, 0, 50, 75], low: -10, high: -2, zero: null },
  { name: 'zero at low boundary', tensor: 'A', values: [0, 2, 8, 0, 4, 6], bins: [0, 25, 99, 0, 50, 75], low: 0, high: 8, zero: 0 },
  { name: 'zero at high boundary', tensor: 'A', values: [-8, -6, 0, -8, -4, -2], bins: [0, 25, 99, 0, 50, 75], low: -8, high: 0, zero: 1 },
  { name: 'outliers', tensor: 'distribution-outliers', values: [-10000, 10000, ...Array.from({ length: 198 }, (_, i) => [-0.125, 0, 0.125][i % 3]!)],
    bins: [0, 99, ...Array.from({ length: 198 }, (_, i) => i % 3 === 0 ? 49 : 50)], low: -10000, high: 10000, zero: 0.5 },
  { name: 'constant zero', tensor: 'A', values: [0, 0, 0, 0, 0, 0], bins: [50, 50, 50, 50, 50, 50], low: 0, high: 0, zero: null },
  { name: 'constant positive', tensor: 'A', values: [3, 3, 3, 3, 3, 3], bins: [50, 50, 50, 50, 50, 50], low: 3, high: 3, zero: null },
  { name: 'nonfinite', tensor: 'A', values: [NaN, Infinity, -Infinity, NaN, NaN, NaN], bins: [-1, -1, -1, -1, -1, -1], low: null, high: null, zero: null },
];

async function expectAlignedRulers(page: Page, finite: boolean, zero: number | null) {
  const geometry = await page.evaluate(() => {
    const bounds = (selector: string) => {
      const rect = document.querySelector(selector)?.getBoundingClientRect();
      return rect && { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom, width: rect.width, height: rect.height };
    };
    return {
      row: bounds('.row-distributions canvas')!, ruler: bounds('.distribution-scale-rows')!,
      low: bounds('.distribution-scale-rows .distribution-low'), high: bounds('.distribution-scale-rows .distribution-high'),
      zero: bounds('.distribution-scale-rows .distribution-zero-tick'),
      column: bounds('.column-distributions canvas')!, columnRuler: bounds('.distribution-scale-columns')!,
      columnLow: bounds('.distribution-scale-columns .distribution-low'), columnHigh: bounds('.distribution-scale-columns .distribution-high'),
    };
  });
  // Compare the chrome to the actual pixel-aligned canvases, including fractional
  // layout offsets, rather than to their grid tracks or nominal 100-bin depth.
  expect(geometry.ruler.left).toBeCloseTo(geometry.row.left, 1);
  expect(geometry.ruler.right).toBeCloseTo(geometry.row.right, 1);
  expect(geometry.ruler.height).toBe(24);
  expect(geometry.ruler.bottom).toBeLessThanOrEqual(geometry.row.top);
  expect(geometry.columnRuler.top).toBeCloseTo(geometry.column.top, 1);
  expect(geometry.columnRuler.bottom).toBeCloseTo(geometry.column.bottom, 1);
  if (finite) {
    expect(geometry.low!.left).toBeCloseTo(geometry.row.left, 1);
    expect(geometry.high!.right).toBeCloseTo(geometry.row.right, 1);
    const labels = [geometry.low!, geometry.high!, geometry.zero].filter(label => label && label.width > 0 && label.height > 0);
    for (let i = 0; i < labels.length; i++) {
      const label = labels[i]!;
      expect(label.bottom).toBeLessThanOrEqual(geometry.row.top);
      for (let j = i + 1; j < labels.length; j++) {
        const other = labels[j]!;
        expect(label.right <= other.left || other.right <= label.left || label.bottom <= other.top || other.bottom <= label.top,
          `row ruler labels ${i} and ${j} must not overlap`).toBe(true);
      }
    }
    expect(geometry.columnLow!.top).toBeCloseTo(geometry.column.top, 1);
    expect(geometry.columnHigh!.bottom).toBeCloseTo(geometry.column.bottom, 1);
  } else {
    expect(geometry.low).toBeUndefined();
    expect(geometry.high).toBeUndefined();
  }
  if (zero === null) expect(geometry.zero).toBeUndefined();
  else expect(geometry.zero!.left + geometry.zero!.width / 2).toBeCloseTo(geometry.row.left + zero * geometry.row.width, 1);
}

for (const fixture of cases) test(`${fixture.name}: authoritative domain, bin placement and independent density`, async ({ page }) => {
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html`);
  await expect(page.getByRole('combobox')).toBeEnabled();
  await page.getByRole('combobox').selectOption('lab/alpha');
  await page.getByRole('button', { name: new RegExp(`^${fixture.tensor} \\[` ) }).click();
  await page.getByRole('button', { name: 'Tensor information', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Tensor information' })).toContainText('Bin domain unavailable');
  await page.keyboard.press('Escape');
  await nativeCamera(page);
  await expectAlignedRulers(page, false, null);
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
  await page.getByRole('button', { name: 'Tensor information', exact: true }).click();
  if (fixture.low === null) {
    await expect(page.getByRole('dialog', { name: 'Tensor information' })).toContainText('No finite values');
    expect(await scales.locator('.distribution-endpoint').count()).toBe(0);
  } else {
    for (const orientation of ['Row', 'Column']) await expect(page.getByRole('img', { name: new RegExp(`^${orientation} bin domain: ${fixture.low} to ${fixture.high};`) })).toBeVisible();
    await expect(page.getByTitle(`True finite minimum: ${fixture.low}`)).toBeVisible();
    await expect(page.getByTitle(`True finite maximum: ${fixture.high}`)).toBeVisible();
  }
  await page.keyboard.press('Escape');
  await expectAlignedRulers(page, fixture.low !== null, fixture.zero);
  const drawing = await page.evaluate(() => {
    const f = window.explorerFixture;
    const [matrix, rows, columns] = f.renderers;
    return {
      counts: [rows!, columns!].map(r => Array.from({ length: r.geometry.count }, (_, i) => r.readCell(Math.floor(i / r.geometry.columns), i % r.geometry.columns))),
      rowPixels: f.pixels(rows!), columnPixels: f.pixels(columns!),
      columns: matrix!.geometry.columns, uploads: f.metrics.uploads,
      guides: [...document.querySelectorAll<HTMLElement>('.distribution-zero')].map(e => {
        const panel = e.parentElement!.querySelector('canvas')!.getBoundingClientRect(), guide = e.getBoundingClientRect();
        const highBoundary = e.style.getPropertyValue('--distribution-zero') === '100%';
        return { hidden: e.hidden, position: e.style.getPropertyValue('--distribution-zero'),
          fraction: e.classList.contains('distribution-zero-rows')
            ? ((highBoundary ? guide.right : guide.left) - panel.left) / panel.width
            : ((highBoundary ? guide.bottom : guide.top) - panel.top) / panel.height };
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
  const domainMarkup = () => scales.evaluateAll(elements => elements.map(e => ({
    label: e.getAttribute('aria-label'), minimum: e.getAttribute('data-minimum'), maximum: e.getAttribute('data-maximum'),
    ticks: e.innerHTML, zero: (e as HTMLElement).style.getPropertyValue('--distribution-zero'),
  })));
  const before = await domainMarkup();
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
  // Escape can restore the prior camera, moving attached rulers. Their numeric
  // domain and ticks remain authoritative and independent of that presentation.
  expect(await domainMarkup()).toEqual(before);
  expect(await page.evaluate(() => window.explorerFixture.metrics.uploads)).toBe(drawing.uploads);
});

test.describe('DPR 2 decimal ruler labels', () => {
  test.use({ deviceScaleFactor: 2 });
  const decimalDomains = [
    { name: 'signed symmetric', low: -0.124, high: 0.124 },
    { name: 'signed asymmetric', low: -0.124, high: 0.372 },
    { name: 'zero near low endpoint', low: -0.00124, high: 0.124 },
    { name: 'zero near high endpoint', low: -0.124, high: 0.00124 },
    { name: 'positive only', low: 0.124, high: 0.372 },
    { name: 'negative only', low: -0.372, high: -0.124 },
    { name: 'constant positive', low: 0.124, high: 0.124 },
    { name: 'constant negative', low: -0.124, high: -0.124 },
  ];

  for (const domain of decimalDomains) test(`${domain.name}: readable labels preserve pending geometry and exact anchors`, async ({ page }) => {
    await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html`);
    await expect(page.getByRole('combobox')).toBeEnabled();
    await page.getByRole('combobox').selectOption('lab/alpha');
    await page.getByRole('button', { name: /^A \[/ }).click();
    await nativeCamera(page);
    await expect(page.locator('.distribution-scale-rows')).toHaveAttribute('aria-label', 'Row bin domain: unavailable');
    await expectAlignedRulers(page, false, null);
    const scientificGeometry = () => page.locator('.matrix-surfaces, .matrix-scroll, .matrix-scroll canvas, .row-distributions canvas, .column-distributions canvas')
      .evaluateAll(elements => elements.map(element => {
        const rect = element.getBoundingClientRect();
        return { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
      }));
    const before = await scientificGeometry();
    const low = Math.fround(domain.low), high = Math.fround(domain.high);
    await page.evaluate(({ low, high }) => {
      const f = window.explorerFixture;
      f.emit(0, 1, f.metadata(0)); f.data(0, [low, high, low, high, low, high]); f.end(0);
      f.emit(2, 1, { ...f.metadata(2), domain_minimum: low, domain_maximum: high });
    }, { low, high });
    const ruler = page.locator('.distribution-scale-rows');
    await expect(ruler).toHaveAttribute('data-minimum', String(low));
    await expect(ruler).toHaveAttribute('data-maximum', String(high));
    await expect(ruler.locator('.distribution-low')).toHaveAttribute('title', `Bin domain low: ${low}`);
    await expect(ruler.locator('.distribution-high')).toHaveAttribute('title', `Bin domain high: ${high}`);
    const zero = low < 0 && high > 0 ? -low / (high - low) : null;
    await expectAlignedRulers(page, true, zero);
    expect(await scientificGeometry()).toEqual(before);
    expect(await page.locator('.row-distributions canvas').evaluate(canvas => canvas.getBoundingClientRect().width)).toBe(50);
    expect(await page.locator('.column-distributions canvas').evaluate(canvas => canvas.getBoundingClientRect().height)).toBe(50);
  });
});

test.describe('fractional physical-pixel layout', () => {
  test.use({ deviceScaleFactor: 1.25 });
  test('rulers follow the histogram canvases after viewport resizing', async ({ page }) => {
    await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html`);
    await expect(page.getByRole('combobox')).toBeEnabled();
    await page.getByRole('combobox').selectOption('lab/alpha');
    await page.getByRole('button', { name: /^A \[/ }).click();
    // Exercise a fractional layout edge as well as a fractional device scale.
    await page.addStyleTag({ content: '.matrix-surfaces { margin-right: 0.25px; }' });
    await nativeCamera(page);
    await page.evaluate(() => {
      const f = window.explorerFixture;
      f.emit(2, 1, { ...f.metadata(2), domain_minimum: -2, domain_maximum: 6 });
    });
    await expect(page.getByRole('img', { name: /^Row bin domain: -2 to 6;/ })).toBeVisible();
    await expectAlignedRulers(page, true, 0.25);
    expect(await page.locator('.row-distributions canvas').evaluate(canvas => (canvas as HTMLElement).style.left)).not.toBe('0px');
    await page.setViewportSize({ width: 601, height: 841 });
    await nativeCamera(page);
    await expectAlignedRulers(page, true, 0.25);
  });
});
