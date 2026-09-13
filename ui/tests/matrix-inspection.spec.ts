import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import type {} from './tensor-explorer-harness';

const decode = (byte: number) => byte / 255 <= 0.04045 ? byte / 255 / 12.92 : ((byte / 255 + 0.055) / 1.055) ** 2.4;
const luminance = (rgb: number[]) => rgb.slice(0, 3).reduce((sum, v, i) => sum + decode(v) * [0.2126, 0.7152, 0.0722][i]!, 0);
// Independent display oracle, with exact float32 input and double-precision transfer.
function expected(value: number, row: boolean, column: boolean) {
  const y = 1 / (1 + Math.exp(-8 * value));
  const direction = [1, (0.0722 * 0.6 - 0.2126) / 0.7152, -0.6];
  let t = row && column ? 0.3 : row || column ? 0.08 : 0;
  direction.forEach((c) => { t = Math.min(t, c > 0 ? (1 - y) / c : -y / c); });
  return direction.map((c) => {
    const v = y + c * t;
    return Math.round(255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055));
  });
}
async function open(page: Page, prefix = 323) {
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html`);
  await expect(page.getByRole('combobox')).toBeEnabled();
  await page.getByRole('combobox').selectOption('lab/alpha');
  await page.evaluate(() => { window.explorerFixture.metrics.bandLimit = 8; });
  await page.getByRole('button', { name: /^inspection \[/ }).click();
  await expect.poll(() => page.evaluate(() => window.explorerFixture.requests.length)).toBe(3);
  await page.evaluate((prefix) => {
    const f = window.explorerFixture;
    f.emit(0, 1, f.metadata(0));
    f.data(0, Array.from({ length: prefix }, (_, i) => i === 0 ? -0 : (i % 17 - 8) / 32));
    if (prefix === 323) f.end(0);
    f.emit(2, 1, f.metadata(2));
    const counts = Array<number>(3600).fill(0);
    for (let row = 0; row < 17; row++) for (let column = 0; column < 19; column++) {
      const value = row === 0 && column === 0 ? -0 : ((row * 19 + column) % 17 - 8) / 32;
      const bin = Math.floor((value + 2) * 25);
      counts[row * 100 + bin]!++; counts[1700 + bin * 19 + column]!++;
    }
    f.data(2, counts); f.end(2);
  }, prefix);
  await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', prefix === 323 ? 'complete' : 'streaming');
  await expect(page.locator('[data-result="distributions"]')).toHaveAttribute('data-state', 'complete');
  await page.locator('.matrix-scroll').scrollIntoViewIfNeeded();
}
async function hover(page: Page, row: number, column: number) {
  const point = await page.evaluate(({ row, column }) => {
    const r = window.explorerFixture.renderers[0]!;
    const rect = r.canvas.getBoundingClientRect();
    return { x: rect.left + (column - r.view!.x + 0.5) / devicePixelRatio,
      y: rect.top + (row - r.view!.y + 0.5) / devicePixelRatio };
  }, { row, column });
  await page.mouse.move(point.x, point.y);
  await expect(page.locator('.inspection-readout')).toContainText(`row ${row} · column ${column}`);
}
const resources = (page: Page) => page.evaluate(() => {
  const f = window.explorerFixture;
  return { textureCreates: f.metrics.textureCreates, float32Allocations: f.metrics.float32Allocations, fetches: f.metrics.fetches, scalar: f.metrics.scalarAllocations, integer: f.metrics.integerAllocations, uploads: f.metrics.uploads,
    live: f.metrics.live.size, requests: f.requests.length, cpu: f.renderers.map((r) => r.diagnostics.cpuBytes),
    geometry: f.renderers.map((r) => r.view), scroll: [document.querySelector('.matrix-scroll')!.scrollLeft, document.querySelector('.matrix-scroll')!.scrollTop] };
});

for (const dpr of [1, 2]) test.describe(`inspection DPR ${dpr}`, () => {
  test.use({ deviceScaleFactor: dpr });
  test('exact hover, linked pixels, all 81 nearest-neighbor cells, no scalar or network work', async ({ page }, testInfo) => {
    await open(page);
    const before = await resources(page);
    for (const [row, column] of [[0, 0], [7, 7], [8, 8], [9, 12], [16, 18]]) {
      await hover(page, row!, column!);
      const actual = await page.evaluate(() => {
        const f = window.explorerFixture;
        const canvas = document.querySelector<HTMLCanvasElement>('.magnifier-card canvas')!;
        return { matrix: f.pixels(f.renderers[0]!), rows: f.pixels(f.renderers[1]!), columns: f.pixels(f.renderers[2]!),
          neighborhood: Array.from(canvas.getContext('2d')!.getImageData(0, 0, 9, 9).data),
          counts: f.renderers.slice(1).map((r) => Array.from({ length: r.geometry.rows }, (_, row) =>
            Array.from({ length: r.geometry.columns }, (_, column) => { const c = r.readCell(row, column)!; return 'value' in c ? c.value : -1; }))),
          value: document.querySelector('.inspection-readout')!.lastElementChild!.textContent,
          card: document.querySelector('.matrix-inspection')!.getBoundingClientRect().toJSON(),
          viewport: [innerWidth, innerHeight], displayAllocations: f.metrics.displayAllocations };
      });
      const value = row! === 0 && column! === 0 ? '-0' : String(Math.fround(((row! * 19 + column!) % 17 - 8) / 32));
      expect(actual.value).toBe(value);
      let maxColorError = 0, maxLuminanceError = 0, wrongScanlines = 0;
      // Check every matrix pixel: exactly one row and column, intersection stronger.
      for (let y = 0; y < 17; y++) for (let x = 0; x < 19; x++) {
        const value = y === 0 && x === 0 ? -0 : Math.fround(((y * 19 + x) % 17 - 8) / 32);
        const rgb = actual.matrix[y]![x]!;
        expected(value, y === row, x === column).forEach((v, i) => { maxColorError = Math.max(maxColorError, Math.abs(rgb[i]! - v)); });
        maxLuminanceError = Math.max(maxLuminanceError, Math.abs(luminance(rgb) - 1 / (1 + Math.exp(-8 * value))));
      }
      for (const [pixels, axis] of [[actual.rows, 'row'], [actual.columns, 'column']] as const) {
        for (let y = 0; y < pixels.length; y++) for (let x = 0; x < pixels[y]!.length; x++) {
          const rgb = pixels[y]![x]!;
          const count = actual.counts[axis === 'row' ? 0 : 1]![y]![x]!;
          const selected = (axis === 'row' ? y === row : x === column) && count > 0;
          if ((rgb[0] !== rgb[1]) !== selected) wrongScanlines++;
          maxLuminanceError = Math.max(maxLuminanceError, Math.abs(luminance(rgb) - Math.log1p(count) / Math.log1p(axis === 'row' ? 19 : 17)));
        }
      }
      expect(maxColorError).toBeLessThanOrEqual(1);
      expect(maxLuminanceError).toBeLessThanOrEqual(0.0045);
      expect(wrongScanlines).toBe(0);
      for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) {
        const sourceY = row! + y - 4, sourceX = column! + x - 4;
        const pixel = actual.neighborhood.slice((y * 9 + x) * 4, (y * 9 + x + 1) * 4);
        expect(pixel).toEqual(sourceY < 0 || sourceY >= 17 || sourceX < 0 || sourceX >= 19 ? [46, 61, 76, 255] : actual.matrix[sourceY]![sourceX]);
      }
      // Inspect actual composited enlargement, not just its 9×9 source buffer.
      const screenshot = await page.locator('.magnifier-card canvas').screenshot();
      const enlarged = await page.evaluate(async (base64) => {
        const img = new Image(); img.src = `data:image/png;base64,${base64}`; await img.decode();
        const canvas = document.createElement('canvas'); canvas.width = img.width; canvas.height = img.height;
        const context = canvas.getContext('2d')!; context.drawImage(img, 0, 0);
        const samples = [];
        for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) {
          for (const offset of [3, 9, 15]) samples.push(Array.from(context.getImageData(
            Math.floor((x * 18 + offset) * img.width / 162), Math.floor((y * 18 + offset) * img.height / 162), 1, 1).data));
        }
        return samples;
      }, screenshot.toString('base64'));
      enlarged.forEach((pixel, index) => expect(pixel).toEqual(actual.neighborhood.slice(Math.floor(index / 3) * 4, (Math.floor(index / 3) + 1) * 4)));
      expect(actual.card.x).toBeGreaterThanOrEqual(0); expect(actual.card.y).toBeGreaterThanOrEqual(0);
      expect(actual.card.right).toBeLessThanOrEqual(actual.viewport[0]!);
      expect(actual.card.bottom).toBeLessThanOrEqual(actual.viewport[1]!);
      expect(actual.displayAllocations).toBe(1);
      if (row === 8) {
        const path = testInfo.outputPath('inspection.png');
        await page.screenshot({ path });
        await testInfo.attach('inspection-composition', { path, contentType: 'image/png' });
      }
      await testInfo.attach(`pixel-data-${row}-${column}`, { body: JSON.stringify(actual), contentType: 'application/json' });
    }
    expect(await resources(page)).toEqual(before);
    await page.mouse.move(0, 0);
    await expect(page.locator('.matrix-inspection')).toHaveCount(0);
    expect(await page.evaluate(() => window.explorerFixture.pixels(window.explorerFixture.renderers[0]!).every((row) => row.every((p) => p[0] === p[1] && p[1] === p[2])))).toBe(true);
    await page.evaluate(() => window.explorerFixture.unmount());
    expect(await page.evaluate(() => [window.explorerFixture.metrics.live.size, window.explorerFixture.metrics.liveDisplays.size])).toEqual([0, 0]);
  });

  test('card flips within all viewport corners without covering the inspected neighborhood', async ({ page }) => {
    await open(page);
    // This isolated magnifier test deliberately moves data outside the workspace.
    // Hide the fixed application bars so they do not intercept those corner probes.
    await page.addStyleTag({ content: '.app-bar, .app-status-bar { visibility: hidden; }' });
    for (const [right, bottom] of [[false, false], [true, false], [false, true], [true, true]]) {
      await page.locator('.matrix-surfaces').evaluate((host: HTMLElement, [right, bottom]) => {
        Object.assign(host.style, { position: 'fixed', width: `${119 / devicePixelRatio + 10}px`, zIndex: '10', left: `${right ? innerWidth - 19 / devicePixelRatio - 2 : 2}px`,
          top: `${bottom ? innerHeight - 17 / devicePixelRatio - 2 : 2}px` });
      }, [right!, bottom!]);
      await hover(page, 8, 8);
      const geometry = await page.evaluate(() => {
        const card = document.querySelector('.matrix-inspection')!.getBoundingClientRect();
        const main = window.explorerFixture.renderers[0]!.canvas.getBoundingClientRect();
        return { card: card.toJSON(), x: main.x + 8.5 / devicePixelRatio, y: main.y + 8.5 / devicePixelRatio, width: innerWidth, height: innerHeight };
      });
      const { card, x, y, width, height } = geometry;
      expect(card.left).toBeGreaterThanOrEqual(0); expect(card.top).toBeGreaterThanOrEqual(0);
      expect(card.right).toBeLessThanOrEqual(width); expect(card.bottom).toBeLessThanOrEqual(height);
      expect(card.left > x + 5 / dpr || card.right < x - 5 / dpr || card.top > y + 5 / dpr || card.bottom < y - 5 / dpr).toBe(true);
    }
  });

  test('native scrolling resolves stationary pointer across bands and keyboard focus has exact text', async ({ page }) => {
    await open(page);
    await page.addStyleTag({ content: `.matrix-scroll { max-width: ${8 / dpr}px !important; max-height: ${8 / dpr}px !important; }` });
    await expect.poll(() => page.evaluate(() => window.explorerFixture.renderers[0]!.view!.width)).toBe(8);
    for (const [x, y] of [[0, 0], [7, 7], [8, 8], [11, 9]]) {
      await page.locator('.matrix-scroll').evaluate((host, [x, y]) => { host.scrollLeft = x! / devicePixelRatio; host.scrollTop = y! / devicePixelRatio; }, [x!, y!]);
      await expect.poll(() => page.evaluate(() => {
        const v = window.explorerFixture.renderers[0]!.view!, host = document.querySelector('.matrix-scroll')!;
        return v.x === Math.min(19 - v.width, Math.round(host.scrollLeft * devicePixelRatio)) &&
          v.y === Math.min(17 - v.height, Math.round(host.scrollTop * devicePixelRatio));
      })).toBe(true);
      const origin = await page.evaluate(() => window.explorerFixture.renderers[0]!.view!);
      await hover(page, origin.y, origin.x);
    }
    await hover(page, 16, 18);
    await page.locator('.matrix-scroll').focus();
    await page.keyboard.press('ArrowLeft');
    await expect(page.locator('.inspection-readout')).toContainText('row 9 · column 10');
    await page.keyboard.press('Escape');
    await expect(page.locator('.matrix-inspection')).toHaveCount(0);
    await page.getByRole('button', { name: /^B \[/ }).click();
    await expect(page.locator('.matrix-inspection')).toHaveCount(0);
    expect(await page.evaluate(() => window.explorerFixture.metrics.liveDisplays.size)).toBe(0);
  });
});

test('missing/nonfinite values, stream arrival and late transfer update the active inspection', async ({ page }) => {
  await open(page, 10);
  await hover(page, 0, 10);
  await expect(page.locator('.inspection-readout')).toContainText('Unavailable — not received');
  expect(await page.locator('.magnifier-card canvas').evaluate((canvas: HTMLCanvasElement) => Array.from(canvas.getContext('2d')!.getImageData(4, 4, 1, 1).data))).toEqual([46, 61, 76, 255]);
  await page.evaluate(() => { const f = window.explorerFixture; f.data(0, [Infinity, -Infinity, NaN, 1.0000001192092896]); });
  for (const [column, value] of [[10, 'Infinity'], [11, '-Infinity'], [12, 'NaN'], [13, '1.0000001192092896']] as const) {
    await hover(page, 0, column);
    await expect(page.locator('.inspection-readout')).toContainText(value);
  }
  const previous = await page.locator('.magnifier-card canvas').evaluate((canvas: HTMLCanvasElement) => Array.from(canvas.getContext('2d')!.getImageData(4, 4, 1, 1).data));
  await page.evaluate(() => { const f = window.explorerFixture; f.emit(1, 1, f.metadata(1)); f.end(1); });
  await expect.poll(() => page.locator('.magnifier-card canvas').evaluate((canvas: HTMLCanvasElement) => Array.from(canvas.getContext('2d')!.getImageData(4, 4, 1, 1).data))).not.toEqual(previous);
  const errors: string[] = []; page.on('pageerror', (e) => errors.push(e.message));
  await page.getByRole('button', { name: /^B \[/ }).click();
  await expect(page.locator('.matrix-inspection')).toHaveCount(0);
  expect(errors).toEqual([]);
});


test('matrix context loss clears active text and inspection resources', async ({ page }) => {
  await open(page);
  await hover(page, 8, 8);
  await page.evaluate(() => {
    window.explorerFixture.renderers[0]!.canvas.getContext('webgl2')!.getExtension('WEBGL_lose_context')!.loseContext();
  });
  await expect(page.locator('.matrix-inspection')).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => window.explorerFixture.metrics.liveDisplays.size)).toBe(0);
  await expect(page.getByRole('alert')).toContainText('Exact rendering is unavailable');
});
