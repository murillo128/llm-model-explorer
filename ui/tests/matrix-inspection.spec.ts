import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import type {} from './tensor-explorer-harness';

import { nativeCamera } from './native-camera';
import { color } from './scalar-oracle';
function expected(value: number, row: boolean, column: boolean) {
  return color(1 / (1 + Math.exp(-12 * value)), row && column ? 0.9 : row || column ? 0.65 : 0);
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
  if (prefix === 323) await expect(page.locator('[data-result="tensor"]')).toHaveCount(0);
  else await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'streaming');
  await expect(page.locator('[data-result="distributions"]')).toHaveCount(0);
  await page.locator('.matrix-scroll').scrollIntoViewIfNeeded();
  await nativeCamera(page);
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
      let maxColorError = 0, wrongScanlines = 0;
      // Check every matrix pixel: exactly one row and column, intersection stronger.
      for (let y = 0; y < 17; y++) for (let x = 0; x < 19; x++) {
        const value = y === 0 && x === 0 ? -0 : Math.fround(((y * 19 + x) % 17 - 8) / 32);
        const rgb = actual.matrix[y]![x]!;
        expected(value, y === row, x === column).forEach((v, i) => { maxColorError = Math.max(maxColorError, Math.abs(rgb[i]! - v)); });
      }
      for (const [pixels, axis] of [[actual.rows, 'row'], [actual.columns, 'column']] as const) {
        for (let y = 0; y < pixels.length; y++) for (let x = 0; x < pixels[y]!.length; x++) {
          const rgb = pixels[y]![x]!;
          const count = actual.counts[axis === 'row' ? 0 : 1]![y]![x]!;
          const selected = axis === 'row' ? y === row : x === column;
          if ((rgb[0]! > rgb[1]!) !== selected) wrongScanlines++;
          color(Math.log1p(count) / Math.log1p(axis === 'row' ? 19 : 17), selected ? 0.65 : 0)
            .forEach((v, i) => { maxColorError = Math.max(maxColorError, Math.abs(rgb[i]! - v)); });
        }
      }
      expect(maxColorError).toBeLessThanOrEqual(1);
      expect(wrongScanlines).toBe(0);
      for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) {
        const sourceY = row! + y - 4, sourceX = column! + x - 4;
        const pixel = actual.neighborhood.slice((y * 9 + x) * 4, (y * 9 + x + 1) * 4);
        expect(pixel).toEqual(sourceY < 0 || sourceY >= 17 || sourceX < 0 || sourceX >= 19 ? [46, 61, 76, 255] : [...expected(sourceY === 0 && sourceX === 0 ? -0 : Math.fround(((sourceY * 19 + sourceX) % 17 - 8) / 32), false, false), 255]);
      }
      // Inspect actual composited enlargement, not just its 9×9 source buffer.
      const screenshot = await page.locator('.magnifier-card canvas').screenshot();
      const enlarged = await page.evaluate(async (base64) => {
        const img = new Image(); img.src = `data:image/png;base64,${base64}`; await img.decode();
        const canvas = document.createElement('canvas'); canvas.width = img.width; canvas.height = img.height;
        const context = canvas.getContext('2d')!; context.drawImage(img, 0, 0);
        const samples = [];
        for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) {
          for (const offset of [3, 15]) samples.push(Array.from(context.getImageData(
            Math.floor((x * 18 + offset) * img.width / 162), Math.floor((y * 18 + offset) * img.height / 162), 1, 1).data));
        }
        return samples;
      }, screenshot.toString('base64'));
      enlarged.forEach((pixel, index) => expect(pixel).toEqual(actual.neighborhood.slice(Math.floor(index / 2) * 4, (Math.floor(index / 2) + 1) * 4)));
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
    expect(await page.evaluate(() => window.explorerFixture.pixels(window.explorerFixture.renderers[0]!).every((row) => row.every((p) => p[1]! > p[0]! && p[1]! > p[2]!)))).toBe(true);
    await page.evaluate(() => window.explorerFixture.unmount());
    expect(await page.evaluate(() => [window.explorerFixture.metrics.live.size, window.explorerFixture.metrics.liveDisplays.size])).toEqual([0, 0]);
  });

  test('card stays inside the scientific pane and clear of distribution data', async ({ page }) => {
    await open(page);
    for (const [row, column] of [[0, 0], [0, 18], [16, 0], [16, 18]]) {
      await hover(page, row!, column!);
      const { card, pane, panels } = await page.evaluate(() => ({
        card: document.querySelector('.matrix-inspection')!.getBoundingClientRect().toJSON(),
        pane: document.querySelector('.matrix-surfaces')!.getBoundingClientRect().toJSON(),
        panels: ['.row-distributions canvas', '.column-distributions canvas'].map((selector) => document.querySelector(selector)!.getBoundingClientRect().toJSON()),
      }));
      expect(card.left).toBeGreaterThanOrEqual(pane.left);
      expect(card.top).toBeGreaterThanOrEqual(pane.top);
      expect(card.right).toBeLessThanOrEqual(pane.right);
      expect(card.bottom).toBeLessThanOrEqual(pane.bottom);
      for (const panel of panels) expect(card.right <= panel.left || card.left >= panel.right || card.bottom <= panel.top || card.top >= panel.bottom).toBe(true);
    }
  });

  test('center border and centered guides have thin independent display geometry', async ({ page }) => {
    await open(page);
    await hover(page, 8, 8);
    const styles = await page.evaluate(() => {
      const read = (selector: string) => {
        const element = document.querySelector(selector)!;
        const style = getComputedStyle(element), rect = element.getBoundingClientRect();
        return { width: rect.width, height: rect.height, x: rect.x + rect.width / 2, y: rect.y + rect.height / 2,
          opacity: Number(style.opacity), border: style.borderTopWidth };
      };
      return { center: read('.magnifier-center'), h: read('.magnifier-guide-horizontal'), v: read('.magnifier-guide-vertical') };
    });
    expect(styles.center.border).toBe('1px');
    expect(styles.h.height).toBe(1); expect(styles.v.width).toBe(1);
    expect(styles.h.width).toBe(162); expect(styles.v.height).toBe(162);
    expect(styles.h.x).toBe(styles.center.x); expect(styles.h.y).toBe(styles.center.y);
    expect(styles.v.x).toBe(styles.center.x); expect(styles.v.y).toBe(styles.center.y);
    expect(styles.h.opacity).toBeLessThan(styles.center.opacity);
    expect(styles.v.opacity).toBeLessThan(styles.center.opacity);
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
      const before = await resources(page);
      const surfaces = await page.evaluate(() => {
        const f = window.explorerFixture;
        return f.renderers.map((r) => ({ view: r.view!, pixels: f.pixels(r) }));
      });
      for (const [index, surface] of surfaces.entries()) {
        for (let y = 0; y < surface.pixels.length; y++) for (let x = 0; x < surface.pixels[y]!.length; x++) {
          const active = (index !== 2 && y + surface.view.y === origin.y) ||
            (index !== 1 && x + surface.view.x === origin.x);
          const rgb = surface.pixels[y]![x]!;
          expect(rgb[0]! > rgb[1]!).toBe(active);
        }
      }
      expect(await resources(page)).toEqual(before);
    }
    await hover(page, 16, 18);
    await page.locator('.matrix-scroll').focus();
    await page.keyboard.press('ArrowLeft');
    await expect(page.locator('.inspection-readout')).toContainText('row 9 · column 10');
    expect(await page.evaluate(() => {
      const f = window.explorerFixture;
      return f.renderers.map((r, index) => {
        const view = r.view!;
        return f.pixels(r).every((scanline, y) => scanline.every((rgb, x) => {
          const active = (index !== 2 && y + view.y === 9) || (index !== 1 && x + view.x === 10);
          return (rgb[0]! > rgb[1]!) === active;
        }));
      });
    })).toEqual([true, true, true]);
    await page.keyboard.press('Escape');
    await expect(page.locator('.inspection-readout')).toContainText('row 9 · column 10');
    await page.getByRole('button', { name: /^B \[/ }).click();
    await expect(page.locator('.matrix-inspection')).toHaveCount(0);
    expect(await page.evaluate(() => window.explorerFixture.metrics.liveDisplays.size)).toBe(0);
  });
});

test('missing/nonfinite values, stream arrival and late transfer update the active inspection', async ({ page }) => {
  await open(page, 10);
  await hover(page, 0, 10);
  await expect(page.locator('.inspection-readout')).toContainText('Unavailable — not received');
  expect(await page.evaluate(() => {
    const f = window.explorerFixture;
    return f.renderers.every((r) => f.pixels(r).every((row) => row.every((rgb) => rgb[0]! < rgb[1]!)));
  })).toBe(true);
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


test('shell typography preserves hit-testing at the final tensor pixel across font stacks', async ({ page }) => {
  await open(page);
  for (const font of ['Arial, sans-serif', 'DejaVu Sans, sans-serif', 'monospace']) {
    await page.locator('.app-shell').evaluate((shell: HTMLElement, font) => { shell.style.fontFamily = font; }, font);
    await page.locator('.matrix-scroll').scrollIntoViewIfNeeded();
  await nativeCamera(page);
    await hover(page, 16, 18);
    expect(await page.evaluate(() => document.documentElement.scrollHeight === innerHeight)).toBe(true);
  }
});
