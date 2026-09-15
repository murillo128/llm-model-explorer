import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import type {} from './matrix-explorer-harness';

const url = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/matrix-explorer.html`;
const geometry = (page: Page) => page.evaluate(() => {
  const v = window.matrixFixture.viewports.at(-1)!;
  const rect = (node: Element) => node.getBoundingClientRect().toJSON();
  return { matrix: rect(v.canvas), host: rect(v.host), pane: rect(document.querySelector('.matrix-surfaces')!),
    rowTrack: rect(document.querySelector('.row-distributions')!), columnTrack: rect(document.querySelector('.column-distributions')!),
    row: rect(document.querySelector('.row-distributions canvas')!), column: rect(document.querySelector('.column-distributions canvas')!),
    clientWidth: v.host.clientWidth, clientHeight: v.host.clientHeight,
    horizontal: v.host.scrollWidth > v.host.clientWidth, vertical: v.host.scrollHeight > v.host.clientHeight,
    view: v.renderer.view!, resources: window.matrixFixture.resources(), uploads: window.matrixFixture.metrics.uploads };
});

for (const dpr of [1, 1.25, 2]) test.describe(`centered scientific content DPR ${dpr}`, () => {
  test.use({ deviceScaleFactor: dpr });
  for (const name of ['short', 'tall', 'full', 'square'] as const) test(`${name}: independent centering, fixed profile tracks, resize and source reset`, async ({ page }) => {
    await page.goto(url);
    await expect(page.locator('.matrix-scroll')).toBeVisible();
    await page.addStyleTag({ content: '#workspace { height: min(600px, 70vh); }' });
    await page.evaluate(name => window.matrixFixture.render(name), name);
    await expect(page.locator('.matrix-surfaces canvas')).toHaveCount(3);
    await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.zoomAt(1, 0, 0));
    const before = await geometry(page);
    for (const size of [page.viewportSize()!, { width: 780, height: 640 }]) {
      await page.setViewportSize(size);
      await expect.poll(async () => {
        const g = await geometry(page);
        const offsetX = Math.floor((Math.floor(g.clientWidth * dpr) - g.view.width) / 2) / dpr;
        const offsetY = Math.floor((Math.floor(g.clientHeight * dpr) - g.view.height) / 2) / dpr;
        return Math.max(Math.abs(g.matrix.left - (Math.round(g.host.left * dpr) / dpr + offsetX)),
          Math.abs(g.matrix.top - (Math.round(g.host.top * dpr) / dpr + offsetY)),
          // Host CSS bounds change before ResizeObserver resolves the available
          // scientific height. Wait for that complete layout, not an old camera
          // still centered within its old client box. Integer client dimensions
          // can exceed fractional CSS pane bounds by less than a device pixel;
          // use the same containment allowance as the final assertions below.
          g.column.bottom - g.pane.bottom - 1 / dpr, g.row.right - g.pane.right - 1 / dpr);
      }).toBeLessThan(0.03);
      const g = await geometry(page);
      expect(g.view.scaleX).toBe(1); expect(g.view.scaleY).toBe(1);
      expect(g.rowTrack.top).toBeCloseTo(g.host.top, 3);
      expect(g.rowTrack.height).toBe(g.clientHeight);
      expect(g.columnTrack.left).toBeCloseTo(g.host.left, 3);
      expect(g.columnTrack.width).toBe(g.clientWidth);
      expect(g.row.top).toBeCloseTo(g.matrix.top, 3);
      expect(g.row.height).toBeCloseTo(g.matrix.height, 3);
      expect(g.column.left).toBeCloseTo(g.matrix.left, 3);
      expect(g.column.width).toBeCloseTo(g.matrix.width, 3);
      expect(Math.abs(g.row.left - (g.host.left + g.clientWidth) - 10)).toBeLessThan(1 / dpr);
      expect(Math.abs(g.column.top - (g.host.top + g.clientHeight) - 10)).toBeLessThan(1 / dpr);
      expect(g.vertical).toBe(g.view.scrollHeight > g.clientHeight);
      expect(g.horizontal).toBe(false);
      expect(g.row.right).toBeLessThanOrEqual(g.pane.right + 1 / dpr);
      expect(g.column.bottom).toBeLessThanOrEqual(g.pane.bottom + 1 / dpr);
      expect(g.resources).toEqual(before.resources); expect(g.uploads).toBe(before.uploads);
    }
    // A DPR/layout change keeps origins and logical values independent of offsets.
    await page.evaluate(() => {
      Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: 2.5 });
      window.dispatchEvent(new Event('resize'));
    });
    await expect.poll(async () => (await geometry(page)).view.dpr).toBe(2.5);
    const changed = await geometry(page);
    expect(changed.row.top).toBeCloseTo(changed.matrix.top, 2);
    expect(changed.column.left).toBeCloseTo(changed.matrix.left, 2);
    await page.evaluate(() => window.matrixFixture.render('large'));
    await expect(page.locator('.matrix-surfaces canvas')).toHaveCount(1);
    expect(await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.renderer.view!.x)).toBe(0);
  });

  test('focal cell stays at its screen point when centered data begins overflowing', async ({ page }) => {
    await page.goto(url);
    await expect(page.locator('.matrix-scroll')).toBeVisible();
    await page.addStyleTag({ content: '#workspace { height: min(600px, 70vh); }' });
    await page.evaluate(() => window.matrixFixture.render('square'));
    await expect(page.locator('.matrix-surfaces canvas')).toHaveCount(3);
    const result = await page.evaluate(() => {
      const v = window.matrixFixture.viewports.at(-1)!;
      v.zoomAt(1, 0, 0);
      const rect = v.canvas.getBoundingClientRect();
      const x = rect.left + 50 / devicePixelRatio, y = rect.top + 60 / devicePixelRatio;
      v.zoomAt(20, x - rect.left, y - rect.top);
      const after = v.canvas.getBoundingClientRect(), view = v.renderer.view!;
      return { column: view.x + (x - after.left) * view.dpr / view.scaleX,
        row: view.y + (y - after.top) * view.dpr / view.scaleY, scale: view.scaleX };
    });
    expect(Math.abs(result.column - 50)).toBeLessThan(2 * dpr / result.scale);
    expect(Math.abs(result.row - 60)).toBeLessThan(2 * dpr / result.scale);
  });
});
