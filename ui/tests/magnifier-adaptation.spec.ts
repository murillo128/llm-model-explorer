import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import type {} from './matrix-explorer-harness';

async function placement(page: Page) {
  const result = await page.evaluate(() => ({
    card: document.querySelector('.matrix-inspection')!.getBoundingClientRect().toJSON(),
    pane: document.querySelector('.matrix-surfaces')!.getBoundingClientRect().toJSON(),
    panels: [...document.querySelectorAll('.row-distributions, .column-distributions')]
      .map((panel) => panel.getBoundingClientRect().toJSON()),
    width: innerWidth, height: innerHeight,
  }));
  expect(result.card.left).toBeGreaterThanOrEqual(Math.max(0, result.pane.left));
  expect(result.card.top).toBeGreaterThanOrEqual(Math.max(0, result.pane.top));
  expect(result.card.right).toBeLessThanOrEqual(Math.min(result.width, result.pane.right));
  expect(result.card.bottom).toBeLessThanOrEqual(Math.min(result.height, result.pane.bottom));
  for (const panel of result.panels) expect(result.card.right <= panel.left || result.card.left >= panel.right ||
    result.card.bottom <= panel.top || result.card.top >= panel.bottom).toBe(true);
  return result.card;
}

async function hoverFraction(page: Page, x: number, y: number) {
  const point = await page.evaluate(({ x, y }) => {
    const viewport = window.matrixFixture.viewports.findLast((v) => v.renderer.state === 'ready')!;
    const rect = viewport.canvas.getBoundingClientRect();
    const px = Math.max(1, Math.min(rect.width - 1, rect.width * x));
    const py = Math.max(1, Math.min(rect.height - 1, rect.height * y));
    return { x: rect.left + px, y: rect.top + py, cell: viewport.renderer.cellAt(px, py)! };
  }, { x, y });
  await page.mouse.move(point.x, point.y);
  await expect(page.locator('.inspection-readout')).toContainText(`row ${point.cell.row} · column ${point.cell.column}`);
  return placement(page);
}

for (const dpr of [1, 1.25, 2]) test.describe(`adaptive magnifier DPR ${dpr}`, () => {
  test.use({ deviceScaleFactor: dpr });
  test('camera hysteresis preserves keyboard selection, numeric readout and scalar resources', async ({ page }) => {
    await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/matrix-explorer.html`);
    await expect(page.locator('.matrix-scroll')).toBeVisible();
    await page.evaluate(() => {
      const f = window.matrixFixture;
      f.subscriptions[0]!.values(new Float32Array([-0, .1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]), 0);
      f.viewports[0]!.zoomAt(7 * devicePixelRatio, 0, 0);
    });
    await page.locator('.matrix-scroll').focus();
    await page.keyboard.press('ArrowRight');
    await expect(page.locator('.magnifier-card')).toBeVisible();
    const before = await page.evaluate(() => {
      const f = window.matrixFixture;
      return { uploads: f.metrics.uploads, scalars: f.metrics.scalarAllocations, selections: f.cells.length };
    });
    for (const [size, visible] of [[9.99, true], [10, false], [9.99, false], [10.01, false], [8, false], [7.99, true], [8.01, true]] as const) {
      await page.evaluate((size) => window.matrixFixture.viewports[0]!.zoomAt(size * devicePixelRatio, 0, 0), size);
      await expect(page.locator('.magnifier-card')).toHaveCount(visible ? 1 : 0);
      await expect(page.locator('.inspection-readout')).toContainText('row 0 · column 1');
      await expect(page.locator('.inspection-readout')).toContainText(String(Math.fround(.1)));
      await expect(page.getByRole('status', { name: 'Linked annotation' })).toHaveText('Row 0, cell 0:1');
    }
    expect(await page.evaluate(() => {
      const f = window.matrixFixture;
      return { uploads: f.metrics.uploads, scalars: f.metrics.scalarAllocations, selections: f.cells.length };
    })).toEqual(before);
  });

  test('pointer placement follows the center and stays legal at every edge after scroll and zoom', async ({ page }) => {
    await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/matrix-explorer.html`);
    await page.addStyleTag({ content: '#workspace { width: min(1050px, 100%); height: 650px; }' });
    await page.evaluate(() => {
      const f = window.matrixFixture;
      f.sources.square = { ...f.sources.square,
        descriptor: { shape: [700, 900], rank: 2, numel: 630000, logical_dtype: 'float32' } };
      f.render('square');
    });
    await expect(page.locator('.row-distributions')).toBeVisible();
    await page.evaluate(() => window.matrixFixture.viewports.findLast((v) => v.renderer.state === 'ready')!.zoomAt(devicePixelRatio, 0, 0));
    const before = await hoverFraction(page, .3, .3);
    const after = await hoverFraction(page, .33, .33);
    expect([after.left, after.top]).not.toEqual([before.left, before.top]);
    for (const scale of [1, 2]) {
      await page.evaluate((scale) => {
        const view = window.matrixFixture.viewports.findLast((v) => v.renderer.state === 'ready')!;
        view.zoomAt(scale * devicePixelRatio, 0, 0);
        view.host.scrollLeft = 180;
        view.host.scrollTop = 140;
        view.refresh();
      }, scale);
      for (const [x, y] of [[0, 0], [.5, 0], [1, 0], [1, .5], [1, 1], [.5, 1], [0, 1], [0, .5], [.5, .5]]) {
        await hoverFraction(page, x!, y!);
      }
    }
    // Leave the pointer stationary while native scroll and camera scale change.
    const point = await page.evaluate(() => {
      const view = window.matrixFixture.viewports.findLast((v) => v.renderer.state === 'ready')!;
      const rect = view.canvas.getBoundingClientRect();
      return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
    });
    for (const change of ['scroll', 'zoom'] as const) {
      const cell = await page.evaluate(({ point, change }) => {
        const view = window.matrixFixture.viewports.findLast((v) => v.renderer.state === 'ready')!;
        if (change === 'scroll') { view.host.scrollTop += 25; view.refresh(); }
        else view.zoomAt(3 * devicePixelRatio, 0, 0);
        const rect = view.canvas.getBoundingClientRect();
        return view.renderer.cellAt(point.x - rect.left, point.y - rect.top)!;
      }, { point, change });
      await expect(page.locator('.inspection-readout')).toContainText(`row ${cell.row} · column ${cell.column}`);
      await placement(page);
    }
  });
});

test('focused inspection follows dimension-preserving layout shifts, resize and DPR changes', async ({ page, context }) => {
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/matrix-explorer.html`);
  await expect(page.locator('.matrix-scroll')).toBeVisible();
  await page.evaluate(() => window.matrixFixture.viewports[0]!.zoomAt(7 * devicePixelRatio, 0, 0));
  await page.locator('.matrix-scroll').focus();
  await expect(page.locator('.magnifier-card')).toBeVisible();
  const before = await placement(page);
  await page.addStyleTag({ content: '#workspace { transform: translateY(30px); }' });
  await expect.poll(async () => (await page.locator('.matrix-inspection').boundingBox())!.y).toBeCloseTo(before.top + 30, 2);
  await expect(page.locator('.inspection-readout')).toContainText('row 0 · column 0');
  const original = page.viewportSize()!;
  await page.setViewportSize({ width: original.width - 30, height: original.height });
  await placement(page);
  const cdp = await context.newCDPSession(page);
  await cdp.send('Emulation.setDeviceMetricsOverride', { width: original.width - 31, height: original.height, deviceScaleFactor: 1.25, mobile: false });
  await expect.poll(() => page.evaluate(() => window.matrixFixture.viewports[0]!.renderer.view!.dpr)).toBe(1.25);
  await expect(page.locator('.inspection-readout')).toContainText('row 0 · column 0');
  await placement(page);
  await page.evaluate(() => window.matrixFixture.render('B'));
  await expect(page.locator('.matrix-inspection')).toHaveCount(0);
});

test('a short pane retains only its complete compact readout', async ({ page }) => {
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/matrix-explorer.html`);
  await page.addStyleTag({ content: '#workspace { height: 130px; }' });
  await page.evaluate(() => window.matrixFixture.viewports[0]!.zoomAt(devicePixelRatio, 0, 0));
  await page.locator('.matrix-scroll').focus();
  await expect(page.locator('.inspection-readout')).toContainText('row 0 · column 0');
  await expect(page.locator('.magnifier-card')).toHaveCount(0);
  await placement(page);
});

test('visibility history belongs to each Matrix Explorer instance', async ({ page }) => {
  await page.setViewportSize({ width: page.viewportSize()!.width, height: 1500 });
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/matrix-explorer.html`);
  // Leave enough protected free space in both panes to isolate zoom policy.
  await page.addStyleTag({ content: '#workspace { height: 600px; }' });
  await page.evaluate(() => window.matrixFixture.renderPair());
  await expect(page.locator('.matrix-scroll')).toHaveCount(2);
  await page.evaluate(() => {
    const f = window.matrixFixture;
    const views = f.viewports.filter((v) => v.renderer.state === 'ready');
    for (const view of views) view.zoomAt(7 * devicePixelRatio, 0, 0);
    views[0]!.zoomAt(10 * devicePixelRatio, 0, 0);
    for (const view of views) view.zoomAt(9 * devicePixelRatio, 0, 0);
  });
  await page.locator('.matrix-scroll').nth(1).focus();
  await expect(page.locator('.magnifier-card')).toBeVisible();
  await page.locator('.matrix-scroll').nth(0).focus();
  await expect(page.locator('.magnifier-card')).toHaveCount(0);
  await expect(page.locator('.inspection-readout')).toContainText('Unavailable — not received');
});
