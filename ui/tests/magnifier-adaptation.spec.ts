import { expect, test } from '@playwright/test';
import type {} from './matrix-explorer-harness';

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
});

test('visibility history belongs to each Matrix Explorer instance', async ({ page }) => {
  await page.setViewportSize({ width: page.viewportSize()!.width, height: 1100 });
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/matrix-explorer.html`);
  // Leave enough protected free space in both panes to isolate zoom policy.
  await page.addStyleTag({ content: '#workspace { height: 430px; }' });
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
