import { expect, test } from '@playwright/test';
import { color } from './scalar-oracle';
import type {} from './matrix-explorer-harness';

for (const dpr of [1, 2]) test(`zoomed profile pixels track exact matrix rows and columns at DPR ${dpr}`, async ({ page }, testInfo) => {
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setDeviceMetricsOverride', { width: page.viewportSize()!.width, height: page.viewportSize()!.height, deviceScaleFactor: dpr, mobile: false });
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/matrix-explorer.html`);
  await page.addStyleTag({ content: '#workspace { height: 600px; max-height: 70vh; }' });
  await expect(page.locator('.matrix-scroll')).toBeVisible();
  await page.evaluate(() => window.matrixFixture.render('square'));
  await expect(page.locator('.matrix-surfaces canvas')).toHaveCount(3);
  const result = await page.evaluate(() => {
    const f = window.matrixFixture, v = f.viewports.at(-1)!;
    const updates = f.subscriptions.at(-1)!;
    updates.distribution('rows', Uint32Array.from({ length: 12000 }, (_, i) => i % 5), 0);
    updates.distribution('columns', Uint32Array.from({ length: 10000 }, (_, i) => i % 3), 0);
    v.zoomAt(4.9, 0, 0); v.host.scrollLeft = 17; v.host.scrollTop = 31; v.refresh();
    const profiles = f.renderers.filter(r => r.state === 'ready' && r !== v.renderer);
    return profiles.map((r, axis) => {
      r.draw();
      const gl = r.canvas.getContext('webgl2')!;
      const bytes = new Uint8Array(r.canvas.width * r.canvas.height * 4);
      gl.readPixels(0, 0, r.canvas.width, r.canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
      const samples = [];
      const length = axis === 0 ? r.canvas.height : r.canvas.width;
      for (let p = 0; p < length; p++) {
        const x = axis === 0 ? 27 : p, y = axis === 0 ? p : 27;
        const cell = r.cellAt((x + .25) / devicePixelRatio, (y + .25) / devicePixelRatio)!;
        const matrixCell = v.renderer.cellAt((axis === 0 ? .25 : p + .25) / devicePixelRatio,
          (axis === 0 ? p + .25 : .25) / devicePixelRatio)!;
        const offset = ((r.canvas.height - y - 1) * r.canvas.width + x) * 4;
        samples.push({ cell, matrixCell, pixel: Array.from(bytes.subarray(offset, offset + 4)) });
      }
      return { axis, view: r.view!, samples };
    });
  });
  for (const { axis, samples } of result) for (const { cell, matrixCell, pixel } of samples) {
    expect(axis === 0 ? cell.row : cell.column).toBe(axis === 0 ? matrixCell.row : matrixCell.column);
    const count = (cell.row * 100 + cell.column) % (axis === 0 ? 5 : 3);
    expect(pixel).toEqual([...color(Math.log1p(count) / Math.log1p(axis === 0 ? 100 : 120)), 255]);
  }
  await testInfo.attach('aligned zoom geometry', { body: JSON.stringify(result.map(({ axis, view }) => ({ axis, view }))), contentType: 'application/json' });
});
