import { expect, test } from '@playwright/test';
import { color } from './scalar-oracle';
import type {} from './matrix-explorer-harness';

for (const dpr of [1, 2]) test(`zoomed profile pixels track exact matrix rows and columns at DPR ${dpr}`, async ({ page }, testInfo) => {
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setDeviceMetricsOverride', { width: page.viewportSize()!.width, height: page.viewportSize()!.height, deviceScaleFactor: dpr, mobile: false });
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/matrix-explorer.html`);
  await page.addStyleTag({ content: '#workspace { --fixture-viewer-height: min(600px, 70vh); }' });
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

for (const dpr of [1, 2]) test(`fractional selection stays aligned with profile guides at DPR ${dpr}`, async ({ page }) => {
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setDeviceMetricsOverride', { width: page.viewportSize()!.width, height: page.viewportSize()!.height, deviceScaleFactor: dpr, mobile: false });
  await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/matrix-explorer.html`);
  await page.addStyleTag({ content: '#workspace { --fixture-viewer-height: min(600px, 70vh); } .matrix-scroll { max-width: 40px; max-height: 40px; }' });
  await expect(page.locator('.matrix-scroll')).toBeVisible();
  await page.evaluate(() => window.matrixFixture.render('square'));
  await expect(page.locator('.matrix-surfaces canvas')).toHaveCount(3);
  const results = await page.evaluate(() => {
    const f = window.matrixFixture, v = f.viewports.at(-1)!;
    const updates = f.subscriptions.at(-1)!;
    updates.values(new Float32Array(12000), 0);
    updates.distribution('rows', new Uint32Array(12000).fill(1), 0);
    updates.distribution('columns', new Uint32Array(10000).fill(1), 0);
    const profiles = f.renderers.filter(r => r.state === 'ready' && r !== v.renderer);
    const before = { uploads: f.metrics.uploads, scalar: f.metrics.scalarAllocations, integer: f.metrics.integerAllocations };
    const output = [];
    for (const scale of [1.25, 1.99, 3.25, 4.9]) for (const scroll of [0, 1, 7]) {
      v.zoomAt(scale, 0, 0); v.host.scrollLeft = scroll; v.host.scrollTop = scroll; v.refresh();
      const canvas = v.renderer.canvas, rect = canvas.getBoundingClientRect();
      const selected = v.renderer.cellAt(2.25 / devicePixelRatio, 2.25 / devicePixelRatio)!;
      canvas.dispatchEvent(new PointerEvent('pointermove', { clientX: rect.left + 2.25 / devicePixelRatio, clientY: rect.top + 2.25 / devicePixelRatio }));
      const surfaces = [v.renderer, ...profiles].map((r) => {
        r.draw();
        const gl = r.canvas.getContext('webgl2')!;
        const bytes = new Uint8Array(r.canvas.width * r.canvas.height * 4);
        gl.readPixels(0, 0, r.canvas.width, r.canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
        const guides = [];
        for (let y = 0; y < r.canvas.height; y++) for (let x = 0; x < r.canvas.width; x++) {
          const offset = ((r.canvas.height - y - 1) * r.canvas.width + x) * 4;
          if (bytes[offset]! > bytes[offset + 1]!) guides.push({ x, y, cell: r.cellAt((x + .25) / devicePixelRatio, (y + .25) / devicePixelRatio)! });
        }
        return { guides, width: r.canvas.width, height: r.canvas.height };
      });
      output.push({ scale, scroll, selected, surfaces });
    }
    return { output, before, after: { uploads: f.metrics.uploads, scalar: f.metrics.scalarAllocations, integer: f.metrics.integerAllocations } };
  });
  for (const { surfaces, selected } of results.output) {
    const [matrix, rows, columns] = surfaces;
    expect(rows!.guides.length).toBe(rows!.width);
    expect(columns!.guides.length).toBe(columns!.height);
    const rowPixels = new Set(rows!.guides.map(p => p.y));
    const columnPixels = new Set(columns!.guides.map(p => p.x));
    expect(rowPixels.size).toBe(1); expect(columnPixels.size).toBe(1);
    rows!.guides.forEach(p => expect(p.cell.row).toBe(selected.row));
    columns!.guides.forEach(p => expect(p.cell.column).toBe(selected.column));
    expect(matrix!.guides.length).toBe(matrix!.width + matrix!.height - 1);
    for (const p of matrix!.guides) {
      expect(rowPixels.has(p.y) || columnPixels.has(p.x)).toBe(true);
      if (rowPixels.has(p.y)) expect(p.cell.row).toBe(selected.row);
      if (columnPixels.has(p.x)) expect(p.cell.column).toBe(selected.column);
    }
  }
  expect(results.after).toEqual(results.before);
});
