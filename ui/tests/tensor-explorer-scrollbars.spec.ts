import { expect, test } from '@playwright/test';

// This project runs headed under Xvfb in CI. The positive scrollbar-thickness
// assertion prevents overlay/headless scrollbars from silently masking the bug.
for (const dpr of [1, 2]) test.describe(`native scrollbars at DPR ${dpr}`, () => {
  test.use({ deviceScaleFactor: dpr });
  for (const [name, rows] of [['wide-vector', 1], ['short-matrix', 2]] as const) {
    test(`${name} retains visible exact data and reaches the final column`, async ({ page }, testInfo) => {
      await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html`);
      await expect(page.getByRole('combobox')).toBeEnabled();
      await page.getByRole('combobox').selectOption('lab/alpha');
      await page.getByRole('button', { name: new RegExp(`^${name} \\[` ) }).click();
      await page.evaluate((rows) => {
        const f = window.explorerFixture;
        f.emit(0, 1, f.metadata(0));
        const values = Array<number>(rows * 1536).fill(0);
        values[0] = -2; values[values.length - 1] = 2;
        f.data(0, values); f.end(0);
      }, rows);
      await expect(page.locator('[data-result="tensor"]')).toHaveAttribute('data-state', 'complete');
      const geometry = () => page.evaluate(() => {
        const host = document.querySelector<HTMLElement>('.matrix-scroll')!;
        const matrix = window.explorerFixture.renderers[0]!;
        return { scrollbar: host.offsetHeight - host.clientHeight, clientHeight: host.clientHeight,
          height: matrix.view!.height, canvasHeight: matrix.canvas.getBoundingClientRect().height * devicePixelRatio,
          width: matrix.view!.width, x: matrix.view!.x, prefix: matrix.populatedPrefix };
      });
      await expect.poll(async () => (await geometry()).height).toBe(rows);
      const first = await geometry();
      expect(first.scrollbar).toBeGreaterThan(0);
      expect(first.clientHeight).toBeGreaterThan(0);
      expect(first.canvasHeight).toBe(rows);
      expect(first.width).toBeLessThan(1536);
      expect(first.prefix).toBe(rows * 1536);
      expect(await page.evaluate(() => {
        const f = window.explorerFixture;
        return f.pixels(f.renderers[0]!)[0]![0];
      })).toEqual([0, 0, 0, 255]);
      await page.locator('.matrix-scroll').evaluate((host) => { host.scrollLeft = 1e9; });
      await expect.poll(async () => { const view = await geometry(); return view.x + view.width; }).toBe(1536);
      expect((await geometry()).height).toBe(rows);
      const last = await page.evaluate((rows) => {
        const f = window.explorerFixture;
        const matrix = f.renderers[0]!;
        return { cell: matrix.readCell(rows - 1, 1535), pixel: f.pixels(matrix)[rows - 1]!.at(-1),
          profiles: f.renderers.slice(1).map((r) => r.view), requests: f.requests.map((r) => r.kind) };
      }, rows);
      expect(last.cell).toMatchObject({ value: 2 });
      expect(last.pixel).toEqual([255, 255, 255, 255]);
      if (rows === 1) expect(last.requests).toEqual(['data', 'statistics']);
      else {
        expect(last.profiles[0]!.height).toBe(rows);
        expect(last.profiles[1]!.x).toBe((await geometry()).x);
      }
      // Removing and restoring overflow must keep the strip visible without
      // retaining scrollbar space or reallocating its data representation.
      await page.setViewportSize({ width: 2200, height: 844 });
      await expect.poll(async () => (await geometry()).width).toBe(1536);
      expect((await geometry()).scrollbar).toBe(0);
      expect((await geometry()).height).toBe(rows);
      await page.setViewportSize({ width: 390, height: 844 });
      await expect.poll(async () => (await geometry()).scrollbar).toBeGreaterThan(0);
      expect((await geometry()).height).toBe(rows);
      expect(await page.evaluate(() => window.explorerFixture.metrics.scalarAllocations)).toBe(1);
      await testInfo.attach('data and scrollbar geometry', {
        body: JSON.stringify({ initial: first, afterResize: await geometry() }), contentType: 'application/json',
      });
      await page.locator('.tensor-explorer').scrollIntoViewIfNeeded();
      await testInfo.attach('native scrollbar and exact data strip', {
        body: await page.locator('.tensor-explorer').screenshot(), contentType: 'image/png',
      });
    });
  }
});
