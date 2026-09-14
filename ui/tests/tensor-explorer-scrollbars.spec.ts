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
      await expect(page.locator('[data-result="tensor"]')).toHaveCount(0);
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

for (const dpr of [1, 2]) test.describe(`workspace panes at DPR ${dpr}`, () => {
  test.use({ deviceScaleFactor: dpr });
  for (const viewport of [{ width: 1440, height: 900 }, { width: 780, height: 640 }, { width: 390, height: 640 }]) {
    test(`independent overflow and aligned profiles at ${viewport.width}×${viewport.height}`, async ({ page }, testInfo) => {
      await page.setViewportSize(viewport);
      await page.goto(`http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/tensor-explorer.html`);
      await expect(page.getByRole('combobox')).toBeEnabled();
      await page.evaluate((dpr) => {
        const tensors = window.explorerFixture.tensors;
        // Exceed every tested viewport with bounded fixtures; geometry checks do
        // not need multi-megapixel allocations far beyond the visible surface.
        const cases = [['fits', 64, 120], ['tall', 800, 120], ['wide', 64, 1100], ['both', 800, 1100]] as const;
        for (const [id, rows, columns] of cases) tensors.push({ ...tensors[0]!, id, name: id, path: [id],
          shape: [rows * dpr, columns * dpr], rank: 2, numel: rows * columns * dpr * dpr });
        for (let i = 0; i < 80; i++) tensors.push({ ...tensors[0]!, id: `inventory-${i}`, name: `inventory-${i}`, path: [`inventory-${i}`] });
      }, dpr);
      await page.getByRole('combobox').selectOption('lab/alpha');
      const inventory = page.getByRole('complementary', { name: 'Tensor inventory' });
      const matrix = page.locator('.matrix-scroll');
      const geometry = () => page.evaluate(() => {
        const rect = (selector: string) => {
          const { top, left, width, height } = document.querySelector(selector)!.getBoundingClientRect();
          return { top, left, width, height };
        };
        const host = document.querySelector<HTMLElement>('.matrix-scroll')!;
        const [m, r, c] = window.explorerFixture.renderers.slice(-3);
        const mr = m!.canvas.getBoundingClientRect(), rr = r!.canvas.getBoundingClientRect(), cr = c!.canvas.getBoundingClientRect();
        return { document: [document.documentElement.scrollWidth, document.documentElement.scrollHeight],
          body: [document.body.scrollWidth, document.body.scrollHeight], windowScroll: [scrollX, scrollY],
          top: rect('.app-bar'), bottom: rect('.app-status-bar'), header: rect('.tensor-header'), pane: rect('.working-surface'),
          inventory: rect('.tensor-layout > aside'), inventoryScroll: document.querySelector('.tensor-layout > aside')!.scrollTop,
          outerOverflow: ['.working-surface', '.surface-content', '.tensor-explorer', '.matrix-surfaces'].map((s) => {
            const e = document.querySelector(s)!;
            return [Math.max(0, e.scrollWidth - e.clientWidth), Math.max(0, e.scrollHeight - e.clientHeight), e.scrollTop, e.scrollLeft];
          }),
          horizontal: host.scrollWidth > host.clientWidth, vertical: host.scrollHeight > host.clientHeight,
          gutter: host.offsetWidth - host.clientWidth, scrollbar: host.offsetHeight - host.clientHeight,
          client: [host.clientWidth, host.clientHeight], scroll: [host.scrollLeft, host.scrollTop],
          states: [m!.state, r!.state, c!.state],
          matrix: m!.view!, row: r!.view!, column: c!.view!,
          alignment: [(mr.top - rr.top) * devicePixelRatio, (mr.left - cr.left) * devicePixelRatio],
          pixels: [mr.width * devicePixelRatio, mr.height * devicePixelRatio],
          canvasInside: mr.top === Math.round((host.getBoundingClientRect().top + host.clientTop) * devicePixelRatio) / devicePixelRatio &&
            mr.bottom <= document.querySelector('.working-surface')!.getBoundingClientRect().bottom &&
            cr.bottom <= document.querySelector('.working-surface')!.getBoundingClientRect().bottom,
        };
      });
      const evidence = [];
      for (const [name, horizontal, vertical] of [['fits', false, false], ['tall', false, true], ['wide', true, false], ['both', true, true]] as const) {
        await page.getByRole('button', { name: new RegExp(`^${name} \\[` ) }).click();
        await expect(matrix).toBeVisible();
        await expect.poll(async () => { const g = await geometry(); return [g.horizontal, g.vertical]; }).toEqual([horizontal, vertical]);
        const initial = await geometry();
        expect(initial.gutter).toBeGreaterThan(0); // Real, non-overlay native scrollbar coverage.
        expect(initial.scrollbar > 0).toBe(horizontal);
        expect(initial.client[0]).toBeGreaterThan(0);
        expect(initial.client[1]).toBeGreaterThan(0);
        // Inventory scrolling must not move either scientific data or chrome.
        await inventory.evaluate((e) => { e.scrollTop = e.scrollHeight; });
        await expect.poll(() => inventory.evaluate((e) => e.scrollTop)).toBeGreaterThan(0);
        const inventoryEnd = await geometry();
        expect(inventoryEnd.scroll).toEqual(initial.scroll);
        expect(inventoryEnd.pane).toEqual(initial.pane);
        expect(inventoryEnd.header).toEqual(initial.header);
        for (const [x, y] of [[1e9, 1e9], [0, 0]]) {
          await matrix.evaluate((e, [x, y]) => { e.scrollLeft = x!; e.scrollTop = y!; }, [x!, y!]);
          await expect.poll(async () => {
            const g = await geometry();
            expect(g.states).toEqual(['ready', 'ready', 'ready']);
            return [g.matrix.x, g.matrix.y, g.column.x, g.row.y];
          }).toEqual(x ? [horizontal ? 1100 * dpr - initial.matrix.width : 0, vertical ? 800 * dpr - initial.matrix.height : 0,
            horizontal ? 1100 * dpr - initial.matrix.width : 0, vertical ? 800 * dpr - initial.matrix.height : 0] : [0, 0, 0, 0]);
          const g = await geometry();
          expect(g.document).toEqual([viewport.width, viewport.height]);
          expect(g.body).toEqual(g.document);
          expect(g.windowScroll).toEqual([0, 0]);
          expect(g.top).toEqual(initial.top);
          expect(g.top.top).toBe(0);
          expect(g.bottom).toEqual(initial.bottom);
          expect(g.bottom.top + g.bottom.height).toBe(viewport.height);
          expect(g.header).toEqual(initial.header);
          expect(g.inventory).toEqual(initial.inventory);
          expect(g.inventoryScroll).toBe(inventoryEnd.inventoryScroll);
          expect(g.outerOverflow).toEqual(Array.from({ length: 4 }, () => [0, 0, 0, 0]));
          expect(g.alignment).toEqual([0, 0]);
          expect(g.row.height).toBe(g.matrix.height);
          expect(g.column.width).toBe(g.matrix.width);
          expect(g.pixels).toEqual([g.matrix.width, g.matrix.height]);
          expect(g.canvasInside).toBe(true);
          evidence.push({ name, ...g });
        }
        // Wheel input at a matrix edge must not chain into another workspace pane.
        await matrix.hover({ position: { x: 8, y: 8 } });
        await page.mouse.wheel(0, -100);
        await expect.poll(async () => (await geometry()).inventoryScroll).toBe(inventoryEnd.inventoryScroll);
      }
      await testInfo.attach('workspace pane geometry', { body: JSON.stringify(evidence), contentType: 'application/json' });
      await testInfo.attach('scroll-contained workspace', { body: await page.screenshot(), contentType: 'image/png' });
    });
  }
});
