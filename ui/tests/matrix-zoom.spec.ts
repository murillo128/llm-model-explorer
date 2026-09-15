import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import type {} from './matrix-explorer-harness';

const url = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/matrix-explorer.html`;
async function open(page: Page, name: 'short' | 'tall' | 'large' | 'square') {
  await page.goto(url);
  await page.addStyleTag({ content: '#workspace { height: min(600px, 70vh); }' });
  await expect(page.locator('.matrix-scroll')).toBeVisible();
  await page.evaluate((name) => window.matrixFixture.render(name), name);
  await expect.poll(() => page.evaluate(() => window.matrixFixture.subscriptions.length)).toBe(2);
}
async function camera(page: Page) {
  return page.evaluate(() => {
    const f = window.matrixFixture, v = f.viewports.at(-1)!, r = v.renderer;
    const row = f.renderers.find((r) => r.state === 'ready' && r.geometry.columns === 100 && r !== v.renderer);
    const column = f.renderers.find((r) => r.state === 'ready' && r.geometry.rows === 100 && r !== v.renderer);
    return { view: r.view!, rows: row?.view, columns: column?.view,
      width: v.host.clientWidth, height: v.host.clientHeight, shape: r.geometry,
      rect: r.canvas.getBoundingClientRect().toJSON(),
      rowRect: row?.canvas.getBoundingClientRect().toJSON(), columnRect: column?.canvas.getBoundingClientRect().toJSON(),
      host: v.host.getBoundingClientRect().toJSON(), resources: f.resources(), uploads: f.metrics.uploads };
  });
}
for (const dpr of [1, 1.25, 2]) test.describe(`zoom DPR ${dpr}`, () => {
  test.use({ deviceScaleFactor: dpr });
  for (const shape of ['short', 'tall', 'large', 'square'] as const) test(`fit, focal zoom, bounds and aligned profiles: ${shape}`, async ({ page }) => {
    await open(page, shape);
    let c = await camera(page);
    const fit = Math.max(1, Math.floor(c.width * dpr) / c.shape.columns);
    expect(c.view.scaleX).toBeCloseTo(fit);
    expect(c.view.scaleY).toBe(c.view.scaleX);
    if (shape !== 'large') expect(c.rect.width).toBeCloseTo(Math.floor(c.width * dpr) / dpr, 3);
    const before = await page.evaluate(() => {
      const v = window.matrixFixture.viewports.at(-1)!;
      v.zoomAt(v.renderer.view!.scaleX * 3, 0, 0);
      v.host.scrollLeft = 40; v.host.scrollTop = 30; v.refresh();
      const a = v.renderer.view!;
      v.zoomAt(a.scaleX * 1.5, 31, 27);
      const b = v.renderer.view!;
      return { a, b };
    });
    // Native CSS scroll rounding can move at most one CSS pixel, plus device snapping.
    expect(Math.abs((before.a.x + 31 * dpr / before.a.scaleX) - (before.b.x + 31 * dpr / before.b.scaleX)))
      .toBeLessThanOrEqual(2 * dpr / before.b.scaleX);
    const wantedY = before.a.y + 27 * dpr / before.a.scaleY - 27 * dpr / before.b.scaleY;
    const boundedY = Math.max(0, Math.min(c.shape.rows - before.b.height / before.b.scaleY, wantedY));
    expect(Math.abs(before.b.y - boundedY)).toBeLessThanOrEqual(2 * dpr / before.b.scaleY);
    c = await camera(page);
    if (c.rows && c.columns) {
      expect(c.rows.y).toBeCloseTo(c.view.y); expect(c.columns.x).toBeCloseTo(c.view.x);
      expect(c.rows.scaleY).toBe(c.view.scaleY); expect(c.columns.scaleX).toBe(c.view.scaleX);
      expect(c.rows.width).toBe(100); expect(c.columns.height).toBe(100);
      expect(c.rowRect!.top).toBeCloseTo(c.rect.top, 3); expect(c.columnRect!.left).toBeCloseTo(c.rect.left, 3);
      expect(c.rowRect!.left - c.rect.right).toBeCloseTo(10, 0);
      expect(c.columnRect!.top - c.rect.bottom).toBeCloseTo(10, 0);
    }
    await page.evaluate(() => {
      const v = window.matrixFixture.viewports.at(-1)!;
      v.host.scrollLeft = 1e8; v.host.scrollTop = 1e8; v.refresh();
    });
    c = await camera(page);
    expect(Math.abs(c.shape.columns - c.view.x - c.view.width / c.view.scaleX)).toBeLessThan(2 * dpr / c.view.scaleX);
    if (c.shape.rows * c.view.scaleY > c.view.height) expect(Math.abs(c.shape.rows - c.view.y - c.view.height / c.view.scaleY)).toBeLessThan(2 * dpr / c.view.scaleY);
    await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.zoomAt(.01, 0, 0));
    expect((await camera(page)).view.scaleX).toBe(1);
    await page.getByRole('button', { name: 'Fit width', exact: true }).click();
    c = await camera(page);
    expect(c.view.scaleX).toBeCloseTo(fit); expect(c.view.x).toBe(0); expect(c.view.y).toBe(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight)).toBe(true);
  });
  test('wheel, trackpad pinch, exact click/pending values and resource lifetime', async ({ page }) => {
    await open(page, 'square');
    await page.evaluate(() => window.matrixFixture.subscriptions.at(-1)!.values(Float32Array.from({ length: 1000 }, (_, i) => i + .25), 0));
    const before = await camera(page);
    const point = { x: before.rect.left + 30, y: before.rect.top + 20 };
    await page.mouse.move(point.x, point.y);
    await page.mouse.wheel(0, -180);
    await expect.poll(async () => (await camera(page)).view.scaleX).toBeGreaterThan(before.view.scaleX);
    await page.locator('.matrix-scroll').dispatchEvent('wheel', { deltaY: -100, ctrlKey: true, clientX: point.x, clientY: point.y });
    const clicked = await page.evaluate(() => {
      const v = window.matrixFixture.viewports.at(-1)!, view = v.renderer.view!, rect = v.canvas.getBoundingClientRect();
      const column = Math.ceil(view.x) + 2, row = Math.ceil(view.y) + 1;
      return { column, row, x: rect.left + (column - view.x + .5) * view.scaleX / view.dpr,
        y: rect.top + (row - view.y + .5) * view.scaleY / view.dpr };
    });
    await page.mouse.click(clicked.x, clicked.y);
    await expect(page.locator('.inspection-readout')).toContainText(`row ${clicked.row} · column ${clicked.column}`);
    await expect(page.locator('.inspection-readout')).toContainText(String(clicked.row * 100 + clicked.column + .25));
    await page.evaluate(() => {
      const v = window.matrixFixture.viewports.at(-1)!;
      v.host.scrollTop = 40 * v.renderer.view!.scaleY / devicePixelRatio; v.refresh();
    });
    await page.mouse.move(point.x + 1, point.y + 1);
    await expect(page.locator('.inspection-readout')).toContainText('Unavailable — not received');
    await page.evaluate(() => {
      const f = window.matrixFixture, v = f.viewports.at(-1)!;
      for (let i = 0; i < 12; i++) {
        Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: i % 2 ? 2 : 1.25 });
        window.dispatchEvent(new Event('resize'));
        v.zoomAt(4 + i, 20, 20);
      }
    });
    const after = await camera(page);
    expect(after.uploads).toBe(before.uploads);
    expect(after.resources.textures).toBe(before.resources.textures);
    expect(after.resources.cpuBytes).toBe(before.resources.cpuBytes);
    await page.evaluate(() => window.matrixFixture.render(null));
    await expect(page.locator('.matrix-explorer')).toHaveCount(0);
    expect(await page.evaluate(() => window.matrixFixture.resources())).toEqual({ textures: 0, displays: 0, framebuffers: 0, buffers: 0, programs: 0, cpuBytes: 0 });
  });
  test('region/range camera history, scoped right-click, fit and source reset', async ({ page }) => {
    await open(page, 'square');
    const initial = (await camera(page)).view;
    await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.zoomToBounds({ columns: [4, 16], rows: [5, 13] }));
    const first = (await camera(page)).view;
    await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.zoomToBounds({ rows: [7, 9] }));
    await page.locator('.matrix-scroll').focus();
    await page.keyboard.press('Escape');
    expect((await camera(page)).view).toEqual(first);
    const handled = await page.locator('.row-distributions canvas').evaluate((canvas) => {
      const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
      canvas.dispatchEvent(event);
      return event.defaultPrevented;
    });
    expect(handled).toBe(true);
    expect((await camera(page)).view).toEqual(initial);
    await page.keyboard.press('Escape');
    expect((await camera(page)).view).toEqual(initial);
    const outside = await page.locator('.matrix-scroll').evaluate((host) => {
      const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
      host.dispatchEvent(event);
      return event.defaultPrevented;
    });
    expect(outside).toBe(false);
    await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.zoomToBounds({ columns: [10, 15] }));
    await page.getByRole('button', { name: 'Fit width', exact: true }).click();
    await page.locator('.matrix-scroll').focus();
    await page.keyboard.press('Escape');
    expect((await camera(page)).view).toEqual(initial);
    await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.zoomToBounds({ rows: [10, 15] }));
    await page.evaluate(() => window.matrixFixture.render('tall'));
    await expect.poll(() => page.evaluate(() => window.matrixFixture.viewports.at(-1)!.renderer.geometry.rows)).toBe(900);
    const replaced = (await camera(page)).view;
    await page.locator('.matrix-scroll').focus();
    await page.keyboard.press('Escape');
    expect((await camera(page)).view).toEqual(replaced);
  });
  test('wheel and trackpad pinch coalesce into separate gesture-level camera states', async ({ page }) => {
    await open(page, 'square');
    const initial = (await camera(page)).view;
    async function wheel(ctrlKey: boolean) {
      await page.locator('.matrix-scroll').evaluate((host, ctrlKey) => {
        const rect = host.querySelector('canvas')!.getBoundingClientRect();
        for (const deltaY of [-60, -50, -40]) host.dispatchEvent(new WheelEvent('wheel', {
          deltaY, ctrlKey, clientX: rect.left + 30, clientY: rect.top + 20, bubbles: true, cancelable: true,
        }));
      }, ctrlKey);
    }
    await wheel(false);
    const first = (await camera(page)).view;
    expect(first.scaleX).toBeGreaterThan(initial.scaleX);
    await page.waitForTimeout(220);
    await wheel(true);
    expect((await camera(page)).view.scaleX).toBeGreaterThan(first.scaleX);
    await page.locator('.matrix-scroll').focus();
    await page.keyboard.press('Escape');
    expect((await camera(page)).view).toEqual(first);
    await page.keyboard.press('Escape');
    expect((await camera(page)).view).toEqual(initial);
    await page.keyboard.press('Escape');
    expect((await camera(page)).view).toEqual(initial);
  });
});

test('touch pinch and two simultaneous cameras remain local', async ({ page }) => {
  await page.goto(url);
  await expect(page.locator('.matrix-scroll')).toBeVisible();
  await page.evaluate(() => window.matrixFixture.renderPair());
  await expect(page.locator('.matrix-scroll')).toHaveCount(2);
  await page.addStyleTag({ content: '#workspace { height: 380px; }' });
  const other = await camera(page);
  const first = page.locator('.matrix-scroll').first();
  const rect = (await first.boundingBox())!;
  const x = rect.x + 100, y = rect.y + 20;
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true });
  const initial = await page.evaluate(() => window.matrixFixture.viewports.find(v => v.renderer.state === 'ready')!.renderer.view!);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: x - 20, y }, { x: x + 20, y }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: x - 40, y }, { x: x + 40, y }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: x - 50, y }, { x: x + 50, y }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await expect.poll(() => page.evaluate(() => window.matrixFixture.viewports.find(v => v.renderer.state === 'ready')!.renderer.view!.scaleX)).toBeGreaterThan(initial.scaleX);
  expect((await camera(page)).view).toEqual(other.view);
  await first.focus();
  await page.keyboard.press('Escape');
  expect(await page.evaluate(() => window.matrixFixture.viewports.find(v => v.renderer.state === 'ready')!.renderer.view!)).toEqual(initial);
  await page.keyboard.press('Escape');
  expect(await page.evaluate(() => window.matrixFixture.viewports.find(v => v.renderer.state === 'ready')!.renderer.view!)).toEqual(initial);
  expect((await camera(page)).view).toEqual(other.view);
  expect(await page.evaluate(() => [scrollX, scrollY, visualViewport!.scale])).toEqual([0, 0, 1]);
});

test('DPR changes preserve logical origins without reuploading', async ({ page }) => {
  await open(page, 'tall');
  const result = await page.evaluate(() => {
    const v = window.matrixFixture.viewports.at(-1)!;
    v.zoomAt(8, 0, 0); v.host.scrollTop = 120; v.refresh();
    const before = v.renderer.view!;
    Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: 2 });
    window.dispatchEvent(new Event('resize'));
    return { before, after: v.renderer.view! };
  });
  expect(result.after.x).toBe(result.before.x);
  expect(result.after.y).toBe(result.before.y);
  expect(result.after.scaleX).toBe(result.after.scaleY);
});

test('zoom-back restores stored logical camera after resize/DPR and respects focused controls', async ({ page }) => {
  await open(page, 'tall');
  const before = await page.evaluate(() => {
    const v = window.matrixFixture.viewports.at(-1)!;
    v.zoomAt(8, 0, 0); v.host.scrollTop = 120; v.refresh();
    const before = v.renderer.view!;
    v.zoomToBounds({ rows: [20, 24] });
    return before;
  });
  const zoomed = (await camera(page)).view;
  await page.evaluate(() => {
    const control = document.createElement('input');
    control.setAttribute('aria-label', 'Unrelated input');
    document.body.append(control); control.focus();
  });
  await page.keyboard.press('Escape');
  expect((await camera(page)).view).toEqual(zoomed);
  await page.evaluate(() => {
    Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: 2 });
    document.getElementById('workspace')!.style.height = '360px';
    window.dispatchEvent(new Event('resize'));
  });
  await page.locator('.matrix-scroll').focus();
  await page.keyboard.press('Escape');
  const after = (await camera(page)).view;
  expect(after.scaleX).toBe(before.scaleX);
  expect(after.x).toBe(before.x); expect(after.y).toBe(before.y);
});
