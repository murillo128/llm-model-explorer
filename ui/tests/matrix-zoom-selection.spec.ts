import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import type {} from './matrix-explorer-harness';

const url = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/matrix-explorer.html`;
async function open(page: Page) {
  await page.goto(url);
  await expect(page.locator('.matrix-scroll')).toBeVisible();
  await page.addStyleTag({ content: '#workspace { height: min(600px, 70vh); }' });
  await page.evaluate(() => window.matrixFixture.render('square'));
  await expect.poll(() => page.evaluate(() => window.matrixFixture.subscriptions.length)).toBe(2);
  await page.evaluate(() => {
    const f = window.matrixFixture, u = f.subscriptions.at(-1)!;
    u.values(Float32Array.from({ length: 12000 }, (_, i) => (i % 101) / 100), 0);
    u.distribution('rows', Uint32Array.from({ length: 12000 }, (_, i) => i % 2), 0);
    u.distribution('columns', Uint32Array.from({ length: 10000 }, (_, i) => i % 2), 0);
    f.viewports.at(-1)!.zoomAt(12, 0, 0);
  });
}
async function state(page: Page) {
  return page.evaluate(() => {
    const f = window.matrixFixture, v = f.viewports.at(-1)!;
    return { view: v.renderer.view!, width: v.host.clientWidth, height: v.host.clientHeight,
      uploads: f.metrics.uploads, resources: f.resources() };
  });
}
async function point(page: Page, column: number, row: number, axis?: 'rows' | 'columns') {
  return page.evaluate(({ column, row, axis }) => {
    const f = window.matrixFixture, matrix = f.viewports.at(-1)!.renderer;
    const r = axis ? f.renderers.find(r => r.state === 'ready' && r !== matrix &&
      (axis === 'rows' ? r.geometry.rows === 120 : r.geometry.rows === 100))! : matrix;
    const v = r.view!, rect = r.canvas.getBoundingClientRect();
    return { x: rect.left + (axis === 'rows' ? 30 / v.dpr : (column - v.x) * v.scaleX / v.dpr),
      y: rect.top + (axis === 'columns' ? 30 / v.dpr : (row - v.y) * v.scaleY / v.dpr) };
  }, { column, row, axis });
}
const preview = (page: Page) => page.locator('.matrix-zoom-preview');
async function drag(page: Page, a: { x: number; y: number }, b: { x: number; y: number }) {
  await page.mouse.move(a.x, a.y); await page.mouse.down(); await page.mouse.move(b.x, b.y, { steps: 3 });
}
for (const dpr of [1, 1.25, 2]) test.describe(`selection DPR ${dpr}`, () => {
  test.use({ deviceScaleFactor: dpr });
  for (const reverse of [false, true]) test(`rectangle ${reverse ? 'reverse' : 'forward'} exact preview, fit and readout`, async ({ page }) => {
    await open(page);
    const before = await state(page);
    const a = await point(page, 2, 2), b = await point(page, 12, 7);
    await drag(page, reverse ? b : a, reverse ? a : b);
    await expect(preview(page)).toHaveAttribute('data-bounds', JSON.stringify({ columns: [2, 12], rows: [2, 7] }));
    const style = await preview(page).evaluate(el => ({ fill: getComputedStyle(el).backgroundColor, border: getComputedStyle(el).borderTopColor }));
    expect(style).toEqual({ fill: 'rgba(245, 154, 56, 0.08)', border: 'rgb(245, 154, 56)' });
    expect((await state(page)).view).toEqual(before.view);
    await expect(page.locator('.inspection-readout')).toHaveCount(0);
    await page.mouse.up();
    await expect(preview(page)).toHaveCount(0);
    const after = await state(page);
    const scale = Math.min(Math.floor(after.width * dpr) / 10, Math.floor(after.height * dpr) / 5);
    expect(after.view.scaleX).toBeCloseTo(scale); expect(after.view.scaleY).toBe(scale);
    const expectedX = Math.max(0, 7 - Math.floor(after.width * dpr) / scale / 2);
    const expectedY = Math.max(0, 4.5 - Math.floor(after.height * dpr) / scale / 2);
    expect(Math.abs(after.view.x - expectedX)).toBeLessThanOrEqual(2 * dpr / scale);
    expect(Math.abs(after.view.y - expectedY)).toBeLessThanOrEqual(2 * dpr / scale);
    expect(after.uploads).toBe(before.uploads);
    expect(after.resources.textures).toBe(before.resources.textures);
    const p = await point(page, 3.5, 3.5);
    await page.mouse.click(p.x, p.y);
    await expect(page.locator('.inspection-readout')).toContainText('row 3 · column 3');
    await expect(page.locator('.inspection-readout')).toContainText('0'); // index 303 wraps to zero.
  });
  for (const axis of ['rows', 'columns'] as const) test(`${axis} range preserves orthogonal center and exact counts`, async ({ page }) => {
    await open(page);
    await page.evaluate(() => { const v = window.matrixFixture.viewports.at(-1)!; v.host.scrollLeft = 30; v.host.scrollTop = 40; v.refresh(); });
    const before = await state(page), origin = axis === 'rows' ? before.view.y : before.view.x;
    const start = Math.ceil(origin) + 2, end = start + 5;
    const a = await point(page, start, start, axis), b = await point(page, end, end, axis);
    await drag(page, b, a);
    await expect(preview(page)).toHaveAttribute('data-bounds', JSON.stringify({ [axis]: [start, end] }));
    await page.mouse.up();
    const after = await state(page), horizontal = axis === 'columns';
    const scale = Math.floor((horizontal ? after.width : after.height) * dpr) / 5;
    expect(after.view.scaleX).toBeCloseTo(scale); expect(after.view.scaleY).toBe(scale);
    const oldCenter = horizontal ? before.view.y + before.view.height / before.view.scaleY / 2 : before.view.x + before.view.width / before.view.scaleX / 2;
    const size = (horizontal ? after.view.height : after.view.width) / scale;
    const expected = Math.max(0, Math.min((horizontal ? 120 : 100) - size, oldCenter - size / 2));
    expect(Math.abs((horizontal ? after.view.y : after.view.x) - expected)).toBeLessThanOrEqual(2 * dpr / scale);
    expect(after.uploads).toBe(before.uploads);
    await expect(preview(page)).toHaveCount(0);
  });
});

test('click threshold, degenerate drag, Escape, pointer cancel, capture loss and wheel cancellation', async ({ page }) => {
  await open(page);
  const before = await state(page), a = await point(page, 2.5, 2.5), b = await point(page, 12, 7);
  await drag(page, a, { x: a.x + 2, y: a.y + 1 }); await page.mouse.up();
  await expect(preview(page)).toHaveCount(0);
  await expect(page.locator('.inspection-readout')).toContainText('row 2 · column 2');
  expect((await state(page)).view).toEqual(before.view);
  await drag(page, a, { x: b.x, y: a.y }); await page.mouse.up();
  expect((await state(page)).view).toEqual(before.view);
  for (const cancel of ['Escape', 'pointercancel', 'lostpointercapture', 'wheel']) {
    await drag(page, a, b); await expect(preview(page)).toBeVisible();
    if (cancel === 'Escape') await page.keyboard.press('Escape');
    else if (cancel === 'wheel') await page.locator('.matrix-scroll').dispatchEvent('wheel', { deltaY: 0 });
    else await page.locator('.matrix-inspectable').dispatchEvent(cancel, { pointerId: 1 });
    await expect(preview(page)).toHaveCount(0); await page.mouse.up();
    expect((await state(page)).view).toEqual(before.view);
  }
});

test('one-finger selection yields to pinch; replacement removes an active preview and resources', async ({ page }) => {
  await open(page);
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true });
  const a = await point(page, 2, 2), b = await point(page, 12, 7);
  const before = await state(page);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ ...a, id: 1 }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ ...b, id: 1 }] });
  await expect(preview(page)).toBeVisible();
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ ...b, id: 1 }, { x: b.x + 30, y: b.y, id: 2 }] });
  await expect(preview(page)).toHaveCount(0);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: b.x - 15, y: b.y, id: 1 }, { x: b.x + 45, y: b.y, id: 2 }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  expect((await state(page)).view.scaleX).toBeGreaterThan(before.view.scaleX);
  await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.zoomAt(12, 0, 0));
  const p = await point(page, 20, 10), q = await point(page, 25, 15);
  await drag(page, p, q); await expect(preview(page)).toBeVisible();
  await page.evaluate(() => window.matrixFixture.render(null));
  await expect(preview(page)).toHaveCount(0); await page.mouse.up();
  expect(await page.evaluate(() => window.matrixFixture.resources())).toEqual({ textures: 0, displays: 0, framebuffers: 0, buffers: 0, programs: 0, cpuBytes: 0 });
});

for (const scale of [3.7, 8.5]) test(`fractional scale ${scale}: exact bounds and unchanged scalar/density pixels`, async ({ page }) => {
  await open(page);
  await page.evaluate(scale => {
    const v = window.matrixFixture.viewports.at(-1)!;
    v.zoomAt(scale, 0, 0); v.host.scrollLeft = 7; v.host.scrollTop = 11; v.refresh();
  }, scale);
  const pixels = () => page.evaluate(() => window.matrixFixture.renderers.filter(r => r.state === 'ready').map(r => {
    r.draw();
    const gl = r.canvas.getContext('webgl2')!, bytes = new Uint8Array(r.canvas.width * r.canvas.height * 4);
    gl.readPixels(0, 0, r.canvas.width, r.canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
    // Retain compact reproducible hashes, not a second scalar representation.
    let hash = 2166136261;
    for (const byte of bytes) hash = Math.imul(hash ^ byte, 16777619) >>> 0;
    return { hash, first: r.readCell(0, 0), last: r.readCell(r.geometry.rows - 1, r.geometry.columns - 1) };
  }));
  const before = await pixels();
  const a = await point(page, 5, 5), b = await point(page, 15, 10);
  await drag(page, a, b);
  await expect(preview(page)).toHaveAttribute('data-bounds', JSON.stringify({ columns: [5, 15], rows: [5, 10] }));
  expect(await pixels()).toEqual(before);
  await page.keyboard.press('Escape'); await page.mouse.up();
  expect(await pixels()).toEqual(before);
});

test('touch drag applies immediately and camera/context changes cancel pending previews', async ({ page }) => {
  await open(page);
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true });
  const a = await point(page, 2, 2), b = await point(page, 12, 7), before = await state(page);
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ ...a, id: 1 }] });
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ ...b, id: 1 }] });
  await expect(preview(page)).toBeVisible();
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await expect(preview(page)).toHaveCount(0);
  expect((await state(page)).view.scaleX).toBeGreaterThan(before.view.scaleX);
  await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.zoomAt(12, 0, 0));
  for (const change of ['scroll', 'dpr', 'context']) {
    const c = await point(page, 2, 2), d = await point(page, 12, 7);
    await drag(page, c, d); await expect(preview(page)).toBeVisible();
    await page.evaluate(change => {
      const v = window.matrixFixture.viewports.at(-1)!;
      if (change === 'scroll') { v.host.scrollTop = 20; v.refresh(); }
      else if (change === 'dpr') {
        Object.defineProperty(window, 'devicePixelRatio', { configurable: true, value: 1.25 });
        window.dispatchEvent(new Event('resize'));
      } else v.canvas.getContext('webgl2')!.getExtension('WEBGL_lose_context')!.loseContext();
    }, change);
    await expect(preview(page)).toHaveCount(0); await page.mouse.up();
    if (change !== 'context') await page.evaluate(() => { const v = window.matrixFixture.viewports.at(-1)!; v.zoomAt(12, 0, 0); v.host.scrollLeft = 0; v.host.scrollTop = 0; v.refresh(); });
  }
});

test('short matrix rectangle fits the available viewport, with no-op and far-edge camera bounds', async ({ page }) => {
  await open(page);
  await page.evaluate(() => window.matrixFixture.render('short'));
  await expect.poll(() => page.evaluate(() => window.matrixFixture.subscriptions.length)).toBe(3);
  const before = await state(page);
  // Programmatic boundary application exercises the same release path when a
  // short matrix's single-row gesture would fall below the CSS drag threshold.
  await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.zoomToBounds({ columns: [0, 20], rows: [0, 2] }));
  const after = await state(page);
  expect(after.view.scaleX).toBeCloseTo(Math.min(after.width * after.view.dpr / 20, after.height * after.view.dpr / 2));
  expect(after.view.scaleX).toBeGreaterThan(before.view.scaleX);
  expect(after.view.x).toBe(0); expect(after.view.y).toBe(0);
  await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.zoomToBounds({ columns: [1, 1] }));
  expect((await state(page)).view).toEqual(after.view);
  await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.zoomToBounds({ columns: [127, 128], rows: [3, 4] }));
  const edge = await state(page), scale = edge.view.scaleX;
  expect(scale).toBeCloseTo(Math.min(edge.width * edge.view.dpr, edge.height * edge.view.dpr));
  expect(Math.abs(edge.view.x + edge.view.width / scale - 128)).toBeLessThanOrEqual(2 * edge.view.dpr / scale);
  expect(Math.abs(edge.view.y + edge.view.height / scale - 4)).toBeLessThanOrEqual(2 * edge.view.dpr / scale);
});
