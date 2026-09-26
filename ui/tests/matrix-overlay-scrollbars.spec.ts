import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';
import type {} from './matrix-explorer-harness';

const url = `http://127.0.0.1:${Number(process.env.UI_TEST_PORT ?? 4173) + 1}/tests/matrix-explorer.html`;
const horizontal = 'Scroll matrix horizontally', vertical = 'Scroll matrix vertically';
async function open(page: Page) {
  await page.goto(url); await expect(page.locator('.matrix-scroll')).toBeVisible();
  await page.addStyleTag({ content: '#workspace { --fixture-viewer-height: min(600px, 70vh); }' });
  await page.evaluate(() => window.matrixFixture.render('square'));
  await expect(page.locator('.row-distributions canvas')).toBeVisible();
}
const state = (page: Page) => page.evaluate(() => {
  const f = window.matrixFixture, v = f.viewports.at(-1)!;
  const rect = (s: string) => document.querySelector(s)!.getBoundingClientRect().toJSON();
  return { view: v.renderer.view!, client: [v.host.clientWidth, v.host.clientHeight],
    chrome: [v.host.offsetWidth - v.host.clientWidth, v.host.offsetHeight - v.host.clientHeight],
    scroll: [v.host.scrollLeft, v.host.scrollTop], extent: [v.host.scrollWidth, v.host.scrollHeight],
    matrix: rect('.matrix-scroll canvas'), rows: rect('.row-distributions'), columns: rect('.column-distributions'),
    resources: f.resources(), uploads: f.metrics.uploads, subscriptions: f.subscriptions.length,
    windowScroll: [scrollX, scrollY] };
});
async function waitForCameraToMatchNativeScroll(page: Page) {
  await expect.poll(() => page.evaluate(() => {
    const v = window.matrixFixture.viewports.at(-1)!, view = v.renderer.view!;
    const origin = (scroll: number, scale: number, size: number, cells: number) =>
      Math.max(0, Math.min(cells * scale - size, Math.round(scroll * devicePixelRatio))) / scale;
    return view.x === origin(v.host.scrollLeft, view.scaleX, view.width, v.renderer.geometry.columns) &&
      view.y === origin(v.host.scrollTop, view.scaleY, view.height, v.renderer.geometry.rows);
  })).toBe(true);
}

for (const dpr of [1, 2]) test.describe(`overlay scrolling DPR ${dpr}`, () => {
  test.use({ deviceScaleFactor: dpr });
  test('all overflow modes retain fixed tracks, exact offsets and unchanged chrome geometry', async ({ page }) => {
    await open(page);
    for (const [shape, x, y] of [['square', false, false], ['tall', false, true], ['short', true, false], ['square', true, true]] as const) {
      await page.evaluate(shape => window.matrixFixture.render(shape), shape);
      await expect.poll(() => page.evaluate(shape => window.matrixFixture.viewports.at(-1)!.renderer.geometry.rows === window.matrixFixture.sources[shape].descriptor.shape[0], shape)).toBe(true);
      await page.evaluate(({ x, y }) => {
        const v = window.matrixFixture.viewports.at(-1)!;
        v.zoomAt(x ? v.host.clientWidth * devicePixelRatio / v.renderer.geometry.columns * 2 : 1, 0, 0);
        if (y) v.zoomAt(Math.max(v.renderer.view!.scaleX, v.host.clientHeight * devicePixelRatio / v.renderer.geometry.rows * 2), 0, 0);
      }, { x, y });
      await expect(page.getByRole('scrollbar')).toHaveCount(Number(x) + Number(y));
      await page.mouse.move(0, 0);
      await waitForCameraToMatchNativeScroll(page);
      const before = await state(page);
      expect(before.chrome).toEqual([0, 0]);
      for (const name of [horizontal, vertical]) {
        const bar = page.getByRole('scrollbar', { name });
        if (!await bar.count()) continue;
        await bar.focus();
        await waitForCameraToMatchNativeScroll(page);
        const now = await state(page);
        expect(now).toEqual(before);
        await bar.press('End');
        const axis = name === horizontal ? 0 : 1;
        await expect.poll(async () => (await state(page)).scroll[axis]).toBe(before.extent[axis]! - before.client[axis]!);
        await expect(bar).toHaveAttribute('aria-valuenow', await bar.getAttribute('aria-valuemax') ?? '');
        await bar.press('Home');
        await expect.poll(async () => (await state(page)).scroll[axis]).toBe(0);
        // Native scroll updates before the animation-frame camera refresh.
        await expect.poll(async () => (await state(page)).view).toEqual(before.view);
      }
      await expect.poll(async () => (await state(page)).view).toEqual(before.view);
      const after = await state(page);
      expect(after.client).toEqual(before.client); expect(after.rows).toEqual(before.rows); expect(after.columns).toEqual(before.columns);
      expect(after.resources).toEqual(before.resources); expect(after.uploads).toBe(before.uploads);
      expect(after.windowScroll).toEqual([0, 0]);
    }
  });

  test('thumb drag maps native range, captures outside, cancels cleanly and never starts region selection', async ({ page }) => {
    await open(page);
    await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.zoomAt(30, 0, 0));
    const before = await state(page);
    for (const name of [horizontal, vertical]) {
      const bar = page.getByRole('scrollbar', { name }), isX = name === horizontal;
      await bar.focus();
      const box = (await bar.boundingBox())!, thumb = (await bar.locator('.matrix-scrollbar-thumb').boundingBox())!;
      expect(isX ? thumb.height : thumb.width).toBe(6);
      expect(isX ? thumb.width : thumb.height).toBeGreaterThanOrEqual(24);
      await page.mouse.move(thumb.x + thumb.width / 2, thumb.y + thumb.height / 2);
      await page.mouse.down();
      await expect(bar).toHaveAttribute('data-dragging', '');
      await page.mouse.move(isX ? box.x + box.width / 2 : thumb.x + 3, isX ? thumb.y + 3 : box.y + box.height / 2, { steps: 5 });
      const maximum = Number(await bar.getAttribute('aria-valuemax'));
      await expect.poll(async () => Math.abs(Number(await bar.getAttribute('aria-valuenow')) - maximum / 2)).toBeLessThanOrEqual(1);
      await expect(page.locator('.matrix-zoom-preview')).toHaveCount(0);
      await page.mouse.move(isX ? box.x + box.width + 100 : thumb.x + 3, isX ? thumb.y + 3 : box.y + box.height + 100);
      await expect(bar).toHaveAttribute('aria-valuenow', String(maximum));
      await page.mouse.up();
      await expect(bar).not.toHaveAttribute('data-dragging');
      await bar.press('Home');
      // Real capture followed by explicit browser capture release exercises lostcapture.
      await bar.locator('.matrix-scrollbar-thumb').hover(); await page.mouse.down();
      await bar.evaluate(node => {
        // Mouse pointers in Chromium use id 1.
        if (node.hasPointerCapture(1)) node.releasePointerCapture(1);
      });
      await page.mouse.move(5, 5); await page.mouse.up();
      await expect(bar).not.toHaveAttribute('data-dragging');
      const stopped = Number(await bar.getAttribute('aria-valuenow'));
      await page.mouse.move(100, 100); expect(Number(await bar.getAttribute('aria-valuenow'))).toBe(stopped);
      await bar.locator('.matrix-scrollbar-thumb').hover(); await page.mouse.down();
      await bar.dispatchEvent('pointercancel', { pointerId: 1 }); await page.mouse.up();
      await expect(bar).not.toHaveAttribute('data-dragging');
      await bar.press('Home');
    }
    const after = await state(page);
    expect(after.client).toEqual(before.client); expect(after.view.scaleX).toBe(before.view.scaleX);
    expect(after.rows).toEqual(before.rows); expect(after.columns).toEqual(before.columns);
    expect(after.resources).toEqual(before.resources); expect(after.subscriptions).toBe(before.subscriptions);
    expect(after.windowScroll).toEqual([0, 0]);
    // Both controls stop before the common corner; neither steals that hit.
    const a = (await page.getByRole('scrollbar', { name: horizontal }).boundingBox())!;
    const b = (await page.getByRole('scrollbar', { name: vertical }).boundingBox())!;
    expect(a.x + a.width).toBeLessThanOrEqual(b.x); expect(b.y + b.height).toBeLessThanOrEqual(a.y);
    const bar = page.getByRole('scrollbar', { name: horizontal });
    await bar.locator('.matrix-scrollbar-thumb').hover(); await page.mouse.down();
    await expect(bar).toHaveAttribute('data-dragging', '');
    await page.setViewportSize({ width: page.viewportSize()!.width === 390 ? 780 : 390, height: 700 });
    await expect(bar).not.toHaveAttribute('data-dragging'); await page.mouse.up();
    expect((await state(page)).view.scaleX).toBe(before.view.scaleX);
  });

  test('inactive wide targets pass selection through, edge/focus reveal and wheel reuse the same camera', async ({ page }) => {
    await open(page); await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.zoomAt(30, 0, 0));
    await page.mouse.move(0, 0);
    const layer = page.locator('.matrix-scrollbars');
    await expect(layer).not.toHaveAttribute('data-active');
    const bar = page.getByRole('scrollbar', { name: horizontal });
    const rect = (await bar.boundingBox())!;
    const hit = await page.evaluate(({ x, y }) => document.elementFromPoint(x, y)?.tagName, { x: rect.x + rect.width / 2, y: rect.y + 1 });
    expect(hit).toBe('CANVAS');
    const before = await state(page);
    await page.mouse.move(rect.x + rect.width / 2, rect.y + 2);
    await expect(bar).toHaveAttribute('data-near', '');
    await expect(layer).toHaveAttribute('data-active', '');
    // Revealed targets leave the unpainted outer rim available for exact cells.
    expect(await page.locator('.matrix-scroll').evaluate(viewport => {
      const r = viewport.getBoundingClientRect();
      return [[r.right - 1, r.top + 1], [r.left + 1, r.bottom - 1]]
        .every(([x, y]) => document.elementFromPoint(x!, y!)?.tagName === 'CANVAS');
    })).toBe(true);
    expect((await state(page)).client).toEqual(before.client);
    await expect(layer).not.toHaveAttribute('data-active');
    await expect(bar).not.toHaveAttribute('data-near');
    await expect(bar.locator('.matrix-scrollbar-thumb')).toHaveCSS('opacity', '0.4');
    await bar.focus(); await page.mouse.move(0, 0);
    await expect(bar.locator('.matrix-scrollbar-thumb')).toHaveCSS('opacity', '0.95');
    await bar.press('PageDown'); await bar.press('ArrowLeft');
    expect(Number(await bar.getAttribute('aria-valuenow'))).toBeCloseTo(Math.round(before.client[0]! * .9) - 40, 0);
    await bar.press('Home');
    await bar.hover(); await page.mouse.wheel(0, -100);
    await expect.poll(async () => (await state(page)).view.scaleX).toBeGreaterThan(before.view.scaleX);
    await page.getByRole('button', { name: 'Fit width' }).click();
    await expect.poll(async () => (await state(page)).view.scaleX).toBeCloseTo(Math.max(1, before.client[0]! * dpr / 100));
    expect((await state(page)).windowScroll).toEqual([0, 0]);
  });

  test('blank profile margins have no logical hit while offset data projects exact selected rows and columns', async ({ page }) => {
    await open(page); await page.evaluate(() => window.matrixFixture.viewports.at(-1)!.zoomAt(1, 0, 0));
    for (const axis of ['rows', 'columns'] as const) {
      const selector = axis === 'rows' ? '.row-distributions' : '.column-distributions';
      const track = (await page.locator(selector).boundingBox())!, data = (await page.locator(`${selector} canvas`).boundingBox())!;
      const blank = { x: track.x + 2, y: track.y + 2 };
      await page.mouse.move(blank.x, blank.y); await page.mouse.down(); await page.mouse.move(blank.x + 8, blank.y + 8); await page.mouse.up();
      await expect(page.locator('.matrix-zoom-preview')).toHaveCount(0);
      expect((await state(page)).view.scaleX).toBe(1);
      const start = { x: data.x + 10 / dpr, y: data.y + 10 / dpr };
      await page.mouse.move(start.x, start.y); await page.mouse.down();
      await page.mouse.move(start.x + (axis === 'columns' ? 20 / dpr : 0), start.y + (axis === 'rows' ? 20 / dpr : 0), { steps: 3 });
      await expect(page.locator(`.matrix-zoom-preview[data-surface=${axis}]`)).toHaveAttribute('data-bounds', JSON.stringify({ [axis]: [10, 30] }));
      const m = (await page.locator('.matrix-zoom-preview[data-surface=matrix]').boundingBox())!;
      const p = (await page.locator(`.matrix-zoom-preview[data-surface=${axis}]`).boundingBox())!;
      expect(axis === 'rows' ? p.y : p.x).toBeCloseTo(axis === 'rows' ? m.y : m.x, 3);
      await page.keyboard.press('Escape'); await page.mouse.up();
    }
    const before = await state(page);
    for (let i = 0; i < 3; i++) {
      await page.evaluate(() => window.matrixFixture.render(null));
      await expect(page.locator('.matrix-scrollbars')).toHaveCount(0);
      expect(await page.evaluate(() => window.matrixFixture.resources())).toEqual({ textures: 0, displays: 0, framebuffers: 0, buffers: 0, programs: 0, cpuBytes: 0 });
      await page.evaluate(() => window.matrixFixture.render('square')); await expect(page.locator('.matrix-scrollbars')).toHaveCount(1);
    }
    expect((await state(page)).resources).toEqual(before.resources);
  });
});
