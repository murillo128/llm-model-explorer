import { expect } from '@playwright/test';
import type { Page } from '@playwright/test';

/** Observe production DOM geometry; never import or drive renderer internals. */
export async function camera(page: Page, rows: number, columns: number) {
  return page.locator('.matrix-scroll:visible').evaluate((host, { rows, columns }) => {
    const canvas = host.querySelector('canvas')!;
    const extent = host.firstElementChild as HTMLElement;
    const [x, y] = canvas.dataset.origin!.split(',').map(Number);
    return { x: x!, y: y!, dpr: devicePixelRatio,
      scaleX: parseFloat(extent.style.width) * devicePixelRatio / columns,
      scaleY: parseFloat(extent.style.height) * devicePixelRatio / rows,
      width: host.clientWidth, height: host.clientHeight,
      rect: canvas.getBoundingClientRect().toJSON() as { left: number; top: number; width: number; height: number },
    };
  }, { rows, columns });
}

export async function zoom(page: Page, rows: number, columns: number, cssCellSize: number) {
  const current = await camera(page, rows, columns);
  const delta = -Math.log(cssCellSize * current.dpr / current.scaleX) / .002;
  // Exercise the actual wheel listener, including its per-event delta clamp.
  const steps = Math.max(1, Math.ceil(Math.abs(delta) / 400));
  await page.locator('.matrix-scroll:visible').evaluate((host, { delta, steps }) => {
    const rect = host.getBoundingClientRect();
    for (let i = 0; i < steps; i++) host.dispatchEvent(new WheelEvent('wheel', {
      deltaY: delta / steps, clientX: rect.left, clientY: rect.top, cancelable: true,
    }));
  }, { delta, steps });
  await expect.poll(async () => (await camera(page, rows, columns)).scaleX / current.dpr).toBeCloseTo(cssCellSize, 4);
}

export async function drag(page: Page, a: { x: number; y: number }, b: { x: number; y: number }) {
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move(b.x, b.y, { steps: 3 });
}

export async function panelGeometry(page: Page) {
  return page.evaluate(() => Object.fromEntries(['.matrix-panel-header', '.matrix-surfaces'].map(selector => {
    const rect = document.querySelector(selector)!.getBoundingClientRect();
    return [selector, [rect.x, rect.y, rect.width, rect.height]];
  })));
}

export async function promptViewport(page: Page) {
  return page.evaluate(() => {
    const editor = document.querySelector('.cm-content')!;
    const scroller = document.querySelector('.cm-scroller')!;
    const selection = getSelection()!;
    return { rect: editor.getBoundingClientRect().toJSON(), scroll: [scroller.scrollLeft, scroller.scrollTop],
      anchor: selection.anchorOffset, focus: selection.focusOffset,
      anchorText: selection.anchorNode?.textContent, focusText: selection.focusNode?.textContent };
  });
}
