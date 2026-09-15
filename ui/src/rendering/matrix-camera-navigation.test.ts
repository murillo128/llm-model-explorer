import { afterEach, expect, it, vi } from 'vitest';
import { MatrixCameraNavigation } from './matrix-camera-navigation';
import type { MatrixViewport } from './matrix-viewport';

const controllers: MatrixCameraNavigation[] = [];
function fixture() {
  const host = document.createElement('div'), canvas = document.createElement('canvas');
  const rows = document.createElement('canvas'), columns = document.createElement('canvas');
  host.tabIndex = 0;
  host.append(canvas, rows, columns);
  document.body.append(host);
  const zoomBack = vi.fn(() => true);
  const viewport = { host, matrix: { renderer: { canvas, state: 'ready' }, zoomBack },
    rows: { canvas: rows, state: 'ready' }, columns: { canvas: columns, state: 'ready' } };
  const controller = new MatrixCameraNavigation(viewport as unknown as MatrixViewport);
  controllers.push(controller);
  return { ...viewport, canvas, rows, columns, zoomBack, controller };
}
function escape(target: Element = document.activeElement!) {
  const event = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
  target.dispatchEvent(event);
  return event;
}
afterEach(() => { controllers.splice(0).forEach((controller) => controller.dispose()); document.body.replaceChildren(); });

it('only the focused local instance handles Escape, and exhausted history is a no-op', () => {
  const a = fixture(), b = fixture();
  a.host.focus();
  expect(escape().defaultPrevented).toBe(true);
  expect(a.zoomBack).toHaveBeenCalledTimes(1);
  expect(b.zoomBack).not.toHaveBeenCalled();
  a.zoomBack.mockReturnValue(false);
  expect(escape().defaultPrevented).toBe(false);
  const outside = document.createElement('button');
  document.body.append(outside); outside.focus();
  escape();
  expect(a.zoomBack).toHaveBeenCalledTimes(2);
  expect(b.zoomBack).not.toHaveBeenCalled();
});

it('allows hovered scientific data with neutral focus and yields to transient dismissal', () => {
  const f = fixture();
  f.canvas.dispatchEvent(new Event('pointerover', { bubbles: true }));
  expect(escape(document.body).defaultPrevented).toBe(true);
  const dismiss = (event: KeyboardEvent) => event.preventDefault();
  document.addEventListener('keydown', dismiss, { once: true });
  escape(document.body);
  expect(f.zoomBack).toHaveBeenCalledTimes(1);
  const dialog = document.createElement('div');
  dialog.tabIndex = 0; document.body.append(dialog); dialog.focus();
  escape(dialog);
  expect(f.zoomBack).toHaveBeenCalledTimes(1);
  f.canvas.dispatchEvent(new Event('pointerout', { bubbles: true }));
  dialog.blur(); escape(document.body);
  expect(f.zoomBack).toHaveBeenCalledTimes(1);
});

it('suppresses context menus only for scientific data canvases and removes handlers on disposal', () => {
  const f = fixture();
  function context(target: Element) {
    const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    target.dispatchEvent(event);
    return event.defaultPrevented;
  }
  expect(context(f.canvas)).toBe(true);
  expect(context(f.rows)).toBe(true);
  expect(context(f.columns)).toBe(true);
  expect(f.zoomBack).toHaveBeenCalledTimes(3);
  expect(context(f.host)).toBe(false);
  expect(context(document.body)).toBe(false);
  f.controller.dispose();
  expect(context(f.canvas)).toBe(false);
  f.host.focus(); escape();
  expect(f.zoomBack).toHaveBeenCalledTimes(3);
});
