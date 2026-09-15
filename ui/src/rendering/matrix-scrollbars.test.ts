import { afterEach, expect, it, vi } from 'vitest';
import { MatrixScrollbars } from './matrix-scrollbars';

const instances: MatrixScrollbars[] = [];
function fixture() {
  const container = document.createElement('div'), viewport = document.createElement('div');
  viewport.tabIndex = 0;
  container.append(viewport); document.body.append(container);
  const size = { clientWidth: 200, clientHeight: 100, scrollWidth: 1000, scrollHeight: 1000 };
  for (const key of Object.keys(size) as (keyof typeof size)[]) Object.defineProperty(viewport, key, { get: () => size[key] });
  const controls = new MatrixScrollbars(viewport, container);
  instances.push(controls);
  const bars = Array.from(controls.element.querySelectorAll<HTMLElement>('[role=scrollbar]'));
  return { viewport, container, controls, size, bars };
}
afterEach(() => {
  instances.splice(0).forEach(instance => instance.dispose());
  document.body.replaceChildren(); vi.useRealTimers(); vi.restoreAllMocks();
});

it('maps native ranges, minimum thumb and corner clearance without emitting scroll events', () => {
  const { controls, viewport, bars } = fixture();
  const scroll = vi.fn(); viewport.addEventListener('scroll', scroll);
  viewport.scrollLeft = 400; viewport.scrollTop = 900; controls.refresh();
  expect(bars[0]!.getAttribute('aria-controls')).toBe(viewport.id);
  expect(bars[0]!.getAttribute('aria-valuemax')).toBe('800');
  expect(bars[0]!.getAttribute('aria-valuenow')).toBe('400');
  expect(bars[0]!.style.width).toBe('182px');
  expect(bars[0]!.firstElementChild!.getAttribute('style')).toContain('width: 36.4px');
  expect(bars[1]!.style.height).toBe('82px');
  expect(bars[1]!.firstElementChild!.getAttribute('style')).toContain('height: 24px');
  expect(bars[1]!.firstElementChild!.getAttribute('style')).toContain('top: 58px');
  expect(scroll).not.toHaveBeenCalled();
});

it('hides nonoverflowing axes and returns focus to the native viewport when a bar disappears', () => {
  const { controls, viewport, bars, size } = fixture();
  bars[1]!.focus(); size.scrollHeight = size.clientHeight; controls.refresh();
  expect(document.activeElement).toBe(viewport);
  expect(bars[1]!.hidden).toBe(true); expect(bars[0]!.hidden).toBe(false);
  expect(bars[0]!.style.width).toBe('200px');
  size.scrollWidth = size.clientWidth; controls.refresh();
  expect(bars[0]!.hidden).toBe(true);
});

it('clamps keyboard navigation and removes listeners and pending attenuation timers on disposal', () => {
  vi.useFakeTimers();
  const registrations = vi.spyOn(EventTarget.prototype, 'addEventListener');
  const removals = vi.spyOn(EventTarget.prototype, 'removeEventListener');
  const { controls, viewport, bars, container } = fixture();
  const targets: EventTarget[] = [viewport, container, window, controls.element, ...bars];
  function key(value: string) {
    const event = new KeyboardEvent('keydown', { key: value, bubbles: true, cancelable: true });
    bars[0]!.dispatchEvent(event); return event;
  }
  expect(key('End').defaultPrevented).toBe(true); expect(viewport.scrollLeft).toBe(800);
  key('ArrowRight'); expect(viewport.scrollLeft).toBe(800);
  key('PageUp'); expect(viewport.scrollLeft).toBe(620);
  key('Home'); key('ArrowLeft'); expect(viewport.scrollLeft).toBe(0);
  expect(vi.getTimerCount()).toBe(1);
  vi.advanceTimersByTime(900); expect(controls.element.hasAttribute('data-active')).toBe(false);
  key('PageDown'); expect(vi.getTimerCount()).toBe(1);
  controls.dispose(); controls.dispose();
  expect(vi.getTimerCount()).toBe(0);
  expect(controls.element.isConnected).toBe(false); expect(viewport.id).toBe('');
  registrations.mock.calls.forEach(([type, listener], i) => {
    const target = registrations.mock.contexts[i];
    if (targets.some(t => t === target)) expect(removals.mock.calls.some(([removedType, removedListener], j) =>
      removedType === type && removedListener === listener && removals.mock.contexts[j] === target)).toBe(true);
  });
  key('End'); expect(viewport.scrollLeft).toBe(180);
  viewport.dispatchEvent(new Event('scroll')); expect(vi.getTimerCount()).toBe(0);
});
