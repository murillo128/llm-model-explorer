import './matrix-scrollbars.css';

const TARGET = 18;
const MIN_THUMB = 24;
let nextRegion = 0;
type Axis = 'horizontal' | 'vertical';
type Bar = { axis: Axis; control: HTMLDivElement; thumb: HTMLDivElement;
  maximum: number; travel: number; length: number; value: number };
type Drag = { bar: Bar; pointer: number; offset: number; geometry: string };

/** Native scrolling is authoritative; this class owns only disposable DOM chrome. */
export class MatrixScrollbars {
  readonly element = document.createElement('div');
  private readonly bars: Bar[];
  private drag: Drag | null = null;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private disposed = false;
  private readonly originalId: string;
  private readonly removeListeners: (() => void)[] = [];

  constructor(private readonly viewport: HTMLElement, private readonly container: HTMLElement) {
    this.originalId = viewport.id;
    if (!viewport.id) viewport.id = `matrix-scroll-region-${++nextRegion}`;
    this.element.className = 'matrix-scrollbars';
    this.bars = (['horizontal', 'vertical'] as const).map(axis => {
      const control = document.createElement('div'), thumb = document.createElement('div');
      control.className = `matrix-scrollbar matrix-scrollbar-${axis}`;
      control.setAttribute('role', 'scrollbar');
      control.setAttribute('aria-label', `Scroll matrix ${axis === 'horizontal' ? 'horizontally' : 'vertically'}`);
      control.setAttribute('aria-orientation', axis);
      control.setAttribute('aria-controls', viewport.id);
      control.setAttribute('aria-valuemin', '0');
      control.tabIndex = 0;
      thumb.className = 'matrix-scrollbar-thumb';
      thumb.setAttribute('aria-hidden', 'true');
      control.append(thumb);
      this.element.append(control);
      const bar: Bar = { axis, control, thumb, maximum: 0, travel: 0, length: 0, value: 0 };
      this.listen(control, 'pointerdown', event => this.down(event as PointerEvent, bar));
      this.listen(control, 'pointermove', event => this.move(event as PointerEvent));
      this.listen(control, 'pointerup', event => this.up(event as PointerEvent));
      this.listen(control, 'pointercancel', () => this.cancel());
      this.listen(control, 'lostpointercapture', () => this.cancel());
      this.listen(control, 'keydown', event => this.key(event as KeyboardEvent, bar));
      this.listen(control, 'click', event => { event.preventDefault(); event.stopPropagation(); });
      return bar;
    });
    container.append(this.element);
    this.listen(viewport, 'scroll', () => { this.refresh(); this.reveal(); });
    this.listen(container, 'pointermove', event => this.approach(event as PointerEvent));
    this.listen(container, 'pointerleave', () => this.bars.forEach(({ control }) => delete control.dataset.near));
    this.listen(window, 'blur', () => this.cancel());
    // Capture can be released before gotpointercapture, when browsers emit no
    // lostpointercapture. The next native event must still end the drag.
    this.listen(window, 'pointermove', event => {
      const drag = this.drag;
      if (drag?.pointer === (event as PointerEvent).pointerId && !drag.bar.control.hasPointerCapture(drag.pointer)) this.cancel();
    });
    for (const type of ['pointerup', 'pointercancel']) this.listen(window, type, event => {
      if (this.drag?.pointer === (event as PointerEvent).pointerId) this.cancel();
    });
    this.listen(this.element, 'wheel', () => this.cancel());
    this.listen(this.element, 'touchstart', event => { if ((event as TouchEvent).touches.length > 1) this.cancel(); });
    this.refresh();
  }

  private listen(target: EventTarget, type: string, listener: EventListener) {
    target.addEventListener(type, listener);
    this.removeListeners.push(() => target.removeEventListener(type, listener));
  }

  refresh() {
    if (this.disposed) return;
    if (this.drag && this.drag.geometry !== this.geometry()) this.cancel();
    const v = this.viewport, rect = v.getBoundingClientRect(), parent = this.container.getBoundingClientRect();
    Object.assign(this.element.style, { left: `${rect.left + v.clientLeft - parent.left - this.container.clientLeft}px`,
      top: `${rect.top + v.clientTop - parent.top - this.container.clientTop}px`,
      width: `${v.clientWidth}px`, height: `${v.clientHeight}px` });
    const horizontal = v.scrollWidth > v.clientWidth, vertical = v.scrollHeight > v.clientHeight;
    for (const bar of this.bars) {
      const isX = bar.axis === 'horizontal';
      const visible = isX ? horizontal : vertical;
      if (!visible && this.drag?.bar === bar) this.cancel();
      if (!visible && document.activeElement === bar.control) v.focus({ preventScroll: true });
      bar.control.hidden = !visible;
      const client = isX ? v.clientWidth : v.clientHeight;
      const extent = isX ? v.scrollWidth : v.scrollHeight;
      const track = Math.max(0, client - ((isX ? vertical : horizontal) ? TARGET : 0));
      bar.maximum = Math.max(0, extent - client);
      bar.value = Math.max(0, Math.min(bar.maximum, isX ? v.scrollLeft : v.scrollTop));
      bar.length = Math.min(track, Math.max(MIN_THUMB, extent ? track * client / extent : track));
      bar.travel = track - bar.length;
      const offset = bar.maximum ? bar.value / bar.maximum * bar.travel : 0;
      bar.control.setAttribute('aria-valuemax', String(bar.maximum));
      bar.control.setAttribute('aria-valuenow', String(bar.value));
      bar.control.style[isX ? 'width' : 'height'] = `${track}px`;
      bar.thumb.style[isX ? 'width' : 'height'] = `${bar.length}px`;
      bar.thumb.style[isX ? 'left' : 'top'] = `${offset}px`;
    }
  }

  private geometry() {
    const v = this.viewport;
    return `${v.clientWidth},${v.clientHeight},${v.scrollWidth},${v.scrollHeight}`;
  }

  private reveal() {
    this.element.dataset.active = '';
    clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = undefined;
      delete this.element.dataset.active;
      this.bars.forEach(({ control }) => delete control.dataset.near);
    }, 900);
  }

  private approach(event: PointerEvent) {
    if (event.buttons || event.pointerType === 'touch') return;
    const rect = this.element.getBoundingClientRect();
    const inside = event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
    let approached = false;
    for (const bar of this.bars) {
      const near = !bar.control.hidden && inside && (bar.axis === 'horizontal' ? rect.bottom - event.clientY : rect.right - event.clientX) <= 24;
      bar.control.toggleAttribute('data-near', near);
      approached ||= near;
    }
    if (approached) this.reveal();
  }

  private coordinate(event: PointerEvent, bar: Bar) {
    const rect = bar.control.getBoundingClientRect();
    return bar.axis === 'horizontal' ? event.clientX - rect.left : event.clientY - rect.top;
  }

  private setValue(bar: Bar, value: number) {
    const next = Math.max(0, Math.min(bar.maximum, value));
    // Do not synthesize scroll events or maintain a separate panning origin.
    if (bar.axis === 'horizontal') this.viewport.scrollLeft = next;
    else this.viewport.scrollTop = next;
    this.refresh();
    this.reveal();
  }

  private down(event: PointerEvent, bar: Bar) {
    if (event.button !== 0 || this.drag || bar.control.hidden) return;
    event.preventDefault(); event.stopPropagation();
    this.refresh();
    bar.control.focus({ preventScroll: true });
    const position = this.coordinate(event, bar);
    const offset = bar.maximum ? bar.value / bar.maximum * bar.travel : 0;
    const onThumb = event.target === bar.thumb;
    this.drag = { bar, pointer: event.pointerId, offset: onThumb ? position - offset : bar.length / 2, geometry: this.geometry() };
    bar.control.dataset.dragging = '';
    bar.control.setPointerCapture(event.pointerId);
    this.move(event);
  }

  private move(event: PointerEvent) {
    const drag = this.drag;
    if (!drag || drag.pointer !== event.pointerId) return;
    event.preventDefault(); event.stopPropagation();
    const position = this.coordinate(event, drag.bar) - drag.offset;
    if (drag.bar.travel > 0) this.setValue(drag.bar, position / drag.bar.travel * drag.bar.maximum);
  }

  private up(event: PointerEvent) {
    if (this.drag?.pointer !== event.pointerId) return;
    this.move(event);
    this.cancel();
  }

  private cancel() {
    const drag = this.drag;
    this.drag = null;
    if (!drag) return;
    delete drag.bar.control.dataset.dragging;
    if (drag.bar.control.hasPointerCapture(drag.pointer)) drag.bar.control.releasePointerCapture(drag.pointer);
  }

  private key(event: KeyboardEvent, bar: Bar) {
    let value: number;
    const page = (bar.axis === 'horizontal' ? this.viewport.clientWidth : this.viewport.clientHeight) * .9;
    switch (event.key) {
      case 'ArrowLeft': case 'ArrowUp': value = bar.value - 40; break;
      case 'ArrowRight': case 'ArrowDown': value = bar.value + 40; break;
      case 'PageUp': value = bar.value - page; break;
      case 'PageDown': value = bar.value + page; break;
      case 'Home': value = 0; break;
      case 'End': value = bar.maximum; break;
      case 'Escape':
        if (!this.drag) return;
        event.preventDefault(); event.stopPropagation(); this.cancel(); return;
      default: return;
    }
    event.preventDefault(); event.stopPropagation();
    this.setValue(bar, value);
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.cancel();
    clearTimeout(this.timer);
    this.removeListeners.forEach(remove => remove());
    this.element.remove();
    if (this.originalId) this.viewport.id = this.originalId;
    else this.viewport.removeAttribute('id');
  }
}
