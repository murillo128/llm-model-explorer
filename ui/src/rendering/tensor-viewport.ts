import { TensorRenderer } from './tensor-renderer';
import type { RendererOptions } from './tensor-renderer';
import type { TensorDescriptor, ViewGeometry } from './geometry';
import { centeredOffset, fitWidthScale, focalScroll } from './geometry';
import { selectionCamera } from './zoom-selection-geometry';
import type { ZoomBounds } from './zoom-selection-geometry';
import { CameraHistory } from './camera-history';

export interface ViewportOptions extends RendererOptions {
  readonly zoom?: boolean;
  readonly onViewChange?: (view: ViewGeometry) => void;
}

/** Small non-React native-scroll adapter. The supplied empty host owns layout. */
export class TensorViewport {
  readonly renderer: TensorRenderer;
  readonly canvas = document.createElement('canvas');
  private readonly extent = document.createElement('div');
  private readonly surface = document.createElement('div');
  private readonly status = document.createElement('div');
  private readonly observer: ResizeObserver;
  private scale = 1;
  private fitting = true;
  private requestedScroll: [number, number] | null = null;
  private pinch: { distance: number; scale: number; x: number; y: number } | null = null;
  private readonly history = new CameraHistory();
  private wheelGesture: { token: object; time: number } | null = null;
  private media: MediaQueryList | null = null;
  private frame = 0;
  private disposed = false;
  private readonly zoomTargets = new Set<HTMLElement>();
  private readonly originalStyle: string | null;

  constructor(readonly host: HTMLElement, descriptor: TensorDescriptor, private readonly options: ViewportOptions = {}) {
    if (host.childNodes.length) throw new Error('TensorViewport requires an empty host.');
    this.originalStyle = host.getAttribute('style');
    Object.assign(host.style, { overflowX: 'auto', overflowY: host.style.overflowY || 'auto', position: 'relative', padding: '0' });
    Object.assign(this.extent.style, { position: 'relative', pointerEvents: 'none' });
    Object.assign(this.canvas.style, { position: 'absolute', display: 'block' });
    // Clip the canvas's untransformed layout box so it cannot enlarge native scroll.
    Object.assign(this.surface.style, { position: 'absolute', overflow: 'hidden' });
    Object.assign(this.status.style, { position: 'sticky', left: '0', top: '0' });
    this.status.setAttribute('role', 'status');
    this.canvas.setAttribute('aria-label', 'Tensor data; exact square cells');
    this.surface.append(this.canvas);
    host.append(this.extent, this.surface, this.status);
    try {
      this.renderer = new TensorRenderer(this.canvas, descriptor, { ...options, onStateChange: (state) => {
        this.canvas.hidden = state !== 'ready';
        // display:block would otherwise override the browser's hidden styling.
        this.canvas.style.display = state === 'ready' ? 'block' : 'none';
        this.status.textContent = state === 'empty' ? 'Empty tensor' : state === 'ready' ? '' :
          state === 'lost' ? 'WebGL2 context lost. Waiting for restoration.' :
            state === 'needs-reconstruction' ? 'WebGL2 restored. Retry to reconstruct tensor data.' :
              state === 'failed' ? 'WebGL2 rendering failed. Retry to reconstruct tensor data.' : '';
        options.onStateChange?.(state);
      } });
    } catch (error) {
      this.canvas.remove();
      this.surface.remove();
      this.extent.remove();
      this.status.textContent = error instanceof Error ? error.message : 'WebGL2 allocation failed.';
      throw error;
    }
    const limits = this.renderer.limits;
    const maxWidth = Math.min(options.framebufferLimit ?? Infinity, limits.renderbufferSize, limits.viewportWidth);
    const maxHeight = Math.min(options.framebufferLimit ?? Infinity, limits.renderbufferSize, limits.viewportHeight);
    const hostMaxWidth = getComputedStyle(host).maxWidth;
    // Keep an inline relative ceiling (e.g. 50vh) responsive across resizes.
    const hostMaxHeight = host.style.maxHeight || getComputedStyle(host).maxHeight;
    this.setCeilings = () => {
      const width = `${maxWidth / window.devicePixelRatio}px`;
      const height = `${maxHeight / window.devicePixelRatio}px`;
      host.style.maxWidth = hostMaxWidth === 'none' ? width : `min(${width}, ${hostMaxWidth})`;
      host.style.maxHeight = hostMaxHeight === 'none' ? height : `min(${height}, ${hostMaxHeight})`;
    };
    this.observer = new ResizeObserver(this.schedule);
    this.observer.observe(host);
    host.addEventListener('scroll', this.schedule);
    if (options.zoom) {
      host.style.touchAction = 'pan-x pan-y';
      this.addZoomTarget(host);
    }
    window.addEventListener('resize', this.dprChanged);
    try { this.dprChanged(); } catch (error) { this.dispose(); throw error; }
  }

  private readonly setCeilings: () => void;

  /** Preserve native wheel ancestry for strips and the existing matrix zoom policy. */
  attachOverlay(target: HTMLElement) {
    if (this.options.zoom) { this.addZoomTarget(target); return; }
    // The native extent supplies the horizontal sticky bounds. Its explicit
    // data height keeps the overlay out of flow sizing; the strip viewport
    // already has enough height for the controls. Browser scrolling remains
    // responsible for wheel units, modifiers, momentum and endpoint clamping.
    this.extent.append(target);
    Object.assign(target.style, { position: 'sticky', left: '0px', top: '0px' });
  }

  private addZoomTarget(target: HTMLElement) {
    if (!this.options.zoom || this.zoomTargets.has(target)) return;
    this.zoomTargets.add(target);
    target.addEventListener('wheel', this.wheel, { passive: false });
    target.addEventListener('touchstart', this.touchStart, { passive: false });
    target.addEventListener('touchmove', this.touchMove, { passive: false });
    target.addEventListener('touchend', this.touchEnd);
    target.addEventListener('touchcancel', this.touchEnd);
  }

  private dprChanged = () => {
    this.media?.removeEventListener('change', this.dprChanged);
    this.media = matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    this.media.addEventListener('change', this.dprChanged);
    const old = this.renderer.view;
    const dpr = window.devicePixelRatio;
    if (this.options.zoom && old && old.dpr !== dpr) {
      // Retain logical origins through monitor/DPR changes, including fit mode.
      const scale = this.fitting ? fitWidthScale(this.renderer.geometry.columns, this.host.clientWidth, dpr) : this.scale;
      this.requestedScroll = [old.x * scale / dpr, old.y * scale / dpr];
    }
    this.setCeilings();
    this.refresh();
  };

  private schedule = () => {
    if (!this.frame) this.frame = requestAnimationFrame(() => { this.frame = 0; this.refresh(); });
  };

  /** Reset this camera to fit width; native scale is the lower bound. */
  fitWidth() {
    if (!this.options.zoom || this.disposed) return;
    this.history.clear();
    this.wheelGesture = null;
    this.pinch = null;
    this.fitting = true;
    this.requestedScroll = [0, 0];
    this.refresh();
  }

  /** Reveal a linked logical row using only this host's vertical scroll. */
  revealRow(row: number) {
    if (this.disposed || !Number.isInteger(row) || row < 0 || row >= this.renderer.geometry.rows) return;
    this.refresh();
    const view = this.renderer.view;
    if (!view) return;
    const top = row * view.scaleY / view.dpr;
    const bottom = (row + 1) * view.scaleY / view.dpr;
    const height = view.height / view.dpr;
    if (top < this.host.scrollTop || bottom - top > height) this.host.scrollTop = top;
    else if (bottom > this.host.scrollTop + height) this.host.scrollTop = bottom - height;
    this.refresh();
  }

  zoomAt(scale: number, cssX: number, cssY: number) {
    this.wheelGesture = null;
    this.zoom(scale, cssX, cssY);
  }

  private cameraState() {
    const view = this.renderer.view;
    return view ? { scale: this.scale, x: view.x, y: view.y } : null;
  }

  private zoom(scale: number, cssX: number, cssY: number, gesture?: object) {
    this.refresh();
    const before = this.cameraState();
    this.applyZoom(scale, cssX, cssY);
    const after = this.cameraState();
    if (before && after) this.history.record(before, after, gesture);
  }

  private applyZoom(scale: number, cssX: number, cssY: number) {
    if (!this.options.zoom || this.disposed || !Number.isFinite(scale)) return;
    this.refresh(); // Reconcile any native scrolling before resolving the focal point.
    const view = this.renderer.view;
    if (!view) return;
    // Accommodate both 64x fit width and direct selection of a single row/column.
    const oldRect = this.canvas.getBoundingClientRect();
    const next = Math.max(1, Math.min(scale, Math.max(this.host.clientWidth * view.dpr,
      this.host.clientHeight * view.dpr, 64 * fitWidthScale(this.renderer.geometry.columns, this.host.clientWidth, view.dpr))));
    this.requestedScroll = [focalScroll(view.x, cssX, this.scale, next, view.dpr),
      focalScroll(view.y, cssY, this.scale, next, view.dpr)];
    this.fitting = false;
    this.scale = next;
    this.refresh();
    const rect = this.canvas.getBoundingClientRect();
    if (rect.left !== oldRect.left || rect.top !== oldRect.top) {
      // Centering can move the canvas as an axis starts/stops overflowing. Keep
      // the focal point in screen space, using the original logical coordinate.
      this.requestedScroll = [
        (view.x + cssX * view.dpr / view.scaleX) * next / view.dpr - cssX + rect.left - oldRect.left,
        (view.y + cssY * view.dpr / view.scaleY) * next / view.dpr - cssY + rect.top - oldRect.top,
      ];
      this.refresh();
    }
  }

  /** Fit exact logical bounds using this same square-cell/native-scroll camera. */
  zoomToBounds(bounds: ZoomBounds) {
    if (!this.options.zoom || this.disposed) return;
    this.wheelGesture = null;
    this.refresh();
    const view = this.renderer.view;
    if (!view) return;
    const before = this.cameraState()!;
    // Scrollbar appearance can change the available height after scaling. Resolve
    // that layout with the original orthogonal center, then fit once more.
    for (let pass = 0; pass < 2; pass++) {
      const camera = selectionCamera(bounds, view, this.renderer.geometry,
        Math.floor(this.host.clientWidth * view.dpr), Math.floor(this.host.clientHeight * view.dpr));
      if (!camera) return;
      this.fitting = false;
      this.scale = camera.scale;
      this.requestedScroll = [camera.x * camera.scale / view.dpr, camera.y * camera.scale / view.dpr];
      this.refresh();
    }
    this.history.record(before, this.cameraState()!);
  }

  /** Restore the previous committed logical camera without adding a new entry. */
  zoomBack() {
    if (!this.options.zoom || this.disposed || this.renderer.state !== 'ready') return false;
    this.wheelGesture = null;
    this.pinch = null;
    const camera = this.history.pop();
    if (!camera) return false;
    this.fitting = false;
    this.scale = camera.scale;
    this.requestedScroll = [camera.x * camera.scale / window.devicePixelRatio,
      camera.y * camera.scale / window.devicePixelRatio];
    this.refresh();
    return true;
  }

  private focal(clientX: number, clientY: number) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: clientX - rect.left, y: clientY - rect.top };
  }

  private wheel = (event: WheelEvent) => {
    event.preventDefault();
    const focal = this.focal(event.clientX, event.clientY);
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? this.host.clientHeight : 1);
    const time = performance.now();
    if (!this.wheelGesture || time - this.wheelGesture.time > 180) this.wheelGesture = { token: {}, time };
    this.wheelGesture.time = time;
    this.zoom(this.scale * Math.exp(-Math.max(-500, Math.min(500, delta)) * 0.002), focal.x, focal.y, this.wheelGesture.token);
  };

  private gesture(event: TouchEvent) {
    if (event.touches.length !== 2) return null;
    const a = event.touches[0]!, b = event.touches[1]!;
    return { ...this.focal((a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2),
      distance: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY) };
  }
  private touchStart = (event: TouchEvent) => {
    const gesture = this.gesture(event);
    if (!gesture || gesture.distance === 0) return;
    event.preventDefault();
    this.wheelGesture = null;
    this.refresh();
    const view = this.renderer.view;
    if (view) this.pinch = { distance: gesture.distance, scale: this.scale,
      x: view.x + gesture.x * view.dpr / this.scale, y: view.y + gesture.y * view.dpr / this.scale };
  };
  private touchMove = (event: TouchEvent) => {
    const gesture = this.gesture(event), start = this.pinch;
    if (!gesture || !start) return;
    event.preventDefault();
    this.refresh();
    const before = this.cameraState();
    this.applyZoom(start.scale * gesture.distance / start.distance, gesture.x, gesture.y);
    // Moving the gesture midpoint follows the same original logical focal point.
    const dpr = window.devicePixelRatio;
    const current = this.gesture(event)!; // Centering may have moved the canvas.
    this.requestedScroll = [start.x * this.scale / dpr - current.x, start.y * this.scale / dpr - current.y];
    this.refresh();
    const after = this.cameraState();
    if (before && after) this.history.record(before, after, start);
  };
  private touchEnd = () => { this.pinch = null; };

  /** Call after an upload/transfer change; scrolling and DPR changes refresh automatically. */
  refresh() {
    if (this.disposed || !['ready', 'empty'].includes(this.renderer.state)) return;
    const dpr = window.devicePixelRatio;
    const { columns, rows, count } = this.renderer.geometry;
    if (this.options.zoom && this.fitting) {
      const next = fitWidthScale(columns, this.host.clientWidth, dpr);
      const old = this.renderer.view;
      if (old && next !== this.scale && !this.requestedScroll) this.requestedScroll = [old.x * next / dpr, old.y * next / dpr];
      this.scale = next;
    }
    this.extent.style.width = `${count ? columns * this.scale / dpr : 0}px`;
    this.extent.style.height = `${count ? rows * this.scale / dpr : 0}px`;
    if (this.requestedScroll) {
      [this.host.scrollLeft, this.host.scrollTop] = this.requestedScroll;
      this.requestedScroll = null;
    }
    // Browsers also have finite CSS scroll ranges. Fail explicitly instead of losing cells.
    const extentRect = this.extent.getBoundingClientRect();
    if (count && (Math.abs(extentRect.width * dpr - columns * this.scale) > 1 || Math.abs(extentRect.height * dpr - rows * this.scale) > 1)) {
      this.canvas.style.display = 'none';
      this.status.textContent = 'Tensor exceeds this browser’s native scroll extent. Exact rendering is unavailable.';
      return;
    }
    if (this.renderer.state === 'ready') {
      this.canvas.style.display = 'block';
      this.status.textContent = '';
    }
    const view = this.renderer.setView(this.host.clientWidth, this.host.clientHeight,
      this.host.scrollLeft, this.host.scrollTop, dpr, this.scale);
    const rect = this.host.getBoundingClientRect();
    const left = rect.left + this.host.clientLeft;
    const top = rect.top + this.host.clientTop;
    // Align the canvas on the physical screen too, including fractional host placement.
    this.surface.style.width = `${view.cssWidth}px`;
    this.surface.style.height = `${view.cssHeight}px`;
    const offsetX = this.options.zoom ? centeredOffset(this.host.clientWidth, view.width, dpr) : 0;
    const offsetY = this.options.zoom ? centeredOffset(this.host.clientHeight, view.height, dpr) : 0;
    // A smaller extent must also move the old absolute surface back inside it.
    // Otherwise the surface itself retains the old range and prevents native
    // scroll clamping after zoom/DPR changes at the far end.
    const scrollX = Math.min(this.host.scrollLeft, Math.max(0, Math.round(extentRect.width) - this.host.clientWidth));
    const scrollY = Math.min(this.host.scrollTop, Math.max(0, Math.round(extentRect.height) - this.host.clientHeight));
    this.surface.style.left = `${scrollX + offsetX + Math.round(left * dpr) / dpr - left}px`;
    this.surface.style.top = `${scrollY + offsetY + Math.round(top * dpr) / dpr - top}px`;
    this.canvas.dataset.origin = `${view.x},${view.y}`;
    this.renderer.draw();
    this.options.onViewChange?.(view);
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.history.clear();
    this.wheelGesture = null;
    this.pinch = null;
    cancelAnimationFrame(this.frame);
    this.observer.disconnect();
    this.media?.removeEventListener('change', this.dprChanged);
    window.removeEventListener('resize', this.dprChanged);
    this.host.removeEventListener('scroll', this.schedule);
    for (const target of this.zoomTargets) {
      target.removeEventListener('wheel', this.wheel);
      target.removeEventListener('touchstart', this.touchStart);
      target.removeEventListener('touchmove', this.touchMove);
      target.removeEventListener('touchend', this.touchEnd);
      target.removeEventListener('touchcancel', this.touchEnd);
    }
    this.zoomTargets.clear();
    this.renderer.dispose();
    this.host.replaceChildren();
    if (this.originalStyle === null) this.host.removeAttribute('style');
    else this.host.setAttribute('style', this.originalStyle);
  }
}
