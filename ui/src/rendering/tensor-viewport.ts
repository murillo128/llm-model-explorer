import { TensorRenderer } from './tensor-renderer';
import type { RendererOptions } from './tensor-renderer';
import type { TensorDescriptor, ViewGeometry } from './geometry';

export interface ViewportOptions extends RendererOptions {
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
  private media: MediaQueryList | null = null;
  private frame = 0;
  private disposed = false;
  private readonly originalStyle: string | null;

  constructor(readonly host: HTMLElement, descriptor: TensorDescriptor, private readonly options: ViewportOptions = {}) {
    if (host.childNodes.length) throw new Error('TensorViewport requires an empty host.');
    this.originalStyle = host.getAttribute('style');
    Object.assign(host.style, { overflow: 'auto', position: 'relative', padding: '0' });
    Object.assign(this.extent.style, { position: 'relative', pointerEvents: 'none' });
    Object.assign(this.canvas.style, { position: 'absolute', display: 'block' });
    // Clip the canvas's untransformed layout box so it cannot enlarge native scroll.
    Object.assign(this.surface.style, { position: 'absolute', overflow: 'hidden' });
    Object.assign(this.status.style, { position: 'sticky', left: '0', top: '0' });
    this.status.setAttribute('role', 'status');
    this.canvas.setAttribute('aria-label', 'Tensor data; one weight per device pixel');
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
    const hostMaxHeight = getComputedStyle(host).maxHeight;
    this.setCeilings = () => {
      const width = `${maxWidth / window.devicePixelRatio}px`;
      const height = `${maxHeight / window.devicePixelRatio}px`;
      host.style.maxWidth = hostMaxWidth === 'none' ? width : `min(${width}, ${hostMaxWidth})`;
      host.style.maxHeight = hostMaxHeight === 'none' ? height : `min(${height}, ${hostMaxHeight})`;
    };
    this.observer = new ResizeObserver(this.schedule);
    this.observer.observe(host);
    host.addEventListener('scroll', this.schedule);
    window.addEventListener('resize', this.dprChanged);
    try { this.dprChanged(); } catch (error) { this.dispose(); throw error; }
  }

  private readonly setCeilings: () => void;

  private dprChanged = () => {
    this.media?.removeEventListener('change', this.dprChanged);
    this.media = matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    this.media.addEventListener('change', this.dprChanged);
    this.setCeilings();
    this.refresh();
  };

  private schedule = () => {
    if (!this.frame) this.frame = requestAnimationFrame(() => { this.frame = 0; this.refresh(); });
  };

  /** Call after an upload/transfer change; scrolling and DPR changes refresh automatically. */
  refresh() {
    if (this.disposed || !['ready', 'empty'].includes(this.renderer.state)) return;
    const dpr = window.devicePixelRatio;
    const { columns, rows, count } = this.renderer.geometry;
    this.extent.style.width = `${count ? columns / dpr : 0}px`;
    this.extent.style.height = `${count ? rows / dpr : 0}px`;
    // Browsers also have finite CSS scroll ranges. Fail explicitly instead of losing cells.
    const extentRect = this.extent.getBoundingClientRect();
    if (count && (Math.abs(extentRect.width * dpr - columns) > 1 || Math.abs(extentRect.height * dpr - rows) > 1)) {
      this.canvas.style.display = 'none';
      this.status.textContent = 'Tensor exceeds this browser’s native scroll extent. Exact rendering is unavailable.';
      return;
    }
    if (this.renderer.state === 'ready') {
      this.canvas.style.display = 'block';
      this.status.textContent = '';
    }
    const view = this.renderer.setView(this.host.clientWidth, this.host.clientHeight,
      this.host.scrollLeft, this.host.scrollTop, dpr);
    const rect = this.host.getBoundingClientRect();
    const left = rect.left + this.host.clientLeft;
    const top = rect.top + this.host.clientTop;
    // Align the canvas on the physical screen too, including fractional host placement.
    this.surface.style.width = `${view.cssWidth}px`;
    this.surface.style.height = `${view.cssHeight}px`;
    this.surface.style.left = `${this.host.scrollLeft + Math.round(left * dpr) / dpr - left}px`;
    this.surface.style.top = `${this.host.scrollTop + Math.round(top * dpr) / dpr - top}px`;
    this.canvas.dataset.origin = `${view.x},${view.y}`;
    this.renderer.draw();
    this.options.onViewChange?.(view);
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    cancelAnimationFrame(this.frame);
    this.observer.disconnect();
    this.media?.removeEventListener('change', this.dprChanged);
    window.removeEventListener('resize', this.dprChanged);
    this.host.removeEventListener('scroll', this.schedule);
    this.renderer.dispose();
    this.host.replaceChildren();
    if (this.originalStyle === null) this.host.removeAttribute('style');
    else this.host.setAttribute('style', this.originalStyle);
  }
}
