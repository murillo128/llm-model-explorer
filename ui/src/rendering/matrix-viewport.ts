import { MatrixInspection } from './matrix-inspection';
import type { Inspection } from './matrix-inspection';
import { DistributionRenderer } from './tensor-renderer';
import type { RendererOptions } from './tensor-renderer';
import { TensorViewport } from './tensor-viewport';
import type { TensorDescriptor, ViewGeometry } from './geometry';

export interface MatrixViewportOptions extends RendererOptions {
  readonly distributions?: boolean;
  readonly onInspection?: (inspection: Inspection | null) => void;
}

/** Three exact surfaces sharing native scroll state. Supply an empty host with a bounded content height. */
export class MatrixViewport {
  readonly matrix: TensorViewport;
  readonly rows?: DistributionRenderer;
  readonly columns?: DistributionRenderer;
  private readonly main = document.createElement('div');
  private readonly rowHost = document.createElement('div');
  private readonly columnHost = document.createElement('div');
  private readonly originalStyle: string | null;
  private disposed = false;
  private inspection?: MatrixInspection;
  private readonly observer: ResizeObserver;

  constructor(readonly host: HTMLElement, descriptor: TensorDescriptor, options: MatrixViewportOptions = {}) {
    if (host.childNodes.length) throw new Error('MatrixViewport requires an empty host.');
    this.originalStyle = host.getAttribute('style');
    host.classList.add('matrix-surfaces');
    Object.assign(host.style, { display: 'grid', gap: '10px', alignItems: 'start', alignContent: 'start', minWidth: '0' });
    this.main.className = 'matrix-scroll';
    Object.assign(this.main.style, { gridColumn: '1', gridRow: '1', minWidth: '0', minHeight: '0', height: 'auto', overflowY: 'scroll', overscrollBehavior: 'contain' });
    this.main.setAttribute('role', 'region');
    this.main.setAttribute('aria-label', 'Tensor matrix; scroll to inspect all values');
    this.main.tabIndex = 0;
    this.rowHost.className = 'row-distributions';
    this.columnHost.className = 'column-distributions';
    Object.assign(this.rowHost.style, { gridColumn: '2', gridRow: '1' });
    Object.assign(this.columnHost.style, { gridColumn: '1', gridRow: '2' });
    host.append(this.main);
    // Auto height includes native horizontal scrollbar chrome in addition to the
    // intrinsic data extent. A fixed data-height border box can hide a short strip.
    const hostMaxHeight = getComputedStyle(this.main).maxHeight;
    this.main.style.maxHeight = hostMaxHeight === 'none' ? 'var(--matrix-height)' : `min(var(--matrix-height), ${hostMaxHeight})`;
    try {
      if (descriptor.rank === 2 && descriptor.numel > 0 && options.distributions !== false) {
        for (const [panel, label] of [[this.rowHost, 'Row distributions'], [this.columnHost, 'Column distributions']] as const) {
          const canvas = document.createElement('canvas');
          canvas.setAttribute('aria-label', label);
          Object.assign(panel.style, { position: 'relative', overflow: 'hidden' });
          Object.assign(canvas.style, { position: 'absolute', display: 'block' });
          panel.append(canvas);
          host.append(panel);
        }
        this.rows = new DistributionRenderer(this.rowHost.firstChild as HTMLCanvasElement, [descriptor.shape[0]!, 100], descriptor.shape[1]!, options);
        this.columns = new DistributionRenderer(this.columnHost.firstChild as HTMLCanvasElement, [100, descriptor.shape[1]!], descriptor.shape[0]!, options);
      }
      const layout = () => {
        const dpr = window.devicePixelRatio;
        // The host receives the pane's remaining content height. Reserve profile
        // depth and gap here; TensorViewport still measures the actual client box.
        const depth = this.rows ? 100 / dpr : 0;
        const gap = this.rows ? parseFloat(getComputedStyle(host).rowGap) : 0;
        host.style.setProperty('--matrix-height', `${Math.max(0, host.clientHeight - depth - gap)}px`);
        // The vertical scrollbar reserves space even without overflow. Include it in
        // the native-width track so a tall, otherwise fitting tensor stays fitting.
        const gutter = this.main.offsetWidth - this.main.clientWidth;
        host.style.gridTemplateColumns = `minmax(0, ${descriptor.shape.at(-1)! / dpr + gutter}px)${this.rows ? ` ${depth}px` : ''}`;
      };
      layout();
      this.matrix = new TensorViewport(this.main, descriptor, { ...options, onStateChange: (state) => {
        if (state !== 'ready') this.inspection?.clear();
        options.onStateChange?.(state);
      }, onViewChange: (view) => {
        layout();
        this.align(view);
        this.inspection?.refresh();
      } });
      if (descriptor.rank === 2 && options.onInspection) this.inspection = new MatrixInspection(this, options.onInspection);
      this.observer = new ResizeObserver(() => { layout(); this.matrix.refresh(); });
      this.observer.observe(host);
    } catch (error) {
      this.rows?.dispose();
      this.columns?.dispose();
      host.replaceChildren();
      this.restoreHost();
      throw error;
    }
  }

  private align(view: ViewGeometry) {
    if (!this.rows || !this.columns) return;
    const pairs = [
      [this.rows, this.rowHost, 100 / view.dpr, view.cssHeight, 0, view.y / view.dpr],
      [this.columns, this.columnHost, view.cssWidth, 100 / view.dpr, view.x / view.dpr, 0],
    ] as const;
    for (const [renderer, host, width, height, x, y] of pairs) {
      host.style.width = `${width}px`;
      host.style.height = `${height}px`;
      if (renderer.state !== 'ready') continue;
      const rect = host.getBoundingClientRect();
      renderer.canvas.style.left = `${Math.round(rect.left * view.dpr) / view.dpr - rect.left}px`;
      renderer.canvas.style.top = `${Math.round(rect.top * view.dpr) / view.dpr - rect.top}px`;
      const panelView = renderer.setView(width, height, x, y, view.dpr);
      renderer.canvas.dataset.origin = `${panelView.x},${panelView.y}`;
      renderer.draw();
    }
  }

  refresh() { this.matrix.refresh(); }
  setLinkedRow(row: number | null) { this.inspection?.setLinkedRow(row); }

  private restoreHost() {
    this.host.classList.remove('matrix-surfaces');
    if (this.originalStyle === null) this.host.removeAttribute('style');
    else this.host.setAttribute('style', this.originalStyle);
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.observer.disconnect();
    this.inspection?.dispose();
    this.matrix.dispose();
    this.rows?.dispose();
    this.columns?.dispose();
    this.host.replaceChildren();
    this.restoreHost();
  }
}
