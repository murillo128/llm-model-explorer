import { MatrixInspection } from './matrix-inspection';
import { MatrixZoomSelection } from './matrix-zoom-selection';
import { MatrixCameraNavigation } from './matrix-camera-navigation';
import type { Inspection } from './matrix-inspection';
import { DistributionRenderer } from './tensor-renderer';
import type { RendererOptions } from './tensor-renderer';
import { TensorViewport } from './tensor-viewport';
import type { TensorDescriptor, ViewGeometry } from './geometry';
import { DistributionScale } from './distribution-scale';
import { MatrixScrollbars } from './matrix-scrollbars';
import { MatrixRowSelection } from './matrix-row-selection';
import type { DistributionDomain } from './distribution-scale';
import './distribution-scale.css';

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
  private zoomSelection?: MatrixZoomSelection;
  private cameraNavigation?: MatrixCameraNavigation;
  private rowScale?: DistributionScale;
  private columnScale?: DistributionScale;
  private scrollbars?: MatrixScrollbars;
  private rowSelection?: MatrixRowSelection;
  private readonly observer: ResizeObserver;

  constructor(readonly host: HTMLElement, descriptor: TensorDescriptor, options: MatrixViewportOptions = {}) {
    if (host.childNodes.length) throw new Error('MatrixViewport requires an empty host.');
    this.originalStyle = host.getAttribute('style');
    host.classList.add('matrix-surfaces');
    Object.assign(host.style, { display: 'grid', position: 'relative', gap: '10px', alignItems: 'start', alignContent: 'start', minWidth: '0' });
    this.main.className = 'matrix-scroll';
    Object.assign(this.main.style, { gridColumn: '1', gridRow: '1', minWidth: '0', minHeight: descriptor.rank === 2 ? '0' : '18px', height: descriptor.rank === 2 ? 'var(--matrix-height)' : 'auto', overflowY: 'auto', overscrollBehavior: 'contain' });
    this.main.setAttribute('role', 'region');
    this.main.setAttribute('aria-label', 'Tensor matrix; scroll to inspect all values');
    this.main.tabIndex = 0;
    this.rowHost.className = 'row-distributions';
    this.columnHost.className = 'column-distributions';
    host.append(this.main);
    // Rank-1 keeps its exact strip within a small usable scroll-control viewport.
    const hostMaxHeight = getComputedStyle(this.main).maxHeight;
    this.main.style.maxHeight = hostMaxHeight === 'none' ? 'var(--matrix-height)' : `min(var(--matrix-height), ${hostMaxHeight})`;
    try {
      if (descriptor.rank === 2 && descriptor.numel > 0 && options.distributions !== false) {
        for (const [panel, label] of [[this.rowHost, 'Row distributions'], [this.columnHost, 'Column distributions']] as const) {
          const canvas = document.createElement('canvas');
          canvas.setAttribute('aria-label', label);
          Object.assign(panel.style, { position: 'absolute', overflow: 'hidden' });
          Object.assign(canvas.style, { position: 'absolute', display: 'block' });
          panel.append(canvas);
          host.append(panel);
        }
        this.rows = new DistributionRenderer(this.rowHost.firstChild as HTMLCanvasElement, [descriptor.shape[0]!, 100], descriptor.shape[1]!, options);
        this.columns = new DistributionRenderer(this.columnHost.firstChild as HTMLCanvasElement, [100, descriptor.shape[1]!], descriptor.shape[0]!, options);
        this.rowScale = new DistributionScale('rows', this.rowHost);
        this.columnScale = new DistributionScale('columns', this.columnHost);
        host.append(this.rowScale.ruler, this.columnScale.ruler);
        this.main.style.gridRow = '2';
        for (const ruler of [this.rowScale.ruler, this.columnScale.ruler]) {
          Object.assign(ruler.style, { position: 'absolute', gridColumn: 'auto', gridRow: 'auto' });
        }
      }
      const layout = () => {
        const dpr = window.devicePixelRatio;
        // The host receives the pane's remaining content height. Reserve profile
        // depth and gap here; TensorViewport still measures the actual client box.
        const depth = this.rows ? 100 / dpr : 0;
        host.style.setProperty('--distribution-depth', `${depth}px`);
        host.style.setProperty('--distribution-pixel', `${1 / dpr}px`);
        const gap = this.rows ? parseFloat(getComputedStyle(host).rowGap) : 0;
        const scaleHeight = this.rowScale ? this.rowScale.ruler.offsetHeight + gap : 0;
        host.style.setProperty('--matrix-height', `${Math.max(0, host.clientHeight - depth - gap - scaleHeight)}px`);
        host.style.gridTemplateColumns = `${descriptor.rank === 2 ? 'minmax(0, 1fr)' : `minmax(0, ${descriptor.shape.at(-1)! / dpr}px)`}${this.rows ? ` ${depth}px` : ''}`;
        host.style.gridTemplateRows = this.rows ? `${this.rowScale!.ruler.offsetHeight}px var(--matrix-height) ${depth}px` : 'var(--matrix-height)';
      };
      layout();
      this.matrix = new TensorViewport(this.main, descriptor, { ...options, zoom: descriptor.rank === 2, onStateChange: (state) => {
        if (state !== 'ready') { this.zoomSelection?.cancel(); this.inspection?.clear(); this.rowSelection?.refresh(); }
        options.onStateChange?.(state);
      }, onViewChange: (view) => {
        layout();
        this.align(view);
        this.rowSelection?.refresh();
        this.scrollbars?.refresh();
        this.zoomSelection?.refresh();
        this.inspection?.refresh();
      } });
      this.scrollbars = new MatrixScrollbars(this.main, descriptor.rank === 1 ? this.main : host);
      this.matrix.attachOverlay(this.scrollbars.element);
      if (descriptor.rank === 2) this.rowSelection = new MatrixRowSelection(this);
      if (descriptor.rank === 2 && options.onInspection) this.inspection = new MatrixInspection(this, options.onInspection);
      if (descriptor.rank === 2 && descriptor.numel > 0) this.zoomSelection = new MatrixZoomSelection(this,
        (active) => this.inspection?.suspend(active));
      if (descriptor.rank === 2 && descriptor.numel > 0) this.cameraNavigation = new MatrixCameraNavigation(this);
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
    // Fixed containers occupy the viewer edges. Only their scientific canvases
    // follow the matrix's device-snapped presentation offsets on the shared axis.
    // During TensorViewport construction its callback precedes field assignment.
    const canvas = this.main.querySelector('canvas')!;
    const matrix = canvas.getBoundingClientRect();
    const viewport = this.main.getBoundingClientRect();
    const pane = this.host.getBoundingClientRect();
    const gap = parseFloat(getComputedStyle(this.host).columnGap);
    const rowGap = parseFloat(getComputedStyle(this.host).rowGap);
    const left = viewport.left - pane.left - this.host.clientLeft;
    const top = viewport.top - pane.top - this.host.clientTop;
    const right = left + this.main.clientWidth + gap;
    const bottom = top + this.main.clientHeight + rowGap;
    Object.assign(this.rowHost.style, { left: `${right}px`, top: `${top}px`, width: `${100 / view.dpr}px`, height: `${this.main.clientHeight}px` });
    Object.assign(this.columnHost.style, { left: `${left}px`, top: `${bottom}px`, width: `${this.main.clientWidth}px`, height: `${100 / view.dpr}px` });
    Object.assign(this.rowScale!.ruler.style, { left: `${right}px`, top: `${top - rowGap - this.rowScale!.ruler.offsetHeight}px`, width: `${100 / view.dpr}px` });
    Object.assign(this.columnScale!.ruler.style, { left: `${right}px`, top: `${bottom}px`, width: `${100 / view.dpr}px` });
    const pairs = [
      [this.rows, this.rowHost, 100 / view.dpr, view.cssHeight, 0, view.y * view.scaleY / view.dpr, 1, view.scaleY],
      [this.columns, this.columnHost, view.cssWidth, 100 / view.dpr, view.x * view.scaleX / view.dpr, 0, view.scaleX, 1],
    ] as const;
    for (const [renderer, host, width, height, x, y, scaleX, scaleY] of pairs) {
      if (renderer.state !== 'ready') continue;
      const rect = host.getBoundingClientRect();
      renderer.canvas.style.left = `${(renderer === this.columns ? matrix.left : Math.round(rect.left * view.dpr) / view.dpr) - rect.left}px`;
      renderer.canvas.style.top = `${(renderer === this.rows ? matrix.top : Math.round(rect.top * view.dpr) / view.dpr) - rect.top}px`;
      if (renderer === this.rows) {
        this.rowScale?.alignBinAxis(renderer.canvas.style.left);
        this.rowScale!.ruler.style.left = `${right + parseFloat(renderer.canvas.style.left)}px`;
      } else {
        this.columnScale?.alignBinAxis(renderer.canvas.style.top);
        this.columnScale!.ruler.style.top = `${bottom + parseFloat(renderer.canvas.style.top)}px`;
      }
      const panelView = renderer.setView(width, height, x, y, view.dpr, scaleX, scaleY);
      // Zero references cover real data only, leaving underfill margins empty.
      const guide = (renderer === this.rows ? this.rowScale : this.columnScale)!.guide;
      if (renderer === this.rows) Object.assign(guide.style, { top: renderer.canvas.style.top, bottom: 'auto', height: `${height}px` });
      else Object.assign(guide.style, { left: renderer.canvas.style.left, right: 'auto', width: `${width}px` });
      renderer.canvas.dataset.origin = `${panelView.x},${panelView.y}`;
      renderer.draw();
    }
  }

  refresh() { this.matrix.refresh(); }
  fitWidth() { this.matrix.fitWidth(); }
  revealRow(row: number) { this.matrix.revealRow(row); }
  setDistributionDomain(domain: DistributionDomain) {
    this.rowScale?.setDomain(domain);
    this.columnScale?.setDomain(domain);
  }
  setLinkedRow(row: number | null) { this.inspection?.setLinkedRow(row); }
  setSelectedRows(rows?: readonly number[]) { this.rowSelection?.setRows(rows); }

  private restoreHost() {
    this.host.classList.remove('matrix-surfaces');
    if (this.originalStyle === null) this.host.removeAttribute('style');
    else this.host.setAttribute('style', this.originalStyle);
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.observer.disconnect();
    this.zoomSelection?.dispose();
    this.cameraNavigation?.dispose();
    this.inspection?.dispose();
    this.rowSelection?.dispose();
    this.scrollbars?.dispose();
    this.matrix.dispose();
    this.rows?.dispose();
    this.columns?.dispose();
    this.host.replaceChildren();
    this.restoreHost();
  }
}
