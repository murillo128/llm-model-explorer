import { rasterEdge } from './geometry';
import type { ViewGeometry } from './geometry';
import type { GridRenderer } from './tensor-renderer';
import type { MatrixViewport } from './matrix-viewport';
import { selectionBoundary } from './zoom-selection-geometry';
import type { ZoomBounds } from './zoom-selection-geometry';

type Surface = { renderer: GridRenderer; axis?: 'rows' | 'columns' };
type Gesture = { surface: Surface; id: number; x: number; y: number; view: ViewGeometry;
  startColumn: number; startRow: number; bounds: ZoomBounds; active: boolean };

/** Transient view navigation; never touches scalar/count storage. */
export class MatrixZoomSelection {
  private readonly surfaces: Surface[];
  private gesture: Gesture | null = null;
  private readonly touches = new Set<number>();
  private readonly overlay = document.createElement('div');
  private suppressClick = false;
  private readonly previousTouch: string[];

  constructor(private readonly viewport: MatrixViewport, private readonly suspendInspection: (active: boolean) => void) {
    this.surfaces = [{ renderer: viewport.matrix.renderer },
      ...(viewport.rows ? [{ renderer: viewport.rows, axis: 'rows' as const }] : []),
      ...(viewport.columns ? [{ renderer: viewport.columns, axis: 'columns' as const }] : [])];
    this.overlay.className = 'matrix-zoom-preview';
    this.overlay.setAttribute('aria-hidden', 'true');
    Object.assign(this.overlay.style, { position: 'absolute', pointerEvents: 'none', boxSizing: 'border-box',
      border: '1px solid #f59a38', background: 'rgba(245, 154, 56, 0.08)', zIndex: '1' });
    this.previousTouch = this.surfaces.map(({ renderer }) => renderer.canvas.style.touchAction);
    for (const { renderer } of this.surfaces) {
      // Keep native scrollbars available; a one-finger data-surface drag is selection.
      // Touch events still reach TensorViewport's two-finger pinch handler.
      renderer.canvas.style.touchAction = 'none';
    }
    const host = viewport.host;
    host.addEventListener('pointerdown', this.down, true);
    host.addEventListener('pointermove', this.move, true);
    host.addEventListener('pointerup', this.up, true);
    host.addEventListener('pointercancel', this.cancelPointer, true);
    host.addEventListener('lostpointercapture', this.cancelPointer, true);
    host.addEventListener('click', this.click, true);
    host.addEventListener('wheel', this.cancel, true);
    host.addEventListener('webglcontextlost', this.cancel, true);
    window.addEventListener('pointerdown', this.trackTouch, true);
    window.addEventListener('pointerup', this.releaseTouch, true);
    window.addEventListener('pointercancel', this.releaseTouch, true);
    window.addEventListener('keydown', this.key, true);
    window.addEventListener('blur', this.reset);
  }

  private trackTouch = (event: PointerEvent) => {
    if (event.pointerType !== 'touch') return;
    this.touches.add(event.pointerId);
    if (this.touches.size > 1) this.cancel();
  };
  private releaseTouch = (event: PointerEvent) => { this.touches.delete(event.pointerId); };
  private reset = () => { this.cancel(); this.touches.clear(); };
  private down = (event: PointerEvent) => {
    if (this.gesture || event.button !== 0 || this.touches.size > 1) return;
    this.suppressClick = false;
    const surface = this.surfaces.find(({ renderer }) => event.target === renderer.canvas);
    if (!surface || surface.renderer.state !== 'ready') return;
    this.viewport.refresh();
    const view = surface.renderer.view;
    if (!view) return;
    const rect = surface.renderer.canvas.getBoundingClientRect();
    if (!surface.renderer.cellAt(event.clientX - rect.left, event.clientY - rect.top)) return;
    const { columns, rows } = this.viewport.matrix.renderer.geometry;
    this.gesture = { surface, id: event.pointerId, x: event.clientX, y: event.clientY, view,
      startColumn: selectionBoundary(view, event.clientX - rect.left, 'columns', columns),
      startRow: selectionBoundary(view, event.clientY - rect.top, 'rows', rows), bounds: {}, active: false };
    surface.renderer.canvas.setPointerCapture(event.pointerId);
  };
  private move = (event: PointerEvent) => {
    const g = this.gesture;
    if (!g || g.id !== event.pointerId) return;
    const dx = event.clientX - g.x, dy = event.clientY - g.y;
    const distance = g.surface.axis === 'columns' ? Math.abs(dx) : g.surface.axis === 'rows' ? Math.abs(dy) : Math.hypot(dx, dy);
    if (!g.active && distance < 5) return;
    if (!g.active) {
      g.active = true;
      this.suppressClick = true;
      this.suspendInspection(true);
      g.surface.renderer.canvas.parentElement!.append(this.overlay);
    }
    event.preventDefault();
    event.stopPropagation();
    const rect = g.surface.renderer.canvas.getBoundingClientRect();
    const { columns, rows } = this.viewport.matrix.renderer.geometry;
    const column = selectionBoundary(g.view, event.clientX - rect.left, 'columns', columns);
    const row = selectionBoundary(g.view, event.clientY - rect.top, 'rows', rows);
    g.bounds = {
      ...(g.surface.axis !== 'rows' ? { columns: [Math.min(column, g.startColumn), Math.max(column, g.startColumn)] as const } : {}),
      ...(g.surface.axis !== 'columns' ? { rows: [Math.min(row, g.startRow), Math.max(row, g.startRow)] as const } : {}),
    };
    this.overlay.dataset.bounds = JSON.stringify(g.bounds);
    const v = g.view;
    const left = g.bounds.columns ? rasterEdge(g.bounds.columns[0], v.x, v.scaleX) / v.dpr : 0;
    const right = g.bounds.columns ? rasterEdge(g.bounds.columns[1], v.x, v.scaleX) / v.dpr : v.cssWidth;
    const top = g.bounds.rows ? rasterEdge(g.bounds.rows[0], v.y, v.scaleY) / v.dpr : 0;
    const bottom = g.bounds.rows ? rasterEdge(g.bounds.rows[1], v.y, v.scaleY) / v.dpr : v.cssHeight;
    Object.assign(this.overlay.style, { left: `${parseFloat(g.surface.renderer.canvas.style.left || '0') + left}px`,
      top: `${parseFloat(g.surface.renderer.canvas.style.top || '0') + top}px`,
      width: `${right - left}px`, height: `${bottom - top}px` });
  };
  private up = (event: PointerEvent) => {
    if (this.gesture?.id !== event.pointerId) return;
    this.move(event); // Include the final release position even without a last move.
    const g = this.gesture;
    this.cancel();
    if (g?.active) {
      event.preventDefault(); event.stopPropagation();
      this.viewport.matrix.zoomToBounds(g.bounds);
    }
  };
  private click = (event: MouseEvent) => {
    if (!this.suppressClick) return;
    this.suppressClick = false;
    event.preventDefault(); event.stopPropagation();
  };
  private cancelPointer = (event: PointerEvent) => { if (this.gesture?.id === event.pointerId) this.cancel(); };
  private key = (event: KeyboardEvent) => {
    if (event.key !== 'Escape' || !this.gesture) return;
    event.preventDefault(); event.stopPropagation(); this.cancel();
  };
  cancel = () => {
    const g = this.gesture;
    this.gesture = null;
    this.overlay.remove();
    if (g?.active) this.suspendInspection(false);
    if (g && g.surface.renderer.canvas.hasPointerCapture(g.id)) g.surface.renderer.canvas.releasePointerCapture(g.id);
  };

  /** A changed camera invalidates a gesture; progressive uploads alone do not. */
  refresh() {
    const g = this.gesture, view = g?.surface.renderer.view;
    if (g && (!view || (['x', 'y', 'scaleX', 'scaleY', 'dpr', 'width', 'height'] as const)
      .some((key) => view[key] !== g.view[key]))) this.cancel();
  }

  dispose() {
    this.reset();
    const host = this.viewport.host;
    host.removeEventListener('pointerdown', this.down, true);
    host.removeEventListener('pointermove', this.move, true);
    host.removeEventListener('pointerup', this.up, true);
    host.removeEventListener('pointercancel', this.cancelPointer, true);
    host.removeEventListener('lostpointercapture', this.cancelPointer, true);
    host.removeEventListener('click', this.click, true);
    host.removeEventListener('wheel', this.cancel, true);
    host.removeEventListener('webglcontextlost', this.cancel, true);
    window.removeEventListener('pointerdown', this.trackTouch, true);
    window.removeEventListener('pointerup', this.releaseTouch, true);
    window.removeEventListener('pointercancel', this.releaseTouch, true);
    window.removeEventListener('keydown', this.key, true);
    window.removeEventListener('blur', this.reset);
    this.surfaces.forEach(({ renderer }, i) => { renderer.canvas.style.touchAction = this.previousTouch[i]!; });
  }
}
