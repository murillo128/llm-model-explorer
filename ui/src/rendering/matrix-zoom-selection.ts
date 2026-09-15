import { rasterEdge } from './geometry';
import type { ViewGeometry } from './geometry';
import type { GridRenderer } from './tensor-renderer';
import type { MatrixViewport } from './matrix-viewport';
import { selectionBoundary } from './zoom-selection-geometry';
import type { ZoomBounds } from './zoom-selection-geometry';

type Surface = { renderer: GridRenderer; axis?: 'rows' | 'columns'; overlay: HTMLDivElement };
type Gesture = { surface: Surface; id: number; x: number; y: number; view: ViewGeometry;
  startColumn: number; startRow: number; bounds: ZoomBounds; active: boolean };

/** Transient view navigation; never touches scalar/count storage. */
export class MatrixZoomSelection {
  private readonly surfaces: Surface[];
  private gesture: Gesture | null = null;
  private readonly touches = new Set<number>();
  private suppressClick = false;
  private readonly previousTouch: string[];

  constructor(private readonly viewport: MatrixViewport, private readonly suspendInspection: (active: boolean) => void) {
    this.surfaces = ([{ renderer: viewport.matrix.renderer },
      ...(viewport.rows ? [{ renderer: viewport.rows, axis: 'rows' as const }] : []),
      ...(viewport.columns ? [{ renderer: viewport.columns, axis: 'columns' as const }] : [])] as Omit<Surface, 'overlay'>[])
      .map((surface) => {
        const overlay = document.createElement('div');
        overlay.className = 'matrix-zoom-preview';
        overlay.dataset.surface = surface.axis ?? 'matrix';
        overlay.setAttribute('aria-hidden', 'true');
        Object.assign(overlay.style, { position: 'absolute', pointerEvents: 'none', boxSizing: 'border-box', zIndex: '1' });
        return { ...surface, overlay };
      });
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
    this.preview(g.bounds);
  };

  private preview(bounds: ZoomBounds) {
    for (const { renderer, axis, overlay } of this.surfaces) {
      const v = renderer.view;
      const projected = axis ? bounds[axis] && { [axis]: bounds[axis] } : bounds;
      if (!v || renderer.state !== 'ready' || !projected) { overlay.remove(); continue; }
      const span = (range: readonly [number, number] | undefined, origin: number, scale: number, extent: number) => {
        if (!range) return [0, extent] as const;
        let start = rasterEdge(range[0], origin, scale), end = rasterEdge(range[1], origin, scale);
        // Match the exact centered, one-device-pixel single-cell inspection guide.
        if (range[1] - range[0] === 1) { start = Math.floor((start + end) / 2); end = start + 1; }
        return [Math.max(0, Math.min(extent, start)), Math.max(0, Math.min(extent, end))] as const;
      };
      const [left, right] = span(projected.columns, v.x, v.scaleX, v.width);
      const [top, bottom] = span(projected.rows, v.y, v.scaleY, v.height);
      const thin = Object.values(projected).some((range) => range[1] - range[0] === 1);
      overlay.dataset.bounds = JSON.stringify(projected);
      Object.assign(overlay.style, {
        left: `${parseFloat(renderer.canvas.style.left || '0') + left / v.dpr}px`,
        top: `${parseFloat(renderer.canvas.style.top || '0') + top / v.dpr}px`,
        width: `${(right - left) / v.dpr}px`, height: `${(bottom - top) / v.dpr}px`,
        border: thin ? 'none' : `${1 / v.dpr}px solid #f59a38`,
        background: thin ? 'rgba(245, 154, 56, 0.65)' : 'rgba(245, 154, 56, 0.08)',
      });
      renderer.canvas.parentElement!.append(overlay);
    }
  }
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
    event.preventDefault(); event.stopImmediatePropagation(); this.cancel();
  };
  cancel = () => {
    const g = this.gesture;
    this.gesture = null;
    this.surfaces.forEach(({ overlay }) => overlay.remove());
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
