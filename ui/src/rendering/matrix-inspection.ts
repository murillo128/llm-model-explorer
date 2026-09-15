import type { MatrixViewport } from './matrix-viewport';
import type { Selection } from './chroma';
import { INSPECTION, inspectionPosition, magnifierVisible } from './inspection-layout';

export function formatFloat32(value: number) {
  if (Object.is(value, -0)) return '-0';
  return String(value); // JS shortest round-trip decimal also round-trips its float32 source.
}

export interface Inspection extends Selection {
  readonly value: string;
  readonly left: number;
  readonly top: number;
  readonly magnifier: boolean;
  readonly width: number;
  readonly draw: (canvas: HTMLCanvasElement) => void;
  readonly onReadoutResize?: (height: number) => void;
}

/** Native-coordinate interaction, independent of React and transport. */
export class MatrixInspection {
  private pointer: { x: number; y: number } | null = null;
  private cell: Selection | null = null;
  private linkedRow: number | null = null;
  private disposed = false;
  private suspended = false;
  private showMagnifier = true;
  private layoutKey = '';
  private animationFrame: number | undefined;
  private measuredReadout: { key: string; height: number } | undefined;
  private readoutKey = '';
  constructor(private readonly viewport: MatrixViewport, private readonly changed: (value: Inspection | null) => void) {
    const { canvas, host } = viewport.matrix;
    canvas.classList.add('matrix-inspectable');
    canvas.addEventListener('pointermove', this.move);
    canvas.addEventListener('click', this.click);
    canvas.addEventListener('pointerleave', this.leave);
    host.addEventListener('focus', this.focus);
    host.addEventListener('blur', this.leave);
    host.addEventListener('keydown', this.key);
    window.addEventListener('scroll', this.windowScroll, true);
  }

  private click = (event: MouseEvent) => { this.pointer = { x: event.clientX, y: event.clientY }; this.refresh(); };
  private move = (event: PointerEvent) => {
    this.pointer = { x: event.clientX, y: event.clientY };
    this.refresh();
  };
  private windowScroll = (event: Event) => {
    // The native viewport's own refresh resolves its updated scroll origin.
    if (event.target !== this.viewport.matrix.host) this.refresh();
  };
  private focus = () => {
    this.pointer = null;
    const view = this.viewport.matrix.renderer.view;
    if (view) { this.cell = { row: Math.floor(view.y), column: Math.floor(view.x) }; this.refresh(); }
  };
  private key = (event: KeyboardEvent) => {
    const step = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[event.key];
    if (!step) return;
    event.preventDefault();
    const { renderer, host } = this.viewport.matrix;
    const view = renderer.view;
    if (!view) return;
    this.pointer = null;
    const cell = this.cell ?? { row: Math.floor(view.y), column: Math.floor(view.x) };
    this.cell = { row: Math.max(0, Math.min(renderer.geometry.rows - 1, cell.row + step[0]!)),
      column: Math.max(0, Math.min(renderer.geometry.columns - 1, cell.column + step[1]!)) };
    if (this.cell.column < view.x) host.scrollLeft = this.cell.column * view.scaleX / view.dpr;
    if (this.cell.column >= view.x + view.width / view.scaleX) host.scrollLeft = ((this.cell.column + 1) * view.scaleX - view.width) / view.dpr;
    if (this.cell.row < view.y) host.scrollTop = this.cell.row * view.scaleY / view.dpr;
    if (this.cell.row >= view.y + view.height / view.scaleY) host.scrollTop = ((this.cell.row + 1) * view.scaleY - view.height) / view.dpr;
    this.viewport.refresh();
  };

  private select(cell: Selection | null) {
    const { matrix, rows, columns } = this.viewport;
    cell ??= this.linkedRow === null ? null : { row: this.linkedRow, column: -1 };
    matrix.renderer.setSelection(cell);
    rows?.setSelection(cell ? { row: cell.row, column: -1 } : null);
    columns?.setSelection(cell ? { row: -1, column: cell.column } : null);
    // Pointer leave/blur may precede the DPR resize notification. Keep selection
    // current, but let the viewport refresh draw once its geometry matches.
    for (const r of [matrix.renderer, rows, columns]) {
      if (r?.state === 'ready' && r.view?.dpr === window.devicePixelRatio) r.draw();
    }
  }

  /** External row context is display-only; it never moves focus or invents a cell. */
  setLinkedRow(row: number | null) {
    this.linkedRow = row !== null && Number.isInteger(row) && row >= 0 && row < this.viewport.matrix.renderer.geometry.rows ? row : null;
    // A new external interaction supersedes prior local inspection, even when
    // the matrix retains keyboard focus. Clear its readout, not browser focus.
    if (this.linkedRow !== null) {
      this.pointer = null;
      this.cell = null;
    }
    this.refresh();
  }

  private layout() {
    const rect = this.viewport.matrix.canvas.getBoundingClientRect();
    const pane = this.viewport.host.getBoundingClientRect();
    const panels = [this.viewport.rows, this.viewport.columns].flatMap((panel) =>
      panel ? [panel.canvas.getBoundingClientRect()] : []);
    return { rect, pane, panels, key: [window.innerWidth, window.innerHeight, window.devicePixelRatio,
      ...[rect, pane, ...panels].flatMap((r) => [r.left, r.top, r.right, r.bottom])].join(',') };
  }

  private watchLayout = () => {
    this.animationFrame = undefined;
    if (this.disposed || this.suspended || !this.cell) return;
    // ResizeObserver cannot detect a translated ancestor or layout shifts that
    // preserve dimensions. Measure only while inspecting and redraw on change.
    if (this.layout().key !== this.layoutKey) this.refresh();
    if (this.cell && this.animationFrame === undefined) this.animationFrame = requestAnimationFrame(this.watchLayout);
  };

  refresh() {
    if (this.disposed || this.suspended) return;
    const { renderer } = this.viewport.matrix;
    if (renderer.state !== 'ready') { this.leave(); return; }
    // The viewport's DPR notification will rebuild geometry and refresh us.
    if (renderer.view?.dpr !== window.devicePixelRatio) return;
    const cellSize = renderer.view!.scaleX / renderer.view!.dpr;
    this.showMagnifier = magnifierVisible(this.showMagnifier, cellSize);
    const { rect, pane, panels, key } = this.layout();
    this.layoutKey = key;
    if (this.pointer) this.cell = renderer.cellAt(this.pointer.x - rect.left, this.pointer.y - rect.top);
    if (!this.cell) { this.leave(); return; }
    const { row, column } = this.cell;
    const value = renderer.readCell(row, column);
    if (!value) { this.leave(); return; }
    this.select(value.state === 'pending' ? null : this.cell);
    const view = renderer.view!;
    const x = this.pointer?.x ?? rect.left + (column - view.x + 0.5) * view.scaleX / view.dpr;
    const y = this.pointer?.y ?? rect.top + (row - view.y + 0.5) * view.scaleY / view.dpr;
    const bounds = { left: Math.max(0, pane.left), top: Math.max(0, pane.top),
      right: Math.min(window.innerWidth, pane.right), bottom: Math.min(window.innerHeight, pane.bottom) };
    const text = 'value' in value ? formatFloat32(value.value) : 'Unavailable — not received';
    const contentKey = `${row},${column},${text},${Math.min(INSPECTION.width, bounds.right - bounds.left)}`;
    this.readoutKey = contentKey;
    const readoutHeight = this.measuredReadout?.key === contentKey ? this.measuredReadout.height : INSPECTION.readoutHeight;
    if (this.animationFrame === undefined) this.animationFrame = requestAnimationFrame(this.watchLayout);
    let magnifier = this.showMagnifier;
    let position = magnifier && bounds.right - bounds.left >= INSPECTION.width ?
      inspectionPosition(x, y, bounds, rect, panels, cellSize, INSPECTION.height - INSPECTION.readoutHeight + readoutHeight) : null;
    if (!position) {
      magnifier = false;
      position = inspectionPosition(x, y, bounds, rect, panels, cellSize, readoutHeight);
    }
    if (!position) { this.changed(null); return; }
    this.changed({ row, column, magnifier, value: text,
      ...position,
      onReadoutResize: (height) => {
        // A queued observer may belong to a replaced cell/source or old width.
        if (this.disposed || this.readoutKey !== contentKey || this.cell?.row !== row || this.cell.column !== column ||
          !Number.isFinite(height) || height <= 0) return;
        const measured = Math.max(INSPECTION.readoutHeight, Math.ceil(height));
        if (this.measuredReadout?.key === contentKey && this.measuredReadout.height === measured) return;
        this.measuredReadout = { key: contentKey, height: measured };
        if (measured !== readoutHeight) this.refresh();
      },
      draw: (target) => {
        // A queued React render cannot read a disposed/replaced/lost tensor.
        if (this.disposed || renderer.state !== 'ready' || this.cell?.row !== row || this.cell.column !== column) return;
        const context = target.getContext('2d');
        if (context) {
          try { renderer.drawNeighborhood(row, column, context); }
          catch { this.leave(); } // Renderer failure is reported through onStateChange.
        }
      } });
  }

  clear = () => { this.leave(); };
  suspend(active: boolean) { this.suspended = active; if (active) this.clear(); }
  private leave = () => {
    if (this.animationFrame !== undefined) cancelAnimationFrame(this.animationFrame);
    this.animationFrame = undefined;
    this.pointer = null;
    this.cell = null;
    this.select(null);
    this.changed(null);
  };

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    const { canvas, host } = this.viewport.matrix;
    canvas.classList.remove('matrix-inspectable');
    canvas.removeEventListener('pointermove', this.move);
    canvas.removeEventListener('click', this.click);
    canvas.removeEventListener('pointerleave', this.leave);
    host.removeEventListener('focus', this.focus);
    host.removeEventListener('blur', this.leave);
    host.removeEventListener('keydown', this.key);
    window.removeEventListener('scroll', this.windowScroll, true);
    this.leave();
  }
}
