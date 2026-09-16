import { rasterEdge } from './geometry';
import type { MatrixViewport } from './matrix-viewport';

/** Half-open native intervals; adjacent positions coalesce, gaps stay gaps. */
export function selectedRowIntervals(rows: readonly number[], count: number): [number, number][] {
  const positions = [...new Set(rows.filter(row => Number.isSafeInteger(row) && row >= 0 && row < count))].sort((a, b) => a - b);
  const intervals: [number, number][] = [];
  for (const row of positions) {
    const previous = intervals.at(-1);
    if (previous?.[1] === row) previous[1] = row + 1;
    else intervals.push([row, row + 1]);
  }
  return intervals;
}

/** Persistent display-only guides. No renderer allocation, draw, or camera calls. */
export class MatrixRowSelection {
  private intervals: [number, number][] = [];
  private readonly surfaces;

  constructor(private readonly viewport: MatrixViewport) {
    this.surfaces = [viewport.matrix.renderer, ...(viewport.rows ? [viewport.rows] : [])]
      .map(renderer => ({ renderer, guides: [] as HTMLDivElement[] }));
  }

  setRows(rows: readonly number[] = []) {
    this.intervals = selectedRowIntervals(rows, this.viewport.matrix.renderer.geometry.rows);
    this.refresh();
  }

  refresh() {
    for (const { renderer, guides } of this.surfaces) {
      const view = renderer.view;
      let used = 0;
      if (view && renderer.state === 'ready' && this.viewport.matrix.renderer.state === 'ready') {
        for (const [from, to] of this.intervals) {
          const start = rasterEdge(from, view.y, view.scaleY), end = rasterEdge(to, view.y, view.scaleY);
          const top = Math.max(0, start), bottom = Math.min(view.height, end);
          if (bottom <= top || view.width <= 0) continue;
          let guide = guides[used++];
          if (!guide) {
            guide = document.createElement('div');
            guide.className = 'matrix-selected-rows';
            guide.setAttribute('aria-hidden', 'true');
            Object.assign(guide.style, { position: 'absolute', pointerEvents: 'none', boxSizing: 'border-box' });
            guides.push(guide);
            renderer.canvas.parentElement!.append(guide);
          }
          guide.dataset.rows = `${from}:${to}`;
          Object.assign(guide.style, {
            left: `${parseFloat(renderer.canvas.style.left || '0')}px`,
            top: `${parseFloat(renderer.canvas.style.top || '0') + top / view.dpr}px`,
            width: `${view.width / view.dpr}px`, height: `${(bottom - top) / view.dpr}px`,
            border: `${1 / view.dpr}px solid rgba(245, 154, 56, 0.7)`,
            borderTopWidth: start < 0 ? '0px' : `${1 / view.dpr}px`,
            borderBottomWidth: end > view.height ? '0px' : `${1 / view.dpr}px`,
          });
        }
      }
      for (const guide of guides.splice(used)) guide.remove();
    }
  }

  dispose() {
    this.intervals = [];
    for (const { guides } of this.surfaces) for (const guide of guides.splice(0)) guide.remove();
  }
}
