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

  constructor(private readonly viewport: MatrixViewport, private readonly axis: 'rows' | 'columns' = 'rows') {
    const profile = axis === 'rows' ? viewport.rows : viewport.columns;
    this.surfaces = [viewport.matrix.renderer, ...(profile ? [profile] : [])]
      .map(renderer => ({ renderer, guides: [] as SVGSVGElement[] }));
  }

  setRows(rows: readonly number[] = []) {
    this.intervals = selectedRowIntervals(rows, this.viewport.matrix.renderer.geometry[this.axis]);
    this.refresh();
  }

  refresh() {
    for (const { renderer, guides } of this.surfaces) {
      const view = renderer.view;
      let used = 0;
      if (view && renderer.state === 'ready' && this.viewport.matrix.renderer.state === 'ready') {
        for (const [from, to] of this.intervals) {
          const rows = this.axis === 'rows';
          const start = rasterEdge(from, rows ? view.y : view.x, rows ? view.scaleY : view.scaleX);
          const end = rasterEdge(to, rows ? view.y : view.x, rows ? view.scaleY : view.scaleX);
          const extent = rows ? view.height : view.width;
          const first = Math.max(0, start), last = Math.min(extent, end);
          if (last <= first || (rows ? view.width : view.height) <= 0) continue;
          let guide = guides[used++];
          if (!guide) {
            guide = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            guide.setAttribute('class', `matrix-selected-${this.axis}`);
            guide.setAttribute('aria-hidden', 'true');
            Object.assign(guide.style, { position: 'absolute', pointerEvents: 'none', overflow: 'hidden', display: 'block' });
            const outline = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            outline.setAttribute('fill', 'none');
            outline.setAttribute('stroke', 'rgba(245, 154, 56, 0.7)');
            outline.setAttribute('stroke-width', '1');
            outline.setAttribute('shape-rendering', 'crispEdges');
            guide.append(outline);
            guides.push(guide);
            renderer.canvas.parentElement!.append(guide);
          }
          guide.dataset[this.axis] = `${from}:${to}`;
          const width = rows ? view.width : last - first, height = rows ? last - first : view.height;
          // SVG units are device pixels. Draw inside the clipped raster box:
          // CSS borders impose a minimum box size and round fractional widths,
          // so they cannot represent singleton rows at native scale / high DPR.
          guide.setAttribute('viewBox', `0 0 ${width} ${height}`);
          guide.firstElementChild!.setAttribute('d', (rows ? [
            `M0.5 0V${height} M${width - 0.5} 0V${height}`,
            ...(start >= 0 ? [`M0 0.5H${width}`] : []),
            ...(end <= extent ? [`M0 ${height - 0.5}H${width}`] : []),
          ] : [
            `M0 0.5H${width} M0 ${height - 0.5}H${width}`,
            ...(start >= 0 ? [`M0.5 0V${height}`] : []),
            ...(end <= extent ? [`M${width - 0.5} 0V${height}`] : []),
          ]).join(' '));
          Object.assign(guide.style, {
            left: `${parseFloat(renderer.canvas.style.left || '0') + (rows ? 0 : first / view.dpr)}px`,
            top: `${parseFloat(renderer.canvas.style.top || '0') + (rows ? first / view.dpr : 0)}px`,
            width: `${width / view.dpr}px`, height: `${height / view.dpr}px`,
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
