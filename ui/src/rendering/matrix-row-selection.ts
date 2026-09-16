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
      .map(renderer => ({ renderer, guides: [] as SVGSVGElement[] }));
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
            guide = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            guide.setAttribute('class', 'matrix-selected-rows');
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
          guide.dataset.rows = `${from}:${to}`;
          const width = view.width, height = bottom - top;
          // SVG units are device pixels. Draw inside the clipped raster box:
          // CSS borders impose a minimum box size and round fractional widths,
          // so they cannot represent singleton rows at native scale / high DPR.
          guide.setAttribute('viewBox', `0 0 ${width} ${height}`);
          guide.firstElementChild!.setAttribute('d', [
            `M0.5 0V${height} M${width - 0.5} 0V${height}`,
            ...(start >= 0 ? [`M0 0.5H${width}`] : []),
            ...(end <= view.height ? [`M0 ${height - 0.5}H${width}`] : []),
          ].join(' '));
          Object.assign(guide.style, {
            left: `${parseFloat(renderer.canvas.style.left || '0')}px`,
            top: `${parseFloat(renderer.canvas.style.top || '0') + top / view.dpr}px`,
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
