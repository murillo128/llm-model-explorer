import { hitTest, rasterEdge } from './geometry';
import type { TensorGeometry, ViewGeometry } from './geometry';

/** Half-open, integer logical bounds. An omitted axis retains its current center. */
export interface ZoomBounds {
  readonly columns?: readonly [number, number];
  readonly rows?: readonly [number, number];
}

export function selectionCamera(bounds: ZoomBounds, view: ViewGeometry, tensor: TensorGeometry,
  width: number, height: number) {
  if ((!bounds.columns && !bounds.rows) || width <= 0 || height <= 0) return null;
  for (const [range, limit] of [[bounds.columns, tensor.columns], [bounds.rows, tensor.rows]] as const) {
    if (range && (!range.every(Number.isSafeInteger) || range[0] < 0 || range[1] > limit || range[1] <= range[0])) return null;
  }
  const scale = Math.max(1, Math.min(bounds.columns ? width / (bounds.columns[1] - bounds.columns[0]) : Infinity,
    bounds.rows ? height / (bounds.rows[1] - bounds.rows[0]) : Infinity));
  const centerX = bounds.columns ? (bounds.columns[0] + bounds.columns[1]) / 2 : view.x + view.width / view.scaleX / 2;
  const centerY = bounds.rows ? (bounds.rows[0] + bounds.rows[1]) / 2 : view.y + view.height / view.scaleY / 2;
  return { scale,
    x: Math.max(0, Math.min(Math.max(0, tensor.columns - width / scale), centerX - width / scale / 2)),
    y: Math.max(0, Math.min(Math.max(0, tensor.rows - height / scale), centerY - height / scale / 2)) };
}

/** Snap to the nearer rasterized logical boundary, sharing the renderer's edge math. */
export function selectionBoundary(view: ViewGeometry, css: number, axis: 'columns' | 'rows', limit: number) {
  const horizontal = axis === 'columns';
  const extent = horizontal ? view.width : view.height;
  const pixel = Math.max(0, Math.min(extent - 1, Math.floor(css * view.dpr)));
  const cell = hitTest(view, horizontal ? pixel / view.dpr : 0, horizontal ? 0 : pixel / view.dpr);
  if (!cell) return 0;
  const index = horizontal ? cell.column : cell.row;
  const origin = horizontal ? view.x : view.y, scale = horizontal ? view.scaleX : view.scaleY;
  const low = rasterEdge(index, origin, scale), high = rasterEdge(index + 1, origin, scale);
  return Math.max(0, Math.min(limit, css * view.dpr < (low + high) / 2 ? index : index + 1));
}
