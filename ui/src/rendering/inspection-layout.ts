/** CSS-pixel UI tokens; camera scale is converted from device pixels once. */
export const INSPECTION = { hideCellSize: 10, showCellSize: 8, width: 170, height: 212, readoutHeight: 42, gap: 12 } as const;

export function magnifierVisible(visible: boolean, cellSize: number) {
  return visible ? cellSize < INSPECTION.hideCellSize : cellSize < INSPECTION.showCellSize;
}

export interface InspectionRect { left: number; top: number; right: number; bottom: number }
const overlap = (a: InspectionRect, b: InspectionRect) =>
  Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) *
  Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));

/** Panels are hard exclusions. Prefer avoiding the neighborhood, then the matrix,
 * then proximity. Boundary candidates also find free space beyond short matrices.
 */
export function inspectionPosition(x: number, y: number, bounds: InspectionRect,
  matrix: InspectionRect, panels: readonly InspectionRect[], cellSize: number, height = INSPECTION.height as number) {
  if (height === INSPECTION.height && bounds.right - bounds.left < INSPECTION.width) return null;
  const width = Math.min(INSPECTION.width, bounds.right - bounds.left);
  if (width <= 0 || height > bounds.bottom - bounds.top) return null;
  const radius = 4.5 * cellSize, gap = radius + INSPECTION.gap;
  const xs = [x + gap, x - gap - width, bounds.left, bounds.right - width];
  const ys = [y + gap, y - gap - height, bounds.top, bounds.bottom - height];
  for (const rect of [matrix, ...panels]) {
    xs.push(rect.right + INSPECTION.gap, rect.left - INSPECTION.gap - width);
    ys.push(rect.bottom + INSPECTION.gap, rect.top - INSPECTION.gap - height);
  }
  const neighborhood = { left: x - radius, right: x + radius, top: y - radius, bottom: y + radius };
  const candidates = xs.flatMap((x) => ys.map((y) => {
    const left = Math.max(bounds.left, Math.min(bounds.right - width, x));
    const top = Math.max(bounds.top, Math.min(bounds.bottom - height, y));
    return { left, top, right: left + width, bottom: top + height, width, height };
  })).filter((rect) => panels.every((panel) => overlap(rect, panel) === 0));
  const scores = (rect: InspectionRect) => [overlap(rect, neighborhood), overlap(rect, matrix),
    Math.hypot((rect.left + rect.right) / 2 - x, (rect.top + rect.bottom) / 2 - y)];
  candidates.sort((a, b) => {
    const sa = scores(a), sb = scores(b);
    return sa[0]! - sb[0]! || sa[1]! - sb[1]! || sa[2]! - sb[2]!;
  });
  return candidates[0] ?? null;
}
