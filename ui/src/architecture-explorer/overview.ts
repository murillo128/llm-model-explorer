import type { Graph, Layout } from './graph';
import { interfaceIndex } from './interfaces';

// 13px component labels retain at least 10.4 CSS pixels on a fresh view.
export const minimumOverviewScale = 0.8;
export const overviewInset = 16;
export const maximumOverviewScale = 1;

/** The visible outer boundary, after passive declarations become ports. */
export function overviewExpansion(graph: Graph): string[] {
  const outer = interfaceIndex(graph).outer;
  return outer.kind === 'source' ? [outer.id] : [];
}

export function visibleBounds(layout: Layout) {
  const points = layout.boxes.flatMap((b) => [{ x: b.absoluteX, y: b.absoluteY },
    { x: b.absoluteX + b.width, y: b.absoluteY + b.height }]);
  for (const route of layout.routes) {
    points.push(...route.sections.flat(), ...route.junctions);
    for (const label of route.labels ?? []) points.push(label, { x: label.x + label.width, y: label.y + label.height });
  }
  const x = Math.min(...points.map((p) => p.x)), y = Math.min(...points.map((p) => p.y));
  return points.length ? { x, y, width: Math.max(...points.map((p) => p.x)) - x, height: Math.max(...points.map((p) => p.y)) - y }
    : { x: 0, y: 0, width: 0, height: 0 };
}
type Bounds = ReturnType<typeof visibleBounds>;
export function overviewScale(bounds: Bounds, width: number, height: number) {
  if (width <= 0 || height <= 0) return undefined;
  return Math.max(0, Math.min(maximumOverviewScale,
    (width - 2 * overviewInset) / Math.max(1, bounds.width), (height - 2 * overviewInset) / Math.max(1, bounds.height)));
}
export function initialViewport(bounds: Bounds, width: number, height: number) {
  const zoom = Math.max(minimumOverviewScale, overviewScale(bounds, width, height) ?? minimumOverviewScale);
  return { x: Math.max(overviewInset, (width - bounds.width * zoom) / 2) - bounds.x * zoom,
    y: overviewInset - bounds.y * zoom, zoom };
}
