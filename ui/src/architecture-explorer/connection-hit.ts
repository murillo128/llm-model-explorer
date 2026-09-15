import type { Point, Route } from './graph';
import type { ProjectedEdge } from './projection';
import { endpointKey } from './projection';

type Segment = { start: Point; end: Point };
type RoutedConnection = { id: string; segments: Segment[] };
const epsilon = 1e-6;

function nearest(point: Point, { start, end }: Segment) {
  const dx = end.x - start.x, dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  const t = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared));
  const at = { x: start.x + t * dx, y: start.y + t * dy };
  return { at, distance: (point.x - at.x) ** 2 + (point.y - at.y) ** 2 };
}

/** A shared segment has positive length; a crossing or touching endpoint alone
 * does not join branches, even when they originate at the same source. */
function sharesAt(owner: Segment, candidate: Segment, point: Point): boolean {
  const dx = owner.end.x - owner.start.x, dy = owner.end.y - owner.start.y;
  const length = Math.hypot(dx, dy), ux = dx / length, uy = dy / length;
  const along = (p: Point) => (p.x - owner.start.x) * ux + (p.y - owner.start.y) * uy;
  const across = (p: Point) => (p.x - owner.start.x) * uy - (p.y - owner.start.y) * ux;
  if (Math.abs(across(candidate.start)) > epsilon || Math.abs(across(candidate.end)) > epsilon) return false;
  const a = along(candidate.start), b = along(candidate.end);
  const from = Math.max(0, Math.min(a, b)), to = Math.min(length, Math.max(a, b));
  const hit = along(point);
  return to - from > epsilon && hit >= from - epsilon && hit <= to + epsilon;
}

/** Index once per generated layout, then examine only the hit connection's
 * source-port fan-out. Presentation source ports preserve exact source identity,
 * including aliases across collapsed groups; labels/shapes never join signals.
 * Snap to the owning line inside its generous CSS hit corridor before resolving
 * collinear overlap, so nearby parallel lines and crossings stay independent. */
export function connectionHitResolver(edges: ProjectedEdge[], routes: Route[]) {
  const routed = new Map(routes.map((route) => [route.id, { id: route.id,
    segments: route.sections.flatMap((points) => points.slice(1).flatMap((end, i) => {
      const start = points[i]!;
      return start.x === end.x && start.y === end.y ? [] : [{ start, end }];
    })),
  }]));
  const fanout = new Map<string, RoutedConnection[]>();
  const sources = new Map<string, string>();
  for (const edge of edges) {
    const key = endpointKey(edge.source), route = routed.get(edge.id);
    sources.set(edge.id, key);
    if (route) {
      const group = fanout.get(key) ?? [];
      group.push(route); fanout.set(key, group);
    }
  }
  return (edgeId: string, point: Point): string[] => {
    const owner = routed.get(edgeId), group = fanout.get(sources.get(edgeId) ?? '');
    if (!owner?.segments.length || !group || group.length === 1) return [edgeId];
    const hits = owner.segments.map((segment) => ({ segment, ...nearest(point, segment) }));
    const distance = Math.min(...hits.map((hit) => hit.distance));
    const closest = hits.filter((hit) => hit.distance <= distance + epsilon);
    return group.filter((candidate) => candidate.id === edgeId || closest.some((hit) =>
      candidate.segments.some((segment) => sharesAt(hit.segment, segment, hit.at)))).map((candidate) => candidate.id);
  };
}
