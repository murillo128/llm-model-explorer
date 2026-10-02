import type { Layout, Point, Route } from './graph';
import { endpointKey } from './projection';
import { endpointCorridor } from './routing-clearance';
import { minimumReadableZoom, terminalFootprint, arrowSeparation, preferredRadius, approachLength, departureLength } from './route-metrics';
const haloPadding = 5.5 / minimumReadableZoom / 2;
const near = (a: Point, b: Point) => Math.abs(a.x - b.x) < 0.001 && Math.abs(a.y - b.y) < 0.001;
export type Rectangle = { x: number; y: number; width: number; height: number };
export type DisplaySection = { path: string; samples: Point[] };
export type RouteDisplay = { sections: DisplaySection[]; target: Point; tangent: Point };

/** Remove only redundant vertices; retain reversals and real branch points. */
export function canonicalPoints(points: Point[], junctions: Point[] = []): Point[] {
  const result: Point[] = [];
  for (const p of points) {
    if (result.length && near(result.at(-1)!, p)) continue;
    while (result.length > 1) {
      const a = result.at(-2)!, b = result.at(-1)!;
      if (junctions.some((j) => near(j, b))) break;
      if ((Math.abs(a.x - b.x) < 0.001 && Math.abs(b.x - p.x) < 0.001 && (b.y - a.y) * (p.y - b.y) >= 0) ||
        (Math.abs(a.y - b.y) < 0.001 && Math.abs(b.y - p.y) < 0.001 && (b.x - a.x) * (p.x - b.x) >= 0)) result.pop();
      else break;
    }
    result.push(p);
  }
  return result;
}

/** A repeated-interior proxy can split one straight lane into two sections.
 * Join unambiguous directed continuations before measuring terminal space. */
export function canonicalSections(sections: Point[][], junctions: Point[]): Point[][] {
  const pending = sections.map((s) => canonicalPoints(s, junctions)), result: Point[][] = [];
  while (pending.length) {
    let section = pending.shift()!;
    for (;;) {
      const following = pending.filter((s) => near(section.at(-1)!, s[0]!));
      const preceding = pending.filter((s) => near(s.at(-1)!, section[0]!));
      if (following.length === 1) {
        const next = following[0]!; pending.splice(pending.indexOf(next), 1);
        section = canonicalPoints([...section, ...next], junctions);
      } else if (preceding.length === 1) {
        const previous = preceding[0]!; pending.splice(pending.indexOf(previous), 1);
        section = canonicalPoints([...previous, ...section], junctions);
      } else break;
    }
    result.push(section);
  }
  return result;
}

/** Resolve directed arrival from the actual terminal, independent of section order. */
export function targetTangent(route: Route, target: Point): Point {
  for (const section of route.sections) {
    const points = near(section.at(-1)!, target) ? [...section].reverse() : near(section[0]!, target) ? section : [];
    const previous = points.find((p) => !near(p, target));
    if (previous) {
      const length = Math.hypot(target.x - previous.x, target.y - previous.y);
      return { x: (target.x - previous.x) / length, y: (target.y - previous.y) / length };
    }
  }
  throw new Error('Connection has no directed target terminal.');
}

export function arrowTransform(target: Point, tangent: Point, zoom: number) {
  const scale = Math.max(minimumReadableZoom, zoom);
  const offset = terminalFootprint + arrowSeparation / scale;
  return `translate(${target.x - tangent.x * offset} ${target.y - tangent.y * offset}) rotate(${Math.atan2(tangent.y, tangent.x) * 180 / Math.PI}) scale(${1 / scale})`;
}

/** The quadratic lies in its control triangle. Its padded bounding rectangle is
 * a conservative painted-envelope test, including the maximum visible halo.
 * Samples index this same curve once, never during pointer movement. */
export function roundedSection(input: Point[], junctions: Point[], obstacles: Rectangle[], source?: Point, target?: Point): DisplaySection {
  const points = canonicalPoints(input, junctions);
  if (!points.length) return { path: '', samples: [] };
  let path = `M${points[0]!.x},${points[0]!.y}`;
  const samples = [points[0]!];
  const line = (p: Point) => { path += ` L${p.x},${p.y}`; samples.push(p); };
  for (let i = 1; i < points.length - 1; i++) {
    const a = points[i - 1]!, b = points[i]!, c = points[i + 1]!;
    const before = Math.hypot(b.x - a.x, b.y - a.y), after = Math.hypot(c.x - b.x, c.y - b.y);
    const u = { x: (a.x - b.x) / before, y: (a.y - b.y) / before }, v = { x: (c.x - b.x) / after, y: (c.y - b.y) / after };
    let radius = Math.min(preferredRadius, before / 2, after / 2);
    if (Math.abs(u.x * v.x + u.y * v.y) > 0.001 || junctions.some((j) => near(j, b))) radius = 0;
    // A branch can lie inside an adjacent segment without being a vertex of
    // this section. Never consume that branch point with a fillet either.
    for (const j of junctions) for (const [direction, length] of [[u, before], [v, after]] as const) {
      const dx = j.x - b.x, dy = j.y - b.y, distance = dx * direction.x + dy * direction.y;
      if (Math.abs(dx * direction.y - dy * direction.x) < 0.001 && distance >= 0 && distance <= length)
        radius = Math.min(radius, distance);
    }
    for (const [length, distance] of [[before, source && near(a, source) ? departureLength : target && near(a, target) ? approachLength : 0],
      [after, source && near(c, source) ? departureLength : target && near(c, target) ? approachLength : 0]] as const) {
      radius = Math.max(0, Math.min(radius, length - distance));
    }
    const ends = (r: number) => [{ x: b.x + u.x * r, y: b.y + u.y * r }, { x: b.x + v.x * r, y: b.y + v.y * r }] as const;
    while (radius >= 0.5) {
      const [start, end] = ends(radius);
      const left = Math.min(start.x, b.x, end.x) - haloPadding, right = Math.max(start.x, b.x, end.x) + haloPadding;
      const top = Math.min(start.y, b.y, end.y) - haloPadding, bottom = Math.max(start.y, b.y, end.y) + haloPadding;
      if (!obstacles.some((r) => left < r.x + r.width && right > r.x && top < r.y + r.height && bottom > r.y)) break;
      radius /= 2;
    }
    if (radius < 0.5) { line(b); continue; }
    const [start, end] = ends(radius);
    line(start); path += ` Q${b.x},${b.y} ${end.x},${end.y}`;
    for (let step = 1; step <= 8; step++) {
      const t = step / 8, s = 1 - t;
      samples.push({ x: s * s * start.x + 2 * s * t * b.x + t * t * end.x,
        y: s * s * start.y + 2 * s * t * b.y + t * t * end.y });
    }
  }
  if (points.length > 1) line(points.at(-1)!);
  return { path, samples };
}

export function routeDisplays(layout: Layout): Map<string, RouteDisplay> {
  const ports = new Map(layout.ports.map((p) => [endpointKey({ node_id: p.nodeId, port_id: p.portId }), p]));
  const nodes = new Map(layout.projection.nodes.map((n) => [n.id, n]));
  const obstacles: Rectangle[] = layout.ports.map((p) => ({ x: p.label.x - p.label.clearance, y: p.label.y - p.label.clearance,
    width: p.label.width + p.label.clearance * 2, height: p.label.height + p.label.clearance * 2 }));
  for (const box of layout.boxes) {
    const node = nodes.get(box.id)!;
    obstacles.push({ x: box.absoluteX, y: box.absoluteY, width: box.width, height: node.expanded ? box.headerHeight ?? 64 : box.height });
  }
  for (const edge of layout.projection.edges) for (const [endpoint, source] of [[edge.source, true], [edge.target, false]] as const)
    obstacles.push(endpointCorridor(ports.get(endpointKey(endpoint))!, source));
  obstacles.push(...layout.routes.flatMap((r) => r.labels ?? []));
  // Reuse a bounded spatial lookup across routes; exhaustive diagrams should
  // not compare every bend against every label in the entire model.
  const buckets = new Map<number, Rectangle[]>(), bucketWidth = 256;
  for (const obstacle of obstacles) for (let i = Math.floor(obstacle.x / bucketWidth); i <= Math.floor((obstacle.x + obstacle.width) / bucketWidth); i++) {
    const bucket = buckets.get(i) ?? []; bucket.push(obstacle); buckets.set(i, bucket);
  }
  const junctions = layout.routes.flatMap((r) => r.junctions);
  const edges = new Map(layout.projection.edges.map((e) => [e.id, e]));
  return new Map(layout.routes.map((route) => {
    const edge = edges.get(route.id)!, from = ports.get(endpointKey(edge.source))!, to = ports.get(endpointKey(edge.target))!;
    const source = { x: from.absoluteX, y: from.absoluteY }, target = { x: to.absoluteX, y: to.absoluteY };
    const candidates = new Set<Rectangle>();
    for (const section of route.sections) for (const p of section) {
      for (let i = Math.floor((p.x - preferredRadius - haloPadding) / bucketWidth); i <= Math.floor((p.x + preferredRadius + haloPadding) / bucketWidth); i++)
        for (const obstacle of buckets.get(i) ?? []) candidates.add(obstacle);
    }
    return [route.id, { target, tangent: targetTangent(route, target),
      sections: route.sections.map((s) => roundedSection(s, junctions, [...candidates], source, target)) }];
  }));
}
