import type { Box, Point, PortPosition, Route } from './graph';
import { endpointKey, type ProjectedEdge, type Projection } from './projection';
import { approachLength, departureLength, arrowWidth, arrowLength, arrowSeparation, terminalFootprint, minimumReadableZoom } from './route-metrics';

const margin = 8;
const laneHalfWidth = 2;
const epsilon = 0.001;
type Rectangle = { x: number; y: number; width: number; height: number };

function decoration(port: PortPosition): Rectangle {
  const width = (arrowLength + arrowSeparation) / minimumReadableZoom, height = arrowWidth / minimumReadableZoom;
  return { x: port.absoluteX - terminalFootprint - width, y: port.absoluteY - height / 2, width, height };
}
function overlaps(a: Rectangle, b: Rectangle) {
  return a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
}

function crosses(a: Point, b: Point, rect: Rectangle) {
  const left = rect.x + epsilon, right = rect.x + rect.width - epsilon;
  const top = rect.y + epsilon, bottom = rect.y + rect.height - epsilon;
  if (Math.abs(a.y - b.y) < epsilon) return a.y > top && a.y < bottom &&
    Math.max(a.x, b.x) > left && Math.min(a.x, b.x) < right;
  if (Math.abs(a.x - b.x) < epsilon) return a.x > left && a.x < right &&
    Math.max(a.y, b.y) > top && Math.min(a.y, b.y) < bottom;
  return true;
}

function segments(route: Route) {
  return route.sections.flatMap((section) => section.slice(1).map((point, index) => [section[index]!, point] as const));
}

export function endpointCorridor(port: PortPosition, source: boolean): Rectangle {
  const x = port.absoluteX;
  const label = port.label;
  // The text lives inside an ordinary card and above an expanded boundary's
  // own cable. Extend the lane past the text only on the side being routed.
  const end = source ? Math.max(x + departureLength, label.x > x ? label.x + label.width + margin : x + departureLength) :
    Math.min(x - approachLength, label.x + label.width < x ? label.x - margin : x - approachLength);
  return { x: Math.min(x, end), y: port.absoluteY - laneHalfWidth,
    width: Math.abs(end - x), height: laneHalfWidth * 2 };
}

function ownTerminal(route: Route, port: PortPosition, source: boolean, corridor: Rectangle) {
  const section = route.sections.find((points) => {
    const point = source ? points[0] : points.at(-1);
    return point && Math.abs(point.x - port.absoluteX) < epsilon && Math.abs(point.y - port.absoluteY) < epsilon;
  });
  if (!section || section.length < 2) return false;
  const next = source ? section[1]! : section.at(-2)!;
  return Math.abs(next.y - port.absoluteY) < epsilon &&
    (source ? next.x >= corridor.x + corridor.width - epsilon : next.x <= corridor.x + epsilon);
}

/** Exact displayed rectangles are checked after ELK has routed. Layout failures
 * stay recoverable instead of publishing a geometry that covers port names. */
export function routeClearanceFailures(projection: Projection, ports: PortPosition[], routes: Route[]): string[] {
  const positions = new Map(ports.map((port) => [endpointKey({ node_id: port.nodeId, port_id: port.portId }), port]));
  const edges = new Map(projection.edges.map((edge) => [edge.id, edge]));
  const corridors = new Map<string, { key: string; source: boolean; rect: Rectangle }>();
  for (const edge of projection.edges) for (const [endpoint, source] of [[edge.source, true], [edge.target, false]] as const) {
    const key = endpointKey(endpoint), port = positions.get(key);
    if (port) corridors.set(`${key}:${source}`, { key, source, rect: endpointCorridor(port, source) });
  }
  type Obstacle = { rect: Rectangle; label: PortPosition } | { rect: Rectangle; corridor: { key: string; source: boolean } };
  const bucketWidth = 256;
  const buckets = new Map<number, Obstacle[]>();
  const add = (obstacle: Obstacle) => {
    const first = Math.floor(obstacle.rect.x / bucketWidth);
    const last = Math.floor((obstacle.rect.x + obstacle.rect.width) / bucketWidth);
    for (let bucket = first; bucket <= last; bucket++) {
      const entries = buckets.get(bucket) ?? [];
      entries.push(obstacle); buckets.set(bucket, entries);
    }
  };
  for (const port of ports) {
    const label = port.label, padding = label.clearance;
    add({ label: port, rect: { x: label.x - padding, y: label.y - padding,
      width: label.width + padding * 2, height: label.height + padding * 2 } });
  }
  for (const corridor of corridors.values()) add({ rect: corridor.rect,
    corridor: { key: corridor.key, source: corridor.source } });
  const failures: string[] = [];
  for (const route of routes) {
    const edge: ProjectedEdge | undefined = edges.get(route.id);
    if (!edge) { failures.push(`${route.id}: unknown connection`); continue; }
    const lines = segments(route);
    const points = route.sections.flat();
    const target = positions.get(endpointKey(edge.target)), glyph = target && decoration(target);
    // The broad phase must include paint above a completely straight spine.
    // Otherwise a raised label can clear the line and still mask the triangle.
    const bounds = { left: Math.min(...points.map((point) => point.x), glyph?.x ?? Infinity),
      right: Math.max(...points.map((point) => point.x), glyph ? glyph.x + glyph.width : -Infinity),
      top: Math.min(...points.map((point) => point.y), glyph?.y ?? Infinity),
      bottom: Math.max(...points.map((point) => point.y), glyph ? glyph.y + glyph.height : -Infinity) };
    for (const [endpoint, source] of [[edge.source, true], [edge.target, false]] as const) {
      const port = positions.get(endpointKey(endpoint));
      if (!port || !ownTerminal(route, port, source, endpointCorridor(port, source))) {
        failures.push(`${route.id}: terminal turns before ${source ? 'departure' : 'approach'} clearance`);
      }
    }
    const candidates = new Set<Obstacle>();
    for (let bucket = Math.floor(bounds.left / bucketWidth); bucket <= Math.floor(bounds.right / bucketWidth); bucket++) {
      for (const obstacle of buckets.get(bucket) ?? []) candidates.add(obstacle);
    }
    for (const obstacle of candidates) {
      const rect = obstacle.rect;
      if (rect.x > bounds.right || rect.x + rect.width < bounds.left || rect.y > bounds.bottom || rect.y + rect.height < bounds.top) continue;
      if ('label' in obstacle) {
        if (lines.some(([a, b]) => crosses(a, b, rect))) failures.push(`${route.id}: crosses ${obstacle.label.nodeId}.${obstacle.label.portId} label`);
        if (glyph) {
          // Conservative maximum glyph envelope; the semantic line still meets
          // the port center. The triangle never extends above/below this band.
          if (overlaps(glyph, rect))
            failures.push(`${route.id}: destination decoration crosses ${obstacle.label.nodeId}.${obstacle.label.portId} label`);
        }
        continue;
      }
      const corridor = obstacle.corridor;
      const own = corridor.source ? endpointKey(edge.source) === corridor.key : endpointKey(edge.target) === corridor.key;
      if (own) {
        const port = positions.get(corridor.key)!;
        const anchored = (point: Point) => Math.abs(point.x - port.absoluteX) < epsilon && Math.abs(point.y - port.absoluteY) < epsilon;
        if (lines.some(([a, b]) => crosses(a, b, rect) && !(corridor.source ? anchored(a) : anchored(b))))
          failures.push(`${route.id}: returns through its terminal corridor`);
        if (route.junctions.some((point) => point.x > rect.x + epsilon &&
          point.x < rect.x + rect.width - epsilon &&
          point.y > rect.y + epsilon && point.y < rect.y + rect.height - epsilon))
          failures.push(`${route.id}: branches beside a terminal`);
      } else if (lines.some(([a, b]) => crosses(a, b, rect))) {
        failures.push(`${route.id}: crosses ${corridor.key} corridor`);
      }
    }
  }
  return failures;
}

export function assertProtectedRoutes(projection: Projection, ports: PortPosition[], routes: Route[], boxes: Box[] = []) {
  const failures = routeClearanceFailures(projection, ports, routes);
  const positions = new Map(ports.map((p) => [endpointKey({ node_id: p.nodeId, port_id: p.portId }), p]));
  const expanded = new Set(projection.nodes.filter((n) => n.expanded).map((n) => n.id));
  const obstacles = [...boxes.map((b) => ({ x: b.absoluteX, y: b.absoluteY, width: b.width, height: expanded.has(b.id) ? b.headerHeight ?? 64 : b.height })),
    ...routes.flatMap((r) => r.labels ?? [])];
  for (const edge of projection.edges) {
    const target = positions.get(endpointKey(edge.target));
    if (target && obstacles.some((rect) => overlaps(decoration(target), rect))) failures.push(`${edge.id}: destination decoration crosses a card/header or dimension label`);
  }
  if (failures.length) throw new Error(`Port-label routing clearance failed (${failures.slice(0, 3).join('; ')}). Collapse groups and retry.`);
}
