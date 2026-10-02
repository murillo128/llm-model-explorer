import type { Box, Point, PortPosition, Route } from './graph';
import { endpointKey, type Projection } from './projection';
import { canonicalSections } from './route-display';
import { approachLength, departureLength } from './route-metrics';
import { endpointCorridor, routeClearanceFailures } from './routing-clearance';

const epsilon = 0.001;
const laneGap = 16;

export class BoundaryRowSpace extends Error {
  constructor(readonly additionalHeights: ReadonlyMap<string, number>) {
    super('Boundary rows need more container space. Collapse groups and retry.');
  }
}

/** ELK chooses the order; regularize boundary rows and their terminal segments
 * while retaining interior route topology and exact source/projection records. */
export function regularizeBoundaryRows(projection: Projection, boxes: Box[], ports: PortPosition[], routes: Route[],
  metrics: ReadonlyMap<string, { headerHeight: number; portGap: number; minimumPortGap: number }>) {
  const positions = new Map(ports.map((port) => [endpointKey({ node_id: port.nodeId, port_id: port.portId }), port]));
  const nodes = new Map(projection.nodes.map((node) => [node.id, node]));
  const inside = (id: string, ancestor: string) => {
    for (let node = nodes.get(id); node?.parentId; node = nodes.get(node.parentId)) if (node.parentId === ancestor) return true;
    return false;
  };
  const original = new Map(ports.map((port) => [port, port.absoluteY]));
  const additionalHeights = new Map<string, number>();
  const byBox = new Map(boxes.map((box) => [box.id, box]));
  let widthGrowth = 0;
  // Inserting space in a gutter is a monotone translation: horizontal segments
  // stretch, vertical segments and ELK's interior topology remain unchanged.
  const growGutter = (at: number, amount: number, outsideLeft: boolean) => {
    const shift = (x: number) => outsideLeft ? x <= at ? x - amount : x : x >= at ? x + amount : x;
    for (const box of boxes) {
      const right = shift(box.absoluteX + box.width);
      box.absoluteX = shift(box.absoluteX); box.width = right - box.absoluteX;
    }
    for (const box of boxes) box.x = box.absoluteX - (box.parentId ? byBox.get(box.parentId)!.absoluteX : 0);
    for (const port of ports) {
      const x = shift(port.absoluteX), delta = x - port.absoluteX;
      port.absoluteX = x; port.x = x - byBox.get(port.nodeId)!.absoluteX; port.label.x += delta;
    }
    const points = new Set(routes.flatMap((route) => [...route.sections.flat(), ...route.junctions]));
    for (const point of points) point.x = shift(point.x);
    for (const route of routes) for (const label of route.labels ?? []) label.x = shift(label.x);
    widthGrowth += amount;
  };
  // Children first: forwarding through two expanded boundaries shares the
  // already normalized interior rows, never a declaration-order permutation.
  for (const node of [...projection.nodes].reverse()) {
    if (!node.expanded) continue;
    const metric = metrics.get(node.id)!;
    for (const side of ['left', 'right'] as const) {
      const ordered = ports.filter((port) => port.nodeId === node.id && port.side === side).sort((a, b) => a.y - b.y);
      if (!ordered.length) continue;
      const rowGaps = ordered.map((port, index) => {
        const previous = ordered[index - 1];
        return previous ? Math.max(metric.minimumPortGap, previous.label.y - previous.absoluteY + previous.label.height + previous.label.clearance -
          (port.label.y - port.absoluteY) + port.label.clearance) : 0;
      });
      const peers = ordered.map((port) => {
        const key = endpointKey({ node_id: port.nodeId, port_id: port.portId });
        const edges = projection.edges.filter((edge) => endpointKey(side === 'left' ? edge.source : edge.target) === key &&
          inside((side === 'left' ? edge.target : edge.source).node_id, node.id));
        return edges.length === 1 ? positions.get(endpointKey(side === 'left' ? edges[0]!.target : edges[0]!.source)) : undefined;
      });
      const box = byBox.get(node.id)!;
      // Prefer the longest ordered 1:1 runs, then singleton wishes. Reserve
      // minimum space for intervening unrelated/unused rows before accepting an
      // anchor. A conflicting wish stays routed; it cannot displace a clear run.
      const desired = ordered.map((port) => port.absoluteY);
      let offset = 0;
      const offsets = rowGaps.map((gap) => (offset += gap));
      const runs: { start: number; end: number }[] = [];
      for (let i = 0; i < peers.length; i++) {
        const first = peers[i];
        if (!first) continue;
        let end = i;
        while (end + 1 < peers.length && peers[end + 1]?.nodeId === first.nodeId && peers[end + 1]?.side === first.side &&
          peers[end + 1]!.absoluteY - peers[end]!.absoluteY >= rowGaps[end + 1]! - epsilon) end++;
        runs.push({ start: i, end }); i = end;
      }
      const anchors = new Map<number, number>();
      const headerMinimum = Math.max(...ordered.map((port, i) => box.absoluteY + metric.headerHeight -
        (port.label.y - port.absoluteY) + port.label.clearance - offsets[i]!));
      for (const run of runs.sort((a, b) => (b.end - b.start) - (a.end - a.start) || a.start - b.start)) {
        const first = peers[run.start]!.absoluteY, last = peers[run.end]!.absoluteY;
        const fits = first >= headerMinimum + offsets[run.start]! - epsilon && [...anchors].every(([i, y]) =>
          i < run.start ? first - y >= offsets[run.start]! - offsets[i]! - epsilon :
            i > run.end ? y - last >= offsets[i]! - offsets[run.end]! - epsilon : true);
        if (fits) for (let i = run.start; i <= run.end; i++) anchors.set(i, peers[i]!.absoluteY);
      }
      let ceiling = Infinity;
      for (let i = desired.length - 1; i >= 0; i--) {
        if (anchors.has(i)) ceiling = anchors.get(i)!;
        desired[i] = anchors.get(i) ?? Math.min(desired[i]!, ceiling);
        ceiling -= rowGaps[i]!;
      }
      // Some forwarded connections pass a boundary without a visible terminal
      // there. Their existing lanes still reserve space beside the port names.
      const keys = new Set(ordered.map((port) => endpointKey({ node_id: node.id, port_id: port.portId })));
      const incident = new Set(projection.edges.filter((edge) => keys.has(endpointKey(edge.source)) || keys.has(endpointKey(edge.target))).map((edge) => edge.id));
      let previousY = -Infinity;
      for (const [index, port] of ordered.entries()) {
        const forbidden: [number, number][] = [];
        let y = Math.max(desired[index]!, previousY + rowGaps[index]!,
          box.absoluteY + metric.headerHeight - (port.label.y - port.absoluteY) + port.label.clearance);
        const label = port.label, key = endpointKey({ node_id: node.id, port_id: port.portId });
        const rectangles = [{ x: label.x - label.clearance, y: label.y - label.clearance,
          width: label.width + 2 * label.clearance, height: label.height + 2 * label.clearance }];
        for (const source of [true, false]) if (projection.edges.some((edge) => endpointKey(source ? edge.source : edge.target) === key)) rectangles.push(endpointCorridor(port, source));
        for (const rect of rectangles) for (const route of routes) if (!incident.has(route.id)) for (const section of route.sections) for (let i = 1; i < section.length; i++) {
          const a = section[i - 1]!, b = section[i]!;
          if (Math.max(a.x, b.x) <= rect.x + epsilon || Math.min(a.x, b.x) >= rect.x + rect.width - epsilon) continue;
          const top = rect.y - port.absoluteY, bottom = top + rect.height;
          forbidden.push([Math.min(a.y, b.y) - bottom, Math.max(a.y, b.y) - top]);
        }
        for (const [low, high] of forbidden.sort((a, b) => a[0] - b[0]))
          if (y > low - epsilon && y < high + epsilon) y = high + 2 * epsilon;
        const bottom = y + Math.max(0, port.label.y - port.absoluteY + port.label.height + port.label.clearance) + laneGap;
        if (bottom > box.absoluteY + box.height + epsilon) additionalHeights.set(node.id,
          Math.max(additionalHeights.get(node.id) ?? 0, bottom - box.absoluteY - box.height));
        const delta = y - port.absoluteY;
        port.absoluteY = y; port.y += delta; port.label.y += delta;
        previousY = y;
      }
    }
  }
  if (additionalHeights.size) throw new BoundaryRowSpace(additionalHeights);
  const byRoute = new Map(routes.map((route) => [route.id, route]));
  const direct = new Set<string>();
  const sourceCounts = new Map<string, number>();
  for (const edge of projection.edges) sourceCounts.set(endpointKey(edge.source), (sourceCounts.get(endpointKey(edge.source)) ?? 0) + 1);
  const edgeById = new Map(projection.edges.map((edge) => [edge.id, edge]));
  const anticipatedSections = (route: Route) => {
    const edge = edgeById.get(route.id)!;
    const extra: Point[][] = [];
    for (const [endpoint, sourceEnd] of [[edge.source, true], [edge.target, false]] as const) {
      const port = positions.get(endpointKey(endpoint))!;
      if (port.absoluteY === original.get(port)) continue;
      for (const section of route.sections) {
        const terminal = sourceEnd ? section[0]! : section.at(-1)!, next = sourceEnd ? section[1] : section.at(-2);
        if (next && Math.abs(terminal.x - port.absoluteX) < epsilon && Math.abs(terminal.y - original.get(port)!) < epsilon)
          extra.push([{ x: port.absoluteX, y: port.absoluteY }, { x: next.x, y: port.absoluteY }]);
      }
    }
    return [...route.sections, ...extra];
  };
  const crosses = (a: Point, b: Point, rect: { x: number; y: number; width: number; height: number }) =>
    Math.max(a.x, b.x) > rect.x + epsilon && Math.min(a.x, b.x) < rect.x + rect.width - epsilon &&
    Math.max(a.y, b.y) > rect.y + epsilon && Math.min(a.y, b.y) < rect.y + rect.height - epsilon;
  for (const edge of projection.edges) {
    const source = positions.get(endpointKey(edge.source))!, target = positions.get(endpointKey(edge.target))!, route = byRoute.get(edge.id)!;
    if (!nodes.get(source.nodeId)!.expanded && !nodes.get(target.nodeId)!.expanded) continue;
    // Shared work keeps its genuine trunk and junctions. Simplify only local
    // forwarding repairs, not arbitrary interior routes from the layout engine.
    if (route.junctions.length || sourceCounts.get(endpointKey(edge.source))! > 1) continue;
    const left = source.absoluteX, right = target.absoluteX, y = source.absoluteY;
    if (right <= left) continue;
    if (route.sections.length === 1 && route.sections[0]!.length === 2 && original.get(source) === y && original.get(target) === target.absoluteY) continue;
    const start = { x: left, y }, end = { x: right, y: target.absoluteY };
    const candidates: Point[][] = [];
    if (Math.abs(y - target.absoluteY) < epsilon) candidates.push([start, end]);
    else if (original.get(source) !== y || original.get(target) !== target.absoluteY) {
      const departure = endpointCorridor(source, true), approach = endpointCorridor(target, false);
      const low = departure.x + departure.width + 2, high = approach.x - 2;
      if (low <= high) for (const x of [(low + high) / 2, low, high])
        candidates.push([start, { x, y }, { x, y: end.y }, end]);
    }
    for (const points of candidates) {
      const lines = points.slice(1).map((p, i) => [points[i]!, p] as const);
      const obstacle = boxes.some((box) => box.id !== source.nodeId && box.id !== target.nodeId && lines.some(([a, b]) => crosses(a, b,
        { x: box.absoluteX, y: box.absoluteY, width: box.width, height: nodes.get(box.id)!.expanded ? metrics.get(box.id)!.headerHeight : box.height })));
      const labelObstacle = routes.some((other) => other.id !== edge.id && other.labels?.some((label) => lines.some(([a, b]) => crosses(a, b, label))));
      const coincides = routes.some((other) => other.id !== edge.id && anticipatedSections(other).some((section) => section.slice(1).some((d, i) => {
        const c = section[i]!;
        return lines.some(([a, b]) => (Math.abs(a.y - b.y) < epsilon && Math.abs(c.y - d.y) < epsilon && Math.abs(a.y - c.y) < epsilon &&
          Math.min(Math.max(a.x, b.x), Math.max(c.x, d.x)) - Math.max(Math.min(a.x, b.x), Math.min(c.x, d.x)) > epsilon) ||
          (Math.abs(a.x - b.x) < epsilon && Math.abs(c.x - d.x) < epsilon && Math.abs(a.x - c.x) < epsilon &&
          Math.min(Math.max(a.y, b.y), Math.max(c.y, d.y)) - Math.max(Math.min(a.y, b.y), Math.min(c.y, d.y)) > epsilon));
      })));
      const candidate: Route = { ...route, sections: [points], junctions: [] };
      if (!obstacle && !labelObstacle && !coincides && !routeClearanceFailures(projection, ports, [candidate]).length) {
        route.sections = candidate.sections; direct.add(edge.id); break;
      }
    }
  }
  // Keep the original ELK route beyond each terminal gutter. Ordered doglegs
  // reconnect moved rows there without moving turns through interior bodies.
  for (const node of projection.nodes.filter((node) => node.expanded)) for (const side of ['left', 'right'] as const) {
    const ordered = ports.filter((port) => port.nodeId === node.id && port.side === side).sort((a, b) => a.y - b.y);
    if (!ordered.some((port) => Math.abs(port.absoluteY - original.get(port)!) >= epsilon)) continue;
    for (const sourceEnd of [true, false]) {
      const terminals = ordered.map((port) => ({ port, sections: projection.edges.flatMap((edge) => {
        if (direct.has(edge.id)) return [];
        if (endpointKey(sourceEnd ? edge.source : edge.target) !== endpointKey({ node_id: port.nodeId, port_id: port.portId })) return [];
        const route = byRoute.get(edge.id)!;
        return route.sections.flatMap((section) => {
          const points = sourceEnd ? section : [...section].reverse(), start = points[0]!;
          return Math.abs(start.x - port.absoluteX) < epsilon && Math.abs(start.y - original.get(port)!) < epsilon ? [{ section, points, route }] : [];
        });
      }) })).filter((terminal) => terminal.sections.length && Math.abs(terminal.port.absoluteY - original.get(terminal.port)!) >= epsilon);
      const direction = sourceEnd ? 1 : -1;
      const minimum = sourceEnd ? departureLength : approachLength;
      const clearance = Math.max(minimum, ...terminals.map(({ port }) => sourceEnd ?
        port.label.x > port.absoluteX ? port.label.x + port.label.width + 8 - port.absoluteX : minimum :
        port.label.x + port.label.width < port.absoluteX ? port.absoluteX - port.label.x + 8 : minimum));
      if (!terminals.length) continue;
      const portX = terminals[0]!.port.absoluteX;
      const top = Math.min(...terminals.flatMap(({ port }) => [port.absoluteY, original.get(port)!]));
      const bottom = Math.max(...terminals.flatMap(({ port }) => [port.absoluteY, original.get(port)!]));
      const bodyDistance = Math.min(Infinity, ...boxes.filter((box) => box.id !== node.id &&
        box.absoluteY < bottom + epsilon && box.absoluteY + (nodes.get(box.id)!.expanded ? metrics.get(box.id)!.headerHeight : box.height) > top - epsilon)
        .map((box) => direction > 0 ? box.absoluteX - portX - approachLength : portX - box.absoluteX - box.width - departureLength).filter((distance) => distance > epsilon),
        ...routes.flatMap((route) => route.labels ?? []).filter((label) => label.y < bottom + epsilon && label.y + label.height > top - epsilon)
          .map((label) => direction > 0 ? label.x - portX : portX - label.x - label.width).filter((distance) => distance > epsilon));
      const endDistance = Math.min(bodyDistance, ...terminals.flatMap(({ port, sections }) => sections.map(({ points }) => Math.abs(points[1]!.x - port.absoluteX))));
      const ranks = new Map<PortPosition, number>();
      // Moving a row down puts its turn before lower original rows; moving up
      // puts it after higher original rows. This is a monotone wiring permutation.
      const channelOrder = [...terminals].sort((a, b) => {
        const ad = a.port.absoluteY - original.get(a.port)!, bd = b.port.absoluteY - original.get(b.port)!;
        if (ad >= 0 && bd < 0) return -1;
        if (ad < 0 && bd >= 0) return 1;
        return ad >= 0 ? a.port.absoluteY - b.port.absoluteY : b.port.absoluteY - a.port.absoluteY;
      });
      for (const [index, { port }] of channelOrder.entries()) {
        const low = Math.min(port.absoluteY, original.get(port)!), high = Math.max(port.absoluteY, original.get(port)!);
        const overlapping = channelOrder.slice(0, index).filter(({ port: other }) =>
          Math.min(other.absoluteY, original.get(other)!) < high + epsilon && Math.max(other.absoluteY, original.get(other)!) > low - epsilon);
        ranks.set(port, Math.max(0, ...overlapping.map(({ port: other }) => ranks.get(other)! + 1)));
      }
      const channels = Math.max(...ranks.values()) + 1;
      const needed = laneGap * (channels + 1);
      const growth = Math.max(0, clearance + needed - endDistance);
      if (growth) {
        const at = portX + direction * (endDistance - 2 * epsilon);
        let root = node;
        while (root.parentId) root = nodes.get(root.parentId)!;
        // Exterior approach space grows toward its producer. The visible root
        // keeps its ELK origin, so reopening it does not move the camera anchor.
        growGutter(at, growth, !sourceEnd && at < byBox.get(root.id)!.absoluteX);
      }
      const available = endDistance + growth - clearance;
      for (const { port, sections } of terminals) {
        const oldY = original.get(port)!;
        if (Math.abs(port.absoluteY - oldY) < epsilon) continue;
        const offset = clearance + available * (ranks.get(port)! + 1) / (channels + 1);
        if (offset < clearance - epsilon) throw new Error('Boundary rows need more routing gutter. Collapse groups and retry.');
        for (const { section, points } of sections) {
          const x = port.absoluteX + direction * offset;
          points.splice(0, 1, { x: port.absoluteX, y: port.absoluteY }, { x, y: port.absoluteY }, { x, y: oldY });
          if (!sourceEnd) section.splice(0, section.length, ...points.reverse());
        }
      }
    }
  }
  const junctions = routes.flatMap((route) => route.junctions);
  for (const route of routes) route.sections = canonicalSections(route.sections, junctions);
  return widthGrowth;
}
