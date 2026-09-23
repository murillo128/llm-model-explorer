import { assertInterfaceCoverage } from '../../tests/architecture-invariants';
import { layoutGraph } from './auto-layout';
import { describe, expect, it } from 'vitest';
import { makeProjectionFixture } from '../../tests/architecture-projection-fixture';
import { makeExplicitFixture } from '../../tests/architecture-explicit-fixture';
import { summaryFixture } from '../../tests/architecture-summary-fixture';
import { interfaceFixture } from '../../tests/architecture-interface-fixture';
import { overviewFixture } from '../../tests/architecture-overview-fixture';
import { groupHeaderHeight, layerGap } from './auto-layout';
import { deriveMlpGroups } from './derived-groups';
import type { Box, Graph, Layout, Point } from './graph';
import { connectionSet, endpointKey, projectGraph, type ProjectionOptions } from './projection';
import { cardMetrics, cardSummary } from './card-summary';

const tolerance = 0.01;
const equals = (a: Point, b: Point) => Math.abs(a.x - b.x) < tolerance && Math.abs(a.y - b.y) < tolerance;
function box(layout: Layout, id: string): Box {
  const result = layout.boxes.find((b) => b.id === id);
  if (!result) throw new Error(`Missing visible box ${id}`);
  return result;
}
function horizontal(layout: Layout, stages: string[]) {
  for (let i = 1; i < stages.length; i++) {
    const previous = box(layout, stages[i - 1]!), next = box(layout, stages[i]!);
    expect(next.parentId, `${next.id} and ${previous.id} must be siblings`).toBe(previous.parentId);
    expect(next.absoluteX, `${previous.id} → ${next.id} must advance horizontally beyond the full previous container`)
      .toBeGreaterThanOrEqual(previous.absoluteX + previous.width + layerGap - tolerance);
  }
}
function segments(points: Point[][]) { return points.flatMap((section) => section.slice(1).map((point, i) => [section[i]!, point] as const)); }
function crosses(a: Point, b: Point, bounds: { x: number; y: number; width: number; height: number }) {
  const left = bounds.x + tolerance, right = bounds.x + bounds.width - tolerance;
  const top = bounds.y + tolerance, bottom = bounds.y + bounds.height - tolerance;
  if (Math.abs(a.x - b.x) < tolerance) return a.x > left && a.x < right && Math.max(a.y, b.y) > top && Math.min(a.y, b.y) < bottom;
  if (Math.abs(a.y - b.y) < tolerance) return a.y > top && a.y < bottom && Math.max(a.x, b.x) > left && Math.min(a.x, b.x) < right;
  return true; // This orthogonal router must not silently introduce diagonal obstacle crossings.
}
function overlap(a: readonly [Point, Point], b: readonly [Point, Point]) {
  const horizontalA = Math.abs(a[0].y - a[1].y) < tolerance, horizontalB = Math.abs(b[0].y - b[1].y) < tolerance;
  if (horizontalA !== horizontalB) return 0;
  const fixed = horizontalA ? 'y' : 'x', axis = horizontalA ? 'x' : 'y';
  if (Math.abs(a[0][fixed] - b[0][fixed]) > tolerance) return 0;
  return Math.min(Math.max(a[0][axis], a[1][axis]), Math.max(b[0][axis], b[1][axis])) -
    Math.max(Math.min(a[0][axis], a[1][axis]), Math.min(b[0][axis], b[1][axis]));
}
function properIntersection(a: readonly [Point, Point], b: readonly [Point, Point]) {
  const horizontalA = Math.abs(a[0].y - a[1].y) < tolerance, horizontalB = Math.abs(b[0].y - b[1].y) < tolerance;
  if (horizontalA && horizontalB) return Math.abs(a[0].y - b[0].y) < tolerance && overlap(a, b) > tolerance;
  if (!horizontalA && !horizontalB) return Math.abs(a[0].x - b[0].x) < tolerance && overlap(a, b) > tolerance;
  const horizontal = horizontalA ? a : b, vertical = horizontalA ? b : a;
  const x = vertical[0].x, y = horizontal[0].y;
  return x > Math.min(horizontal[0].x, horizontal[1].x) + tolerance &&
    x < Math.max(horizontal[0].x, horizontal[1].x) - tolerance &&
    y > Math.min(vertical[0].y, vertical[1].y) + tolerance &&
    y < Math.max(vertical[0].y, vertical[1].y) - tolerance;
}
function boundaryRoutes(layout: Layout, nodeId: string, side: 'left' | 'right', portIds?: string[]) {
  const edges = new Map(layout.projection.edges.map((edge) => [edge.id, edge]));
  const wanted = portIds && new Set(portIds);
  return layout.routes.flatMap((route) => {
    const edge = edges.get(route.id), endpoint = edge && (side === 'left' ? edge.source : edge.target);
    if (!edge || !endpoint || endpoint.node_id !== nodeId || wanted && !wanted.has(endpoint.port_id)) return [];
    return [{ edge, route }];
  });
}
function assertNoBoundaryCrossings(layout: Layout, nodeId: string, side: 'left' | 'right', portIds: string[]) {
  const routed = boundaryRoutes(layout, nodeId, side, portIds), failures: string[] = [];
  for (let i = 0; i < routed.length; i++) for (let j = i + 1; j < routed.length; j++) {
    const a = routed[i]!, b = routed[j]!;
    if (endpointKey(a.edge.source) === endpointKey(b.edge.source)) continue; // Genuine fan-out may share its source trunk.
    if (segments(a.route.sections).some((one) => segments(b.route.sections).some((two) => properIntersection(one, two)))) {
      failures.push(`${a.edge.id} crosses ${b.edge.id}`);
    }
  }
  expect(failures, `Independent signals must not cross the ${side} boundary of ${nodeId}`).toEqual([]);
}
function assertBoundaryOrder(layout: Layout, nodeId: string, side: 'left' | 'right', portIds: string[]) {
  const nodes = new Map(layout.projection.nodes.map((node) => [node.id, node]));
  const positions = new Map(layout.ports.map((port) => [endpointKey({ node_id: port.nodeId, port_id: port.portId }), port]));
  const inside = (id: string) => {
    for (let node = nodes.get(id); node; node = node.parentId ? nodes.get(node.parentId) : undefined) if (node.id === nodeId) return true;
    return false;
  };
  const groups = new Map<string, { anchorY: number; interiorYs: number[] }>();
  for (const { edge } of boundaryRoutes(layout, nodeId, side, portIds)) {
    const boundary = side === 'left' ? edge.source : edge.target, other = side === 'left' ? edge.target : edge.source;
    const anchor = positions.get(endpointKey(boundary)), interior = positions.get(endpointKey(other));
    if (!anchor || !interior || !inside(interior.nodeId)) continue;
    const current = groups.get(boundary.port_id) ?? { anchorY: anchor.absoluteY, interiorYs: [] };
    current.interiorYs.push(interior.absoluteY); groups.set(boundary.port_id, current);
  }
  const median = (values: number[]) => {
    const sorted = [...values].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
  };
  const ordered = [...groups.values()].sort((a, b) => median(a.interiorYs) - median(b.interiorYs));
  for (let i = 1; i < ordered.length; i++) expect(ordered[i]!.anchorY).toBeGreaterThanOrEqual(ordered[i - 1]!.anchorY - tolerance);
}
function stableLayout(layout: Layout) {
  return { boxes: layout.boxes, ports: layout.ports, routes: layout.routes, edgeIds: layout.edgeIds,
    width: layout.width, height: layout.height, projection: layout.projection };
}
function geometry(layout: Layout) {
  const nodes = new Map(layout.projection.nodes.map((n) => [n.id, n]));
  const ports = new Map(layout.ports.map((p) => [endpointKey({ node_id: p.nodeId, port_id: p.portId }), p]));
  const routes = new Map(layout.routes.map((r) => [r.id, r]));
  expect(new Set(layout.boxes.map((b) => b.id))).toEqual(new Set(nodes.keys()));
  expect(new Set(routes.keys())).toEqual(new Set(layout.projection.edges.map((e) => e.id)));
  const failures: string[] = [];
  for (const current of layout.boxes) {
    expect([current.absoluteX, current.absoluteY, current.width, current.height].every(Number.isFinite)).toBe(true);
    if (current.parentId) {
      const parent = box(layout, current.parentId);
      expect(current.x).toBeGreaterThanOrEqual(0); expect(current.y).toBeGreaterThanOrEqual(groupHeaderHeight);
      expect(current.x + current.width).toBeLessThanOrEqual(parent.width + tolerance);
      expect(current.y + current.height).toBeLessThanOrEqual(parent.height + tolerance);
      expect(current.absoluteX).toBeCloseTo(parent.absoluteX + current.x, 5);
      expect(current.absoluteY).toBeCloseTo(parent.absoluteY + current.y, 5);
    }
  }
  for (const edge of layout.projection.edges) {
    const route = routes.get(edge.id)!, source = ports.get(endpointKey(edge.source))!, target = ports.get(endpointKey(edge.target))!;
    expect(source, `Missing source ${endpointKey(edge.source)}`).toBeDefined(); expect(target).toBeDefined();
    const starts = route.sections.map((s) => s[0]!), ends = route.sections.map((s) => s.at(-1)!);
    expect(starts.some((p) => equals(p, { x: source.absoluteX, y: source.absoluteY })), `Route ${edge.id} misses its source port`).toBe(true);
    expect(ends.some((p) => equals(p, { x: target.absoluteX, y: target.absoluteY })), `Route ${edge.id} misses its destination port`).toBe(true);
    for (const [a, b] of segments(route.sections)) {
      if (Math.abs(a.x - b.x) >= tolerance && Math.abs(a.y - b.y) >= tolerance) failures.push(`Diagonal segment: ${edge.id}`);
      for (const obstacle of layout.boxes) {
        if (obstacle.id === edge.source.node_id || obstacle.id === edge.target.node_id) continue;
        const node = nodes.get(obstacle.id)!;
        const height = node.expanded ? groupHeaderHeight : obstacle.height;
        if (crosses(a, b, { x: obstacle.absoluteX, y: obstacle.absoluteY, width: obstacle.width, height })) {
          failures.push(`${edge.id} crosses ${node.expanded ? 'header' : 'node'} ${obstacle.id}`);
        }
      }
    }
  }
  expect([...new Set(failures)].slice(0, 12), 'Routes must avoid unrelated node bodies and group headers').toEqual([]);
  expect(Number.isFinite(layout.milliseconds)).toBe(true);
}
function distinguishSignals(layout: Layout) {
  const edges = new Map(layout.projection.edges.map((e) => [e.id, e]));
  const routed = layout.routes.map((r) => ({ edge: edges.get(r.id)!, segments: segments(r.sections) }));
  const conflicts: string[] = [];
  for (let i = 0; i < routed.length; i++) for (let j = i + 1; j < routed.length; j++) {
    const a = routed[i]!, b = routed[j]!;
    if (endpointKey(a.edge.source) === endpointKey(b.edge.source)) continue; // A genuine fan-out may share its own trunk.
    if (a.segments.some((one) => b.segments.some((two) => overlap(one, two) > tolerance))) conflicts.push(`${a.edge.id} coincides with ${b.edge.id}`);
  }
  expect(conflicts.slice(0, 12), 'Distinct source signals must remain individually traceable').toEqual([]);
}
function dimensionLabels(layout: Layout) {
  const rectangles = layout.routes.flatMap((r) => (r.labels ?? []).map((label) => ({ ...label, edgeId: r.id })));
  const intersects = (a: { x: number; y: number; width: number; height: number }, b: typeof a) =>
    a.x + a.width > b.x + tolerance && b.x + b.width > a.x + tolerance &&
    a.y + a.height > b.y + tolerance && b.y + b.height > a.y + tolerance;
  const failures: string[] = [];
  expect(rectangles).toHaveLength(layout.projection.edges.length);
  for (const [index, label] of rectangles.entries()) {
    expect(label.lines.length).toBeGreaterThan(0);
    expect([label.x, label.y, label.width, label.height].every(Number.isFinite)).toBe(true);
    for (const current of layout.boxes) {
      const expanded = layout.projection.nodes.find((n) => n.id === current.id)!.expanded;
      const bounds = { x: current.absoluteX, y: current.absoluteY, width: current.width, height: expanded ? groupHeaderHeight : current.height };
      if (intersects(label, bounds)) failures.push(`${label.edgeId} dimensions overlap ${current.id}`);
    }
    for (const other of rectangles.slice(index + 1)) if (intersects(label, other)) failures.push(`${label.edgeId} and ${other.edgeId} dimension labels overlap`);
    for (const route of layout.routes) if (route.id !== label.edgeId) {
      if (segments(route.sections).some(([a, b]) => crosses(a, b, label))) failures.push(`${route.id} crosses dimensions of ${label.edgeId}`);
    }
  }
  expect(failures.slice(0, 12), 'Generated dimension labels must avoid bodies, headers, other labels and unrelated routes').toEqual([]);
}
function layerStages(layout: Layout, graph: Graph, id: string) {
  const mlp = deriveMlpGroups(graph).find((g) => g.parentId === id)!;
  horizontal(layout, [`${id}.input-norm`, `${id}.attention`, `${id}.residual-1`, `${id}.post-norm`, mlp.id, `${id}.residual-2`]);
}

function invertedDeclarationFixture(): Graph {
  const port = (id: string, direction: 'input' | 'output') => ({ id, label: id, direction, shape: [] });
  const input = (id: string) => ({ id, parent_id: 'model', label: id, kind: 'input' as const, ports: [port('out', 'output')], parameter_ids: [], references: [], attributes: [], provenance: [] });
  const output = (id: string) => ({ id, parent_id: 'model', label: id, kind: 'output' as const, ports: [port('in', 'input')], parameter_ids: [], references: [], attributes: [], provenance: [] });
  const model = { id: 'model', label: 'Model', kind: 'group' as const, children: ['input-b', 'input-a', 'output-b', 'output-a', 'operation'], ports: [], parameter_ids: [], references: [], attributes: [], provenance: [] };
  const operation = { id: 'operation', parent_id: 'model', label: 'Operation', kind: 'operation' as const, operation: 'identity',
    ports: [port('a', 'input'), port('b', 'input'), port('a-out', 'output'), port('b-out', 'output')], parameter_ids: [], references: [], attributes: [], provenance: [] };
  return { graph_id: 'inverted-declaration-fixture', scope: 'language_model', coverage: 'complete', symbols: [], parameters: [], repetitions: [], diagnostics: [],
    nodes: [input('input-b'), input('input-a'), output('output-b'), output('output-a'), model, operation], edges: [
      { id: 'input-a-edge', source: { node_id: 'input-a', port_id: 'out' }, target: { node_id: 'operation', port_id: 'a' }, kind: 'data', provenance: [] },
      { id: 'input-b-edge', source: { node_id: 'input-b', port_id: 'out' }, target: { node_id: 'operation', port_id: 'b' }, kind: 'data', provenance: [] },
      { id: 'output-a-edge', source: { node_id: 'operation', port_id: 'a-out' }, target: { node_id: 'output-a', port_id: 'in' }, kind: 'data', provenance: [] },
      { id: 'output-b-edge', source: { node_id: 'operation', port_id: 'b-out' }, target: { node_id: 'output-b', port_id: 'in' }, kind: 'data', provenance: [] },
    ] };
}

function twoLevelBoundaryFixture(): Graph {
  const port = (id: string, direction: 'input' | 'output') => ({ id, label: id, direction, shape: [] });
  const input = (id: string) => ({ id, parent_id: 'model', label: id, kind: 'input' as const, ports: [port('out', 'output')], parameter_ids: [], references: [], attributes: [], provenance: [] });
  const model = { id: 'model', label: 'Model', kind: 'group' as const, children: ['input-b', 'input-a', 'layer'], ports: [], parameter_ids: [], references: [], attributes: [], provenance: [] };
  const layer = { id: 'layer', parent_id: 'model', label: 'Layer', kind: 'group' as const, children: ['operation'], ports: [port('b', 'input'), port('a', 'input')], parameter_ids: [], references: [], attributes: [], provenance: [] };
  const operation = { id: 'operation', parent_id: 'layer', label: 'Operation', kind: 'operation' as const, operation: 'identity',
    ports: [port('a', 'input'), port('b', 'input')], parameter_ids: [], references: [], attributes: [], provenance: [] };
  return { graph_id: 'two-level-boundary-fixture', scope: 'language_model', coverage: 'complete', symbols: [], parameters: [], repetitions: [], diagnostics: [],
    nodes: [input('input-b'), input('input-a'), model, layer, operation], edges: [
      { id: 'input-b-to-layer', source: { node_id: 'input-b', port_id: 'out' }, target: { node_id: 'layer', port_id: 'b' }, kind: 'data', provenance: [] },
      { id: 'layer-b-to-operation', source: { node_id: 'layer', port_id: 'b' }, target: { node_id: 'operation', port_id: 'b' }, kind: 'data', provenance: [] },
      { id: 'input-a-to-layer', source: { node_id: 'input-a', port_id: 'out' }, target: { node_id: 'layer', port_id: 'a' }, kind: 'data', provenance: [] },
      { id: 'layer-a-to-operation', source: { node_id: 'layer', port_id: 'a' }, target: { node_id: 'operation', port_id: 'a' }, kind: 'data', provenance: [] },
    ] };
}

function nestedBoundaryOrderingFixture(): Graph {
  const graph = interfaceFixture('dense', true);
  const model = graph.nodes.find((node) => node.id === 'model')!;
  if (model.kind !== 'group') throw new Error('Fixture model must be a group');
  const embedding = graph.nodes.find((node) => node.id === 'Embedding')!;
  const head = graph.nodes.find((node) => node.id === 'LM head')!;
  const output = graph.nodes.find((node) => node.id === 'logits')!;
  const inputIds = graph.nodes.filter((node) => node.kind === 'input').map((node) => node.id);
  const layer = { id: 'layer', parent_id: 'model', label: 'Layer', kind: 'group' as const, children: [...inputIds, 'Embedding', 'LM head', 'logits'], ports: [],
    parameter_ids: [], references: [], attributes: [], provenance: [] };
  for (const node of graph.nodes) if (inputIds.includes(node.id)) node.parent_id = 'layer';
  model.children = model.children.filter((id) => !inputIds.includes(id) && !['Embedding', 'LM head', 'logits'].includes(id)); model.children.push('layer');
  embedding.parent_id = 'layer'; head.parent_id = 'layer'; output.parent_id = 'layer';
  graph.nodes.push(layer);
  const fanout = { id: 'fanout', parent_id: 'layer', label: 'Fan-out consumer', kind: 'operation' as const, operation: 'identity', ports: [{ id: 'in', label: 'in', direction: 'input' as const, shape: [] }],
    parameter_ids: [], references: [], attributes: [], provenance: [] };
  layer.children.push(fanout.id); graph.nodes.push(fanout);
  graph.edges.push({ id: 'fanout-edge', source: { node_id: 'auxiliary_2', port_id: 'out' }, target: { node_id: fanout.id, port_id: 'in' }, kind: 'data', provenance: [] });
  const rootPort = (id: string, direction: 'input' | 'output') => ({ id, label: id, direction, shape: [] });
  const rootInput = (id: string) => ({ id, parent_id: 'model', label: id, kind: 'input' as const, ports: [rootPort('out', 'output')], parameter_ids: [], references: [], attributes: [], provenance: [] });
  const rootA = rootInput('outer-a'), rootB = rootInput('outer-b');
  const rootConsumer = { id: 'outer-consumer', parent_id: 'model', label: 'Outer consumer', kind: 'operation' as const, operation: 'identity', ports: [rootPort('a', 'input'), rootPort('b', 'input')],
    parameter_ids: [], references: [], attributes: [], provenance: [] };
  model.children.unshift(rootA.id, rootB.id, rootConsumer.id); graph.nodes.push(rootA, rootB, rootConsumer);
  graph.edges.push(
    { id: 'outer-a-edge', source: { node_id: rootA.id, port_id: 'out' }, target: { node_id: rootConsumer.id, port_id: 'a' }, kind: 'data', provenance: [] },
    { id: 'outer-b-edge', source: { node_id: rootB.id, port_id: 'out' }, target: { node_id: rootConsumer.id, port_id: 'b' }, kind: 'data', provenance: [] });
  graph.graph_id = 'nested-boundary-ordering-fixture'; return graph;
}

describe('generated horizontal graph geometry', () => {
  it('preserves routed order at an expanded model boundary without remapping interface identities', async () => {
    const graph = invertedDeclarationFixture();
    const layout = await layoutGraph(graph, { expanded: ['model'], showUnused: true });
    const ports = layout.projection.nodes.find((node) => node.id === 'model')!.ports;
    const inputs = ports.filter((port) => port.interfaces?.length && port.direction === 'input').map((port) => port.id);
    const outputs = ports.filter((port) => port.interfaces?.length && port.direction === 'output').map((port) => port.id);
    geometry(layout); distinguishSignals(layout);
    assertBoundaryOrder(layout, 'model', 'left', inputs); assertNoBoundaryCrossings(layout, 'model', 'left', inputs);
    assertBoundaryOrder(layout, 'model', 'right', outputs); assertNoBoundaryCrossings(layout, 'model', 'right', outputs);
    expect(boundaryRoutes(layout, 'model', 'left', inputs)).toHaveLength(2); expect(boundaryRoutes(layout, 'model', 'right', outputs)).toHaveLength(2);
    expect(new Set(layout.projection.edges.filter((edge) => inputs.includes(edge.source.port_id)).flatMap((edge) => edge.paths.flatMap((path) => path.map((source) => source.source.node_id))))).toEqual(new Set(['input-a', 'input-b']));
    expect(new Set(layout.projection.edges.filter((edge) => outputs.includes(edge.target.port_id)).flatMap((edge) => edge.paths.flatMap((path) => path.map((source) => source.target.node_id))))).toEqual(new Set(['output-a', 'output-b']));
  });

  it('keeps nested boundary order independently, preserves fan-out and is deterministic', async () => {
    const graph = nestedBoundaryOrderingFixture();
    const first = await layoutGraph(graph, { expanded: ['model', 'layer'], showUnused: true });
    const second = await layoutGraph(graph, { expanded: ['model', 'layer'], showUnused: true });
    const modelPorts = first.projection.nodes.find((node) => node.id === 'model')!.ports.filter((port) => port.id.includes('outer-a') || port.id.includes('outer-b')).map((port) => port.id);
    const layerPorts = first.projection.nodes.find((node) => node.id === 'layer')!.ports.filter((port) => port.id.includes('auxiliary_2') || port.id.includes('auxiliary_3')).map((port) => port.id);
    geometry(first); distinguishSignals(first);
    assertBoundaryOrder(first, 'model', 'left', modelPorts); assertBoundaryOrder(first, 'layer', 'left', layerPorts);
    assertNoBoundaryCrossings(first, 'model', 'left', modelPorts); assertNoBoundaryCrossings(first, 'layer', 'left', layerPorts);
    expect(boundaryRoutes(first, 'model', 'left', modelPorts)).toHaveLength(2);
    expect(boundaryRoutes(first, 'layer', 'left', layerPorts)).toHaveLength(3);
    const fanout = first.projection.edges.filter((edge) => edge.source.node_id === 'layer' && edge.source.port_id.includes('auxiliary_2'));
    expect(fanout).toHaveLength(2); expect(new Set(fanout.map((edge) => endpointKey(edge.source))).size).toBe(1);
    expect(fanout.flatMap((edge) => edge.paths.flatMap((path) => path.map((source) => source.source.node_id)))).toEqual(['auxiliary_2', 'auxiliary_2']);
    expect(stableLayout(second)).toEqual(stableLayout(first));
  });

  it('keeps the same signals ordered across two expanded hierarchy boundaries', async () => {
    const graph = twoLevelBoundaryFixture();
    const first = await layoutGraph(graph, { expanded: ['model', 'layer'], showUnused: true });
    const second = await layoutGraph(graph, { expanded: ['model', 'layer'], showUnused: true });
    const signals = new Set(['input-a', 'input-b']);
    const model = first.projection.nodes.find((node) => node.id === 'model')!;
    const layer = first.projection.nodes.find((node) => node.id === 'layer')!;
    const modelPorts = model.ports.filter((port) => port.interfaces?.some((id) => signals.has(id))).map((port) => port.id);
    const layerPorts = layer.ports.filter((port) => port.interfaces?.some((id) => signals.has(id))).map((port) => port.id);
    const boundaryEvidence = (nodeId: string, side: 'left' | 'right', portIds: string[]) => {
      const node = first.projection.nodes.find((candidate) => candidate.id === nodeId)!;
      return new Set(boundaryRoutes(first, nodeId, side, portIds).flatMap(({ edge }) => {
        const endpoint = side === 'left' ? edge.source : edge.target;
        const signal = node.ports.find((port) => port.id === endpoint.port_id)?.interfaces?.find((id) => signals.has(id));
        return signal ? [{ signal, path: edge.paths[0]! }] : [];
      }));
    };
    geometry(first); distinguishSignals(first);
    assertBoundaryOrder(first, 'model', 'left', modelPorts); assertNoBoundaryCrossings(first, 'model', 'left', modelPorts);
    assertBoundaryOrder(first, 'layer', 'left', layerPorts); assertNoBoundaryCrossings(first, 'layer', 'left', layerPorts);
    expect(boundaryRoutes(first, 'model', 'left', modelPorts)).toHaveLength(2);
    expect(boundaryRoutes(first, 'layer', 'left', layerPorts)).toHaveLength(2);
    const modelEvidence = boundaryEvidence('model', 'left', modelPorts), layerEvidence = boundaryEvidence('layer', 'left', layerPorts);
    expect(new Set([...modelEvidence].map(({ signal }) => signal))).toEqual(signals);
    expect(new Set([...layerEvidence].map(({ signal }) => signal))).toEqual(signals);
    expect(new Map([...modelEvidence].map(({ signal, path }) => [signal, path[0]!.source.node_id]))).toEqual(new Map([['input-a', 'input-a'], ['input-b', 'input-b']]));
    expect(new Map([...layerEvidence].map(({ signal, path }) => [signal, path.at(-1)!.target.port_id]))).toEqual(new Map([['input-a', 'a'], ['input-b', 'b']]));
    expect(stableLayout(second)).toEqual(stableLayout(first));
  });

  it.each([false, true])('routes complete isolated interfaces without overlapping signals (dimensions=%s)', async (dimensions) => {
    const graph = interfaceFixture('hybrid', true);
    const declared = graph.nodes.find((n) => n.id === 'language')!.ports;
    for (const expanded of [[], ['language']]) {
      const layout = await layoutGraph(graph, { scope: 'language', expanded, showUnused: true, dimensions });
      geometry(layout); distinguishSignals(layout);
      for (const direction of ['input', 'output']) {
        const ordered = layout.ports.filter((p) => p.nodeId === 'language' && p.side === (direction === 'input' ? 'left' : 'right')).sort((a, b) => a.y - b.y);
        if (!expanded.length) expect(ordered.map((p) => p.portId)).toEqual(declared.filter((p) => p.direction === direction).map((p) => p.id));
        else assertNoBoundaryCrossings(layout, 'language', direction === 'input' ? 'left' : 'right', ordered.map((p) => p.portId));
        for (let i = 1; i < ordered.length; i++) expect(ordered[i]!.y - ordered[i - 1]!.y).toBeGreaterThanOrEqual(20);
      }
    }
  });

  it.each([false, true])('reserves group-owned summary space above children and boundary ports (dimensions=%s)', async (dimensions) => {
    const graph = summaryFixture(), group = graph.nodes[0]!;
    group.parameter_ids = ['norm-weight'];
    group.formula = 'Explicitly supplied group signature';
    group.ports = [{ id: 'x', label: 'x', direction: 'input', shape: null }, { id: 'out', label: 'out', direction: 'output', shape: null }];
    graph.edges.push(
      { id: 'enter', source: { node_id: 'block', port_id: 'x' }, target: { node_id: 'norm', port_id: 'x' }, kind: 'data', provenance: [] },
      { id: 'leave', source: { node_id: 'linear', port_id: 'out' }, target: { node_id: 'block', port_id: 'out' }, kind: 'data', provenance: [] });
    const layout = await layoutGraph(graph, { expanded: ['block'], dimensions, showUnused: true });
    const parent = box(layout, 'block');
    // Title + formula + a port row + one owned tensor row, with extra shape lines.
    const contentBottom = dimensions ? 164 : 132;
    for (const id of ['norm', 'linear']) {
      expect(box(layout, id).y).toBeGreaterThan(contentBottom);
      expect(box(layout, id).absoluteY + box(layout, id).height).toBeLessThan(parent.absoluteY + parent.height);
    }
    for (const port of layout.ports.filter((p) => p.nodeId === 'block')) expect(port.y).toBeGreaterThan(contentBottom);
    horizontal(layout, ['norm', 'linear']); geometry(layout);
  });
  it('isolates the same component geometry regardless of surrounding model size', async () => {
    const small = makeExplicitFixture({ count: 4 }), large = makeExplicitFixture({ count: 48 });
    for (const scope of ['layer-3.attention', 'layer-3.mlp', 'layer-3.attention.core']) {
      const options = { scope, expanded: [scope] };
      const before = await layoutGraph(small, options), after = await layoutGraph(large, options);
      expect(after.boxes).toEqual(before.boxes); expect(after.ports).toEqual(before.ports);
      expect(after.routes).toEqual(before.routes);
      expect([after.width, after.height]).toEqual([before.width, before.height]);
      expect(before.boxes.some((box) => box.id === 'model' || box.id === 'layer-3')).toBe(false);
      geometry(before); distinguishSignals(before);
      if (scope.endsWith('.mlp')) horizontal(before, ['layer-3.gate', 'layer-3.silu', 'layer-3.multiply', 'layer-3.down']);
      if (scope.endsWith('.attention')) horizontal(before, ['layer-3.attention.Q', 'layer-3.attention.rope-Q', 'layer-3.attention.core', 'layer-3.attention.output']);
    }
  }, 30_000);
  it.each([16, 29])('keeps explicit Attention/MLP boundaries horizontal at width %i', async (hiddenSize) => {
    const graph = makeExplicitFixture({ count: 2, hiddenSize, variants: ['full_attention', 'linear_attention'] });
    for (const index of [0, 1]) {
      const id = `layer-${index}`;
      const layout = await layoutGraph(graph, { expanded: ['model', id, `${id}.attention`, `${id}.mlp`] });
      horizontal(layout, [`${id}.input-norm`, `${id}.attention`, `${id}.residual-1`, `${id}.post-norm`, `${id}.mlp`, `${id}.residual-2`]);
      horizontal(layout, [`${id}.gate`, `${id}.silu`, `${id}.multiply`, `${id}.down`]);
      horizontal(layout, [`${id}.up`, `${id}.multiply`]);
      geometry(layout); distinguishSignals(layout);
    }
  });
  it('lays out the model overview and a bounded serial-layer window in actual dependency order', async () => {
    const graph = makeProjectionFixture({ count: 7 });
    const overview = await layoutGraph(graph, { expanded: ['model'] });
    const stack = overview.projection.nodes.find((n) => n.presentation === 'repetition')!;
    horizontal(overview, ['embedding', stack.id, 'final-norm']); horizontal(overview, ['model', 'head']); geometry(overview); distinguishSignals(overview);
    const window = await layoutGraph(graph, { expanded: ['model'], repetitions: { 'decoder-layers': { start: 2, count: 3 } } });
    horizontal(window, ['embedding', 'repeat:decoder-layers:0:1', 'layer-2', 'layer-3', 'layer-4', 'repeat:decoder-layers:5:6', 'final-norm']);
    geometry(window); distinguishSignals(window);
  }, 30_000);

  it.each([0, 3])('places every serial scope horizontally inside expanded layer %i and nested attention/MLP', async (index) => {
    const graph = makeProjectionFixture({ count: 4 }), id = `layer-${index}`;
    const mlp = deriveMlpGroups(graph).find((g) => g.parentId === id)!;
    const layout = await layoutGraph(graph, { expanded: ['model', id, `${id}.attention`, mlp.id] });
    layerStages(layout, graph, id);
    horizontal(layout, [`${id}.gate`, `${id}.silu`, `${id}.multiply`, `${id}.down`]);
    horizontal(layout, [`${id}.up`, `${id}.multiply`]);
    if (index === 0) horizontal(layout, [`${id}.attention.input`, `${id}.attention.conv`, `${id}.attention.delta`, `${id}.attention.output`]);
    else {
      horizontal(layout, [`${id}.attention.Q`, `${id}.attention.rope-Q`, `${id}.attention.core`, `${id}.attention.output`]);
      horizontal(layout, [`${id}.attention.K`, `${id}.attention.rope-K`, `${id}.attention.core`]);
      horizontal(layout, [`${id}.attention.V`, `${id}.attention.core`]);
      // Q, K and V are separate branches, not a fabricated serial chain.
      const branchYs = ['Q', 'K', 'V'].map((name) => box(layout, `${id}.attention.${name}`).absoluteY);
      expect(new Set(branchYs).size).toBe(3);
    }
    geometry(layout); distinguishSignals(layout);
  }, 30_000);

  it('keeps first/last selection and changed count, order and dimensions topology-driven', async () => {
    const graph = makeProjectionFixture({ count: 5, hiddenSize: 29,
      variants: ['full_attention', 'full_attention', 'linear_attention', 'full_attention', 'linear_attention'] });
    for (const index of [0, 4]) {
      const id = `layer-${index}`, layout = await layoutGraph(graph, { expanded: ['model', id] });
      layerStages(layout, graph, id);
      horizontal(layout, index === 0 ? ['embedding', id, 'repeat:decoder-layers:1:4', 'final-norm'] : ['embedding', 'repeat:decoder-layers:0:3', id, 'final-norm']);
      geometry(layout); distinguishSignals(layout);
    }
  }, 30_000);

  it('preserves serial placement recursively for all 24 instances in explicit exhaustive mode', async () => {
    const graph = makeProjectionFixture(), layout = await layoutGraph(graph, { expanded: [], exhaustive: true });
    horizontal(layout, ['embedding', ...Array.from({ length: 24 }, (_, i) => `layer-${i}`), 'final-norm']);
    for (let i = 0; i < 24; i++) {
      const id = `layer-${i}`;
      horizontal(layout, [`${id}.input-norm`, `${id}.attention`, `${id}.residual-1`, `${id}.post-norm`, `${id}.gate`, `${id}.silu`, `${id}.multiply`, `${id}.down`, `${id}.residual-2`]);
      horizontal(layout, [`${id}.up`, `${id}.multiply`]);
      if (i % 4 !== 3) horizontal(layout, [`${id}.attention.input`, `${id}.attention.conv`, `${id}.attention.delta`, `${id}.attention.output`]);
      else horizontal(layout, [`${id}.attention.Q`, `${id}.attention.rope-Q`, `${id}.attention.core`, `${id}.attention.output`]);
    }
    assertInterfaceCoverage(graph, layout.projection); expect(new Set(layout.edgeIds)).toEqual(new Set(graph.edges.map((e) => e.id)));
    geometry(layout);
  }, 60_000);

  it('retains both independent stacks and their own horizontal layer order', async () => {
    const graph = makeProjectionFixture({ count: 3, secondStack: 2 });
    const layout = await layoutGraph(graph, { expanded: ['model', 'encoder', 'predictor'],
      repetitions: { 'encoder-layers': { start: 0, count: 3 }, 'predictor-layers': { start: 0, count: 2 } } });
    horizontal(layout, ['embedding', 'encoder', 'predictor', 'final-norm']);
    horizontal(layout, ['encoder.layer-0', 'encoder.layer-1', 'encoder.layer-2']); horizontal(layout, ['predictor.layer-0', 'predictor.layer-1']);
    geometry(layout); distinguishSignals(layout);
  }, 30_000);

  it('excludes collapsed descendants from bounds and returns stable geometry after an expansion round trip', async () => {
    const graph = makeProjectionFixture({ count: 4 }), options: ProjectionOptions = { expanded: ['model'] };
    const before = await layoutGraph(graph, options);
    const changed = structuredClone(graph);
    for (const n of changed.nodes) if (n.parent_id?.startsWith('layer-')) n.label = 'Hidden label with a deliberately much larger display width and height';
    const hiddenChanged = await layoutGraph(changed, options);
    expect(hiddenChanged.boxes).toEqual(before.boxes); expect(hiddenChanged.routes).toEqual(before.routes);
    await layoutGraph(graph, { expanded: ['model', 'layer-3', 'layer-3.attention'] });
    const after = await layoutGraph(graph, options);
    expect(after.boxes).toEqual(before.boxes); expect(after.ports).toEqual(before.ports); expect(after.routes).toEqual(before.routes);
  }, 30_000);

  it.each(['overview', 'linear', 'full'])('routes dimensions with clear labels and unchanged source connections in the %s view', async (view) => {
    const graph = makeProjectionFixture({ count: 4, hiddenSize: 29 });
    // Unknown rank is valid contract metadata and must remain distinguishable from the source shape.
    graph.nodes.find((n) => n.id === 'head')!.ports.find((p) => p.id === 'x')!.shape = null;
    const index = view === 'linear' ? 0 : 3, id = `layer-${index}`;
    const mlp = deriveMlpGroups(graph).find((g) => g.parentId === id)!;
    const options: ProjectionOptions = { expanded: view === 'overview' ? ['model'] : ['model', id, `${id}.attention`, mlp.id] };
    const before = await layoutGraph(graph, options), dimensions = await layoutGraph(graph, { ...options, dimensions: true });
    expect(before.routes.every((r) => !r.labels?.length)).toBe(true);
    expect(dimensions.projection).toEqual(before.projection); expect(dimensions.edgeIds).toEqual(before.edgeIds);
    if (view === 'overview') horizontal(dimensions, ['embedding', 'repeat:decoder-layers:0:3', 'final-norm']);
    else layerStages(dimensions, graph, id);
    const output = dimensions.projection.edges.find((e) => e.target.node_id === 'head')!;
    expect(dimensions.routes.find((r) => r.id === output.id)!.labels![0]!.lines).toEqual(['source [B × S × 29]', 'target ? (unknown rank)']);
    expect(dimensions.routes.some((r) => r.labels?.some((l) => l.lines.length === 1 && l.lines[0] === '[B × S × 29]'))).toBe(true);
    geometry(dimensions); distinguishSignals(dimensions); dimensionLabels(dimensions);
    const after = await layoutGraph(graph, options);
    expect(after.boxes).toEqual(before.boxes); expect(after.routes).toEqual(before.routes);
  }, 30_000);

  it('handles valid open group input/output interfaces without moving them to the opposite side', async () => {
    const graph: Graph = { graph_id: 'open-interface-fixture', scope: 'language_model', coverage: 'complete', symbols: [], parameters: [], repetitions: [], diagnostics: [],
      nodes: [
        { id: 'scope', kind: 'group', label: 'Open interface', children: ['operation'], ports: [{ id: 'in', direction: 'input', label: 'in', shape: [] }, { id: 'out', direction: 'output', label: 'out', shape: [] }], parameter_ids: [], references: [], attributes: [], provenance: [] },
        { id: 'operation', parent_id: 'scope', kind: 'operation', label: 'Operation', operation: 'linear', ports: [{ id: 'in', direction: 'input', label: 'in', shape: [] }, { id: 'out', direction: 'output', label: 'out', shape: [] }], parameter_ids: [], references: [], attributes: [], provenance: [] },
      ], edges: [
        { id: 'enter', source: { node_id: 'scope', port_id: 'in' }, target: { node_id: 'operation', port_id: 'in' }, kind: 'data', provenance: [] },
        { id: 'leave', source: { node_id: 'operation', port_id: 'out' }, target: { node_id: 'scope', port_id: 'out' }, kind: 'data', provenance: [] },
      ] };
    const projection = projectGraph(graph, { expanded: ['scope'] });
    expect(projection.edges.map((e) => e.paths.map((p) => p.map((s) => s.id)))).toEqual([[['enter']], [['leave']]]);
    const layout = await layoutGraph(graph, { expanded: ['scope'] });
    const input = layout.ports.find((p) => p.nodeId === 'scope' && p.portId === 'in')!, output = layout.ports.find((p) => p.nodeId === 'scope' && p.portId === 'out')!;
    expect(input.side).toBe('left'); expect(output.side).toBe('right');
    expect(input.absoluteX).toBeLessThan(box(layout, 'operation').absoluteX);
    expect(output.absoluteX).toBeGreaterThan(box(layout, 'operation').absoluteX + box(layout, 'operation').width);
    geometry(layout);
  });
});

it.each([interfaceFixture('dense', true), interfaceFixture('visual'), overviewFixture('training')])(
  'places visible children below their actual header without reserving a second interface band: $graph_id', async (graph) => {
    const layout = await layoutGraph(graph, { expanded: graph.nodes.filter((n) => n.kind === 'group').map((n) => n.id), showUnused: true });
    geometry(layout); distinguishSignals(layout);
    for (const node of layout.projection.nodes.filter((n) => n.expanded)) {
      const children = layout.boxes.filter((b) => b.parentId === node.id);
      const header = cardMetrics(node, cardSummary(node.record, new Map()), false).headerHeight;
      const top = Math.min(...children.map((b) => b.y));
      expect(top).toBeGreaterThanOrEqual(header + 16);
      expect(top, `${node.id} has no route requiring a large empty top band`).toBeLessThanOrEqual(header + 40);
    }
  });

it.each([interfaceFixture('dense', true), interfaceFixture('hybrid', true), interfaceFixture('visual')])(
  'keeps internal port label rectangles above their cables with shared metrics: $graph_id', async (graph) => {
    const parameters = new Map(graph.parameters.map((parameter) => [parameter.id, parameter]));
    const annotated = new Set([...graph.repetitions.flatMap((r) => r.instances.map((instance) => instance.node_id)),
      ...graph.diagnostics.flatMap((diagnostic) => diagnostic.node_id ? [diagnostic.node_id] : [])]);
    for (const dimensions of [false, true]) {
      const layout = await layoutGraph(graph, { expanded: graph.nodes.filter((node) => node.kind === 'group').map((node) => node.id),
        showUnused: true, dimensions });
      expect(layout.ports.some((port) => port.side === 'left' && port.label.raised)).toBe(true);
      expect(layout.ports.some((port) => port.side === 'right' && port.label.raised)).toBe(true);
      for (const node of layout.projection.nodes) {
        const box = layout.boxes.find((candidate) => candidate.id === node.id)!;
        const positions = layout.ports.filter((port) => port.nodeId === node.id);
        const raised = new Set(positions.filter((port) => port.label.raised).map((port) => port.portId));
        const metrics = cardMetrics(node, cardSummary(node.record, parameters), dimensions, annotated.has(node.id), raised);
        for (const [index, port] of positions.entries()) {
          const label = port.label, metric = metrics.portLabels[port.portId]!;
          expect(label).toEqual({ x: port.absoluteX + (port.side === 'left' ? 9 : -9 - metric.width),
            y: port.absoluteY + metric.top, width: metric.width, height: metric.height,
            clearance: metric.clearance, raised: metric.raised });
          if (node.expanded) expect(label.y).toBeGreaterThanOrEqual(box.absoluteY + metrics.headerHeight + 4);
          if (label.raised) {
            expect(label.y + label.height + label.clearance).toBeLessThanOrEqual(port.absoluteY);
            for (const edge of layout.projection.edges.filter((candidate) =>
              endpointKey(candidate.source) === endpointKey({ node_id: node.id, port_id: port.portId }) ||
              endpointKey(candidate.target) === endpointKey({ node_id: node.id, port_id: port.portId }))) {
              const route = layout.routes.find((candidate) => candidate.id === edge.id)!;
              // The boundary's own cable may bend after leaving the port, but
              // no section may pass through the displayed label itself.
              expect(segments(route.sections).some(([a, b]) => crosses(a, b, label)), `${node.id}.${port.portId} / ${edge.id}`).toBe(false);
            }
          } else expect(label.y).toBe(port.absoluteY - 8);
          for (const other of positions.slice(index + 1)) if (other.side === port.side) {
            expect(label.y + label.height + 4 <= other.label.y || other.label.y + other.label.height + 4 <= label.y,
              `${node.id}.${port.portId} overlaps ${other.portId}`).toBe(true);
          }
        }
      }
    }
  });

it('bounds long labels and dimensions without moving a collapsed external terminal', async () => {
  const graph = interfaceFixture('hybrid');
  graph.nodes.find((node) => node.id === 'positions')!.label = 'A very long declared position interface name';
  const collapsed = await layoutGraph(graph, { expanded: [], dimensions: false });
  const expanded = await layoutGraph(graph, { expanded: ['model', 'language'], dimensions: true, showUnused: true });
  expect(collapsed.ports.filter((port) => port.nodeId === 'model').every((port) => !port.label.raised)).toBe(true);
  expect(expanded.ports.some((port) => port.label.raised && port.label.width === 120)).toBe(true);
  for (const layout of [collapsed, expanded]) for (const port of layout.ports) {
    expect(port.label.width).toBeLessThanOrEqual(120);
    expect(port.label.height).toBe(layout === expanded ? 32 : 16);
  }
});

it('reserves disjoint opposite-side labels on an ordinary operation card', async () => {
  const graph = interfaceFixture('dense');
  const operation = graph.nodes.find((node) => node.id === 'Embedding')!;
  operation.ports.find((port) => port.id === 'Token IDs')!.label = 'An unusually long attention mask input name';
  operation.ports.find((port) => port.id === 'out')!.label = 'An unusually long hidden state output name';
  for (const dimensions of [false, true]) {
    const layout = await layoutGraph(graph, { expanded: ['model'], showUnused: true, dimensions });
    const node = layout.projection.nodes.find((candidate) => candidate.id === operation.id)!;
    const metrics = cardMetrics(node, cardSummary(operation, new Map()), dimensions);
    const box = layout.boxes.find((candidate) => candidate.id === operation.id)!;
    const input = layout.ports.find((port) => port.nodeId === operation.id && port.portId === 'Token IDs')!;
    const output = layout.ports.find((port) => port.nodeId === operation.id && port.portId === 'out')!;
    expect(metrics.portLabels['Token IDs']!.width).toBe(120);
    expect(metrics.portLabels.out!.width).toBe(120);
    expect(box.width).toBeGreaterThanOrEqual(288);
    expect(box.width).toBe(metrics.width);
    expect(input.label.y).toBe(output.label.y);
    expect(output.label.x - (input.label.x + input.label.width)).toBeGreaterThanOrEqual(8);
    expect(input.label.x).toBeGreaterThan(box.absoluteX);
    expect(output.label.x + output.label.width).toBeLessThan(box.absoluteX + box.width);
    expect(input.label.raised).toBe(false);
    expect(output.label.raised).toBe(false);
  }
});

it('keeps equal names, an unused port and isolated forwarding bound to their exact ports', async () => {
  const graph = interfaceFixture('hybrid', true);
  graph.nodes.find((node) => node.id === 'positions')!.label = 'Shared name';
  graph.nodes.find((node) => node.id === 'mask')!.label = 'Shared name';
  for (const port of graph.nodes.find((node) => node.id === 'language')!.ports.filter((port) => ['positions', 'mask'].includes(port.id))) {
    port.label = 'Shared name';
  }
  const embedding = graph.nodes.find((node) => node.id === 'Embedding')!;
  embedding.ports.push({ ...embedding.ports[0]!, id: 'unused', label: 'Unused label' });
  for (const options of [{ expanded: ['model', 'language'], showUnused: true },
    { scope: 'language', expanded: ['language'], showUnused: true }]) {
    const layout = await layoutGraph(graph, options);
    const owner = layout.projection.nodes.find((node) => node.id === 'language')!;
    const first = owner.ports.find((port) => port.id === 'positions')!, second = owner.ports.find((port) => port.id === 'mask')!;
    expect(first.interfaceLabel ?? first.label).toBe(second.interfaceLabel ?? second.label);
    const left = { node_id: owner.id, port_id: first.id }, right = { node_id: owner.id, port_id: second.id };
    const firstEdges = connectionSet(layout.projection, { port: left });
    const secondEdges = connectionSet(layout.projection, { port: right });
    expect(firstEdges.length).toBeGreaterThan(0);
    expect(secondEdges.length).toBeGreaterThan(0);
    expect(firstEdges).not.toEqual(secondEdges);
    const unused = { node_id: embedding.id, port_id: 'unused' };
    expect(layout.ports.find((port) => port.nodeId === unused.node_id && port.portId === unused.port_id)?.label.width).toBeGreaterThan(0);
    expect(connectionSet(layout.projection, { port: unused })).toEqual([]);
  }
});
