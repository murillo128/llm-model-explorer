import { expect, it } from 'vitest';
import { layoutGraph } from './auto-layout';
import { routeClearanceFailures } from './routing-clearance';
import { endpointKey } from './projection';
import { interfaceFixture } from '../../tests/architecture-interface-fixture';
import { makeExplicitFixture } from '../../tests/architecture-explicit-fixture';
import { rotaryContextFixture } from '../../tests/architecture-routing-fixture';
import type { Graph, Layout, Point } from './graph';

const port = (id: string, direction: 'input' | 'output') => ({ id, label: id, direction, shape: [{ kind: 'constant' as const, value: 16 }] });
const common = { parameter_ids: [], references: [], attributes: [], provenance: [] };

function segments(layout: Layout) {
  return layout.routes.flatMap((route) => route.sections.flatMap((section) => section.slice(1).map((point, index) =>
    ({ edgeId: route.id, a: section[index]!, b: point }))));
}
function properCross(a: { a: Point; b: Point }, b: { a: Point; b: Point }) {
  if (a.a.y !== a.b.y || b.a.x !== b.b.x) return false;
  return b.a.x > Math.min(a.a.x, a.b.x) && b.a.x < Math.max(a.a.x, a.b.x) &&
    a.a.y > Math.min(b.a.y, b.b.y) && a.a.y < Math.max(b.a.y, b.b.y);
}

it.each([false, true])('routes rotary sin and causal mask to exact context and internal ports with protected labels (dimensions=%s)', async (dimensions) => {
  const graph = rotaryContextFixture(), snapshot = JSON.stringify(graph);
  const layout = await layoutGraph(graph, { expanded: ['component'], dimensions, showContext: true });
  const again = await layoutGraph(graph, { expanded: ['component'], dimensions, showContext: true });
  expect(JSON.stringify(graph)).toBe(snapshot);
  expect(layout.projection.edges.flatMap((edge) => edge.originalEdgeIds).sort()).toEqual(graph.edges.map((edge) => edge.id).sort());
  expect(layout.projection.edges.map((edge) => [endpointKey(edge.source), endpointKey(edge.target)])).toEqual([
    [endpointKey(graph.edges[0]!.source), endpointKey(graph.edges[2]!.target)],
    [endpointKey(graph.edges[1]!.source), endpointKey(graph.edges[3]!.target)],
  ]);
  expect(layout.ports).toEqual(again.ports);
  expect(layout.routes).toEqual(again.routes);
  expect(routeClearanceFailures(layout.projection, layout.ports, layout.routes)).toEqual([]);
  const sin = layout.routes.find((route) => route.id.includes('sin-enter'))!;
  const source = layout.ports.find((position) => position.nodeId === 'sin' && position.portId === 'out')!;
  const destination = layout.ports.find((position) => position.nodeId === 'use' && position.portId === 'sin')!;
  const maskSource = layout.ports.find((position) => position.nodeId === 'mask' && position.portId === 'out')!;
  const maskDestination = layout.ports.find((position) => position.nodeId === 'use' && position.portId === 'causal mask')!;
  expect(Math.sign(source.absoluteY - maskSource.absoluteY)).toBe(Math.sign(destination.absoluteY - maskDestination.absoluteY));
  expect(sin.sections[0]![0]).toEqual({ x: source.absoluteX, y: source.absoluteY });
  expect(sin.sections[0]!.at(-1)).toEqual({ x: destination.absoluteX, y: destination.absoluteY });
  expect(segments(layout).filter((segment) => segment.edgeId === sin.id).length).toBeLessThanOrEqual(4);
});

it('protects many converted input rows, a nested nonzero instance and isolated boundary forwarding', async () => {
  const dense = interfaceFixture('dense', true);
  const nested = makeExplicitFixture({ count: 3 });
  const cases: [Graph, Parameters<typeof layoutGraph>[1]][] = [
    [dense, { expanded: ['model'], showUnused: true }],
    [dense, { expanded: ['model'], showUnused: true, dimensions: true }],
    [nested, { expanded: ['model', 'layer-2', 'layer-2.attention', 'layer-2.mlp'], showContext: true }],
    [nested, { scope: 'layer-2.attention', expanded: ['layer-2.attention'], dimensions: true }],
  ];
  for (const [graph, options] of cases) {
    const layout = await layoutGraph(graph, options);
    expect(routeClearanceFailures(layout.projection, layout.ports, layout.routes)).toEqual([]);
  }
});

it('rejects a bend beside a terminal instead of returning a misleading layout', async () => {
  const graph = rotaryContextFixture();
  const layout = await layoutGraph(graph, { expanded: ['component'] });
  const damaged = structuredClone(layout.routes);
  const route = damaged.find((candidate) => candidate.id.includes('sin-enter'))!;
  const start = route.sections[0]![0]!;
  route.sections[0]!.splice(1, 0, { x: start.x + 2, y: start.y }, { x: start.x + 2, y: start.y + 6 });
  expect(routeClearanceFailures(layout.projection, layout.ports, damaged).some((failure) => failure.includes('terminal turns'))).toBe(true);
});

it('permits a forced crossing only in open space between fixed, opposite-order terminal pairs', async () => {
  const graph: Graph = { graph_id: 'crossing-routing', scope: 'language_model', coverage: 'partial', symbols: [], parameters: [], repetitions: [], diagnostics: [],
    nodes: [
      { ...common, id: 'source', kind: 'operation', label: 'Source', ports: [port('top', 'output'), port('bottom', 'output')] },
      { ...common, id: 'target', kind: 'operation', label: 'Target', ports: [port('top', 'input'), port('bottom', 'input')] },
    ], edges: [
      { id: 'falling', source: { node_id: 'source', port_id: 'top' }, target: { node_id: 'target', port_id: 'bottom' }, kind: 'data', provenance: [] },
      { id: 'rising', source: { node_id: 'source', port_id: 'bottom' }, target: { node_id: 'target', port_id: 'top' }, kind: 'data', provenance: [] },
    ] };
  const layout = await layoutGraph(graph, { expanded: [] });
  // Fix the two terminal orders in opposition inside a bounded strip. Any
  // monotone paths between these endpoints intersect, so keep the crossing in
  // the gap after both departure corridors and before both approach corridors.
  const ports = structuredClone(layout.ports);
  const sourceTop = ports.find((position) => position.nodeId === 'source' && position.portId === 'top')!;
  const sourceBottom = ports.find((position) => position.nodeId === 'source' && position.portId === 'bottom')!;
  const left = sourceTop.absoluteX;
  const right = ports.find((position) => position.nodeId === 'target')!.absoluteX;
  const top = sourceTop.absoluteY, bottom = sourceBottom.absoluteY, middle = (top + bottom) / 2;
  for (const position of ports.filter((candidate) => candidate.nodeId === 'target')) {
    const delta = position.portId === 'bottom' ? bottom - top : top - bottom;
    position.y += delta; position.absoluteY += delta; position.label.y += delta;
  }
  const routes = [
    { id: 'connection:falling', sections: [[{ x: left, y: top }, { x: left + 14, y: top }, { x: left + 14, y: middle },
      { x: right - 18, y: middle }, { x: right - 18, y: bottom }, { x: right, y: bottom }]], junctions: [] },
    { id: 'connection:rising', sections: [[{ x: left, y: bottom }, { x: left + 18, y: bottom }, { x: left + 18, y: middle - 4 },
      { x: right - 16, y: middle - 4 }, { x: right - 16, y: top }, { x: right, y: top }]], junctions: [] },
  ];
  expect(routeClearanceFailures(layout.projection, ports, routes)).toEqual([]);
  const routed = { ...layout, routes };
  const falling = segments(routed).filter((segment) => segment.edgeId === 'connection:falling');
  const rising = segments(routed).filter((segment) => segment.edgeId === 'connection:rising');
  expect(falling.some((a) => rising.some((b) => properCross(a, b) || properCross(b, a)))).toBe(true);
});
