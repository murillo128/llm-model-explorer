import { expect, it } from 'vitest';
import { referenceFixture } from '../../tests/architecture-fixtures';
import { layoutGraph } from './auto-layout';
import { endpointKey, projectGraph } from './projection';

it('lays out every concrete expert and route in a large repeated-MoE graph', async () => {
  const graph = referenceFixture('glm4').graph;
  for (const expert of graph.nodes.filter((node) => /^layer-0-\d+-expert-\d+$/.test(node.id))) {
    const layer = expert.id.slice(0, expert.id.indexOf('-expert-'));
    for (const suffix of ['a', 'b']) {
      const input = `in-${suffix}`, output = `out-${suffix}`;
      expert.ports.push({ ...expert.ports[0]!, id: input, label: input },
        { ...expert.ports[1]!, id: output, label: output });
      graph.edges.push({ id: `${expert.id}-route-${input}`, kind: 'data', provenance: [],
        source: { node_id: `${layer}-router`, port_id: 'out' }, target: { node_id: expert.id, port_id: input } },
      { id: `${expert.id}-route-${output}`, kind: 'data', provenance: [],
        source: { node_id: expert.id, port_id: output }, target: { node_id: `${layer}-scatter`, port_id: 'in' } });
    }
  }
  const last = graph.nodes.find((node) => node.id === 'layer-0-46-expert-63')!;
  last.ports.push({ ...last.ports[0]!, id: 'extra', label: 'Extra input' });
  graph.edges.push({ ...graph.edges[0]!, id: 'unequal-expert-edge',
    source: { node_id: 'layer-0-46-router', port_id: 'out' }, target: { node_id: last.id, port_id: 'extra' } });
  const options = { expanded: [], exhaustive: true };
  const projection = projectGraph(graph, options);
  expect(projection.nodes.length).toBeGreaterThan(4_000);
  expect(graph.edges.length).toBeGreaterThan(18_000);
  expect(graph.repetitions.filter((record) => record.instances.length === 64)).toHaveLength(46);
  const layout = await layoutGraph(graph, options);
  expect(new Set(layout.boxes.map((box) => box.id))).toEqual(new Set(projection.nodes.map((node) => node.id)));
  expect(new Set(layout.routes.map((route) => route.id))).toEqual(new Set(projection.edges.map((edge) => edge.id)));
  expect(layout.edgeIds).toEqual(graph.edges.map((edge) => edge.id));
  const boxes = new Map(layout.boxes.map((box) => [box.id, box]));
  for (const box of layout.boxes) if (box.parentId) {
    const parent = boxes.get(box.parentId)!;
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.y).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(parent.width + 0.01);
    expect(box.y + box.height).toBeLessThanOrEqual(parent.height + 0.01);
  }
  const ports = new Map(layout.ports.map((port) => [endpointKey({ node_id: port.nodeId, port_id: port.portId }), port]));
  const routes = new Map(layout.routes.map((route) => [route.id, route]));
  const misses: string[] = [];
  for (const edge of projection.edges) {
    const route = routes.get(edge.id)!;
    const source = ports.get(endpointKey(edge.source))!, target = ports.get(endpointKey(edge.target))!;
    const first = route.sections[0]?.[0], lastPoint = route.sections.at(-1)?.at(-1);
    if (!first || !lastPoint || Math.abs(first.x - source.absoluteX) > 0.01 || Math.abs(first.y - source.absoluteY) > 0.01 ||
      Math.abs(lastPoint.x - target.absoluteX) > 0.01 || Math.abs(lastPoint.y - target.absoluteY) > 0.01) misses.push(edge.id);
    for (const section of route.sections) for (let i = 1; i < section.length; i++) {
      const before = section[i - 1]!, after = section[i]!;
      if (Math.abs(before.x - after.x) > 0.01 && Math.abs(before.y - after.y) > 0.01) misses.push(edge.id);
    }
  }
  expect(misses).toEqual([]);
}, 60_000);

it('retains dimension labels and source endpoints across a split expert interior', async () => {
  const graph = referenceFixture('glm4').graph;
  const options = { scope: 'layer-0-1-routed-experts', expanded: [], exhaustive: true, dimensions: true };
  const projection = projectGraph(graph, options);
  const layout = await layoutGraph(graph, options);
  const routes = new Map(layout.routes.map((route) => [route.id, route]));
  expect(routes.size).toBe(projection.edges.length);
  expect(projection.nodes.filter((node) => node.id.includes('-expert-'))).toHaveLength(64);
  for (const edge of projection.edges) {
    const route = routes.get(edge.id)!;
    expect(route.labels).toHaveLength(1);
    expect(route.labels![0]!.lines.length).toBeGreaterThan(0);
  }
}, 30_000);
