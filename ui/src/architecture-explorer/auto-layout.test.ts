import { layoutGraph } from './auto-layout';
import { describe, expect, it } from 'vitest';
import { makeProjectionFixture } from '../../tests/architecture-projection-fixture';
import { makeExplicitFixture } from '../../tests/architecture-explicit-fixture';
import { summaryFixture } from '../../tests/architecture-summary-fixture';
import { groupHeaderHeight, layerGap } from './auto-layout';
import { deriveMlpGroups } from './derived-groups';
import type { Box, Graph, Layout, Point } from './graph';
import { endpointKey, projectGraph, type ProjectionOptions } from './projection';

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

describe('generated horizontal graph geometry', () => {
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
    expect(layout.boxes).toHaveLength(graph.nodes.length); expect(new Set(layout.edgeIds)).toEqual(new Set(graph.edges.map((e) => e.id)));
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
