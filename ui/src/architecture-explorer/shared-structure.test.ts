import { expect, it } from 'vitest';
import { toggleComponent } from "./component-actions";
import { makeIndexedFixture, makeTemplateFixture } from '../../tests/architecture-template-fixture';
import { validateArchitecture } from '../api/architecture-validation';
import { layoutGraph } from './auto-layout';
import { GraphViews } from './graph';
import { connectionSet, projectGraph } from './projection';
import { bindTemplateLayout, bindTemplatePortSelection, enterSharedStructure, remapNode, templateGraph } from './shared-structure';
import { backFromComponent, enterComponent, projectionOptions, returnToModel, snapshotView } from './scope-navigation';

it('uses an independently authored complete correspondence; ordinary projections ignore optional annotations', () => {
  const graph = makeTemplateFixture();
  validateArchitecture({ model_id: 'fixture', status: 'available', diagnostics: [], graph }, { modelId: 'fixture' });
  const ordinary = structuredClone(graph); delete ordinary.templates;
  for (const options of [{ expanded: ['model'] }, { expanded: [], exhaustive: true }, { expanded: ['layer-2.attention'], scope: 'layer-2.attention' }]) {
    expect(projectGraph(graph, options)).toEqual(projectGraph(ordinary, options));
  }
});

it('reuses all geometry and hit identities while rebinding exact node, port, edge and parameter records', async () => {
  const graph = makeTemplateFixture(), before = structuredClone(graph), template = graph.templates![0]!;
  const [first, second] = template.instances;
  const local = templateGraph(graph, first!);
  const layout = await layoutGraph(local, { expanded: first!.nodes.map((n) => n.node_id), showUnused: true, deriveMlp: false, scope: first!.node_id });
  expect(layout.projection.nodes.map((n) => n.id).sort()).toEqual(first!.nodes.map((n) => n.node_id).sort());
  const neutral = bindTemplateLayout(layout, graph, template, first!, null);
  const chosen = bindTemplateLayout(layout, graph, template, first!, second!);
  for (const view of [neutral, chosen]) {
    expect(view.boxes).toBe(layout.boxes); expect(view.ports).toBe(layout.ports); expect(view.routes).toBe(layout.routes);
    expect(view.projection.edges.map((e) => [e.id, e.source, e.target])).toEqual(layout.projection.edges.map((e) => [e.id, e.source, e.target]));
    expect(connectionSet(view.projection, { port: { node_id: first!.node_id, port_id: 'x' } })).toEqual(connectionSet(layout.projection, { port: { node_id: first!.node_id, port_id: 'x' } }));
  }
  expect(neutral.edgeIds).toEqual([]);
  expect(neutral.projection.nodes.every((n) => n.sourceIds.length === 0 && n.record?.parameter_ids.length === 0 && n.record.references.length === 0)).toBe(true);
  for (const mapping of first!.nodes) {
    const target = second!.nodes.find((m) => m.role === mapping.role)!;
    const node = chosen.projection.nodes.find((n) => n.id === mapping.node_id)!;
    expect(node.record).toBe(graph.nodes.find((n) => n.id === target.node_id));
    expect(node.sourceIds).toEqual([target.node_id]);
    for (const port of node.ports) expect(port.endpoints).toContainEqual({ node_id: target.node_id, port_id: port.id });
  }
  expect(new Set(chosen.edgeIds)).toEqual(new Set(second!.edges.map((e) => e.edge_id)));
  expect(chosen.projection.edges.flatMap((e) => e.paths.flat()).every((e) => graph.edges.includes(e))).toBe(true);
  expect(remapNode('layer-0.attention.Q', first!, second!)).toBe('layer-2.attention.Q');
  expect(remapNode('layer-1.attention.input', first!, second!)).toBeNull();
  expect(chosen.projection.nodes.find((n) => n.id === 'layer-0.attention.Q')!.record!.parameter_ids).toEqual(['parameter.model.layers.2.self_attn.q_proj.weight']);
  expect(graph).toEqual(before);
  expect(local.nodes.find((n) => n.id === first!.node_id)!.parent_id).toBeUndefined();
  expect(graph.nodes.find((n) => n.id === first!.node_id)!.parent_id).toBe('layer-0');
});

it('restores ordinary and nested context and clears unavailable correspondence on graph replacement', () => {
  const graph = makeTemplateFixture(), template = graph.templates![0]!, views = new GraphViews();
  const view = views.get('model', graph);
  view.update({ viewport: { x: 10, y: 20, zoom: 1.1 }, selected: 'layer-2.attention.Q' });
  const global = snapshotView(view);
  enterComponent(view, graph, 'layer-2'); const isolated = snapshotView(view);
  enterSharedStructure(view, template, 'layer-2.attention');
  expect(view.shared?.instanceId).toBe('layer-2.attention');
  backFromComponent(view); expect(snapshotView(view)).toEqual(isolated);
  backFromComponent(view); expect(snapshotView(view)).toEqual(global);
  enterSharedStructure(view, template, null);
  expect(view.shared?.instanceId).toBeNull();
  expect(projectionOptions(view).scope).toBe('layer-0.attention');
  returnToModel(view); expect(snapshotView(view)).toEqual(global);
  for (let i = 0; i < 30; i++) enterSharedStructure(view, template, null);
  expect(view.history).toHaveLength(16); expect(JSON.stringify(view.history)).not.toContain('parameter_ids');
  const replaced = views.get('model', { ...graph, graph_id: 'replacement' });
  expect(replaced.shared).toBeUndefined(); expect(replaced.selected).toBeNull(); expect(replaced.notice).toContain('cleared');
  enterSharedStructure(replaced, template, null);
  const removed = views.get('model', { ...graph, graph_id: 'replacement', templates: [] });
  expect(removed.shared).toBeUndefined(); expect(removed.notice).toContain('correspondence changed');
});

it('resolves external declaration metadata for a nonzero shared instance without changing geometry', async () => {
  const graph = makeTemplateFixture(), before = structuredClone(graph), template = graph.templates![0]!;
  const [first, second] = template.instances;
  const layout = await layoutGraph(templateGraph(graph, first!), { scope: first!.node_id,
    expanded: first!.nodes.map((n) => n.node_id), showUnused: true });
  const chosen = bindTemplateLayout(layout, graph, template, first!, second!);
  const port = chosen.projection.nodes.find((n) => n.id === first!.node_id)!.ports.find((p) => p.id === 'positions')!;
  const isolated = projectGraph(graph, { scope: second!.node_id, expanded: [second!.node_id], showUnused: true });
  const expected = isolated.nodes.find((n) => n.id === second!.node_id)!.ports.find((p) => p.id === 'positions')!;
  expect(port.endpoints).toEqual(expected.endpoints);
  expect(port.interfaces).toEqual(['positions']);
  expect(port.endpoints).toContainEqual({ node_id: 'positions', port_id: 'out' });
  expect(port.endpoints).toContainEqual({ node_id: second!.node_id, port_id: 'positions' });
  expect(chosen.boxes).toBe(layout.boxes); expect(chosen.ports).toBe(layout.ports); expect(chosen.routes).toBe(layout.routes);
  const neutral = bindTemplateLayout(layout, graph, template, first!, null);
  expect(neutral.projection.nodes.flatMap((n) => n.ports).every((p) => !p.endpoints.length && !p.interfaces?.length)).toBe(true);
  expect(graph).toEqual(before);
});

it('retains a source-free template port identity through snapshots, blur targets and exact instance binding', async () => {
  const graph = makeTemplateFixture(), template = graph.templates![0]!, [first, second] = template.instances;
  const layout = await layoutGraph(templateGraph(graph, first!), { scope: first!.node_id,
    expanded: first!.nodes.map((n) => n.node_id), showUnused: true });
  const neutral = bindTemplateLayout(layout, graph, template, first!, null);
  const port = neutral.projection.nodes.find((n) => n.id === first!.node_id)!.ports.find((p) => p.id === 'positions')!;
  expect(port.endpoints).toEqual([]);
  expect(port.templatePort).toEqual({ kind: 'template-port', templateId: template.id, nodeRole: 'root', portRole: 'root.positions' });
  const selection = bindTemplatePortSelection(graph, template, first!, null, port.templatePort!)!;
  expect(selection.owner.kind).toBe('presentation'); expect(selection.endpoints).toEqual([]);
  const view = new GraphViews().get('fixture', graph);
  enterSharedStructure(view, template, null); view.update({ selected: null, boundary: selection });
  const snapshot = snapshotView(view);
  view.update({ boundary: undefined }); view.update(snapshot);
  expect(view.boundary).toEqual(selection); expect(view.selected).toBeNull(); expect(view.shared?.instanceId).toBeNull();
  const selected = neutral.projection.nodes.flatMap((n) => n.ports.filter((p) =>
    p.templatePort?.portRole === view.boundary!.templatePort!.portRole).map((p) => ({ node_id: n.id, port_id: p.id })));
  expect(selected).toEqual([{ node_id: first!.node_id, port_id: 'positions' }]);
  expect(connectionSet(neutral.projection, { port: selected[0]! })).toHaveLength(2);
  for (const instance of [second!, first!]) {
    const concrete = bindTemplatePortSelection(graph, template, first!, instance, selection.templatePort!)!;
    expect(concrete.owner).toEqual({ kind: 'source', id: instance.node_id });
    expect(concrete.endpoints).toEqual([{ node_id: instance.node_id, port_id: 'positions' }, { node_id: 'positions', port_id: 'out' }]);
    expect(concrete.templatePort).toEqual(selection.templatePort);
    const rebound = bindTemplateLayout(layout, graph, template, first!, instance);
    expect(rebound.boxes).toBe(layout.boxes); expect(rebound.ports).toBe(layout.ports); expect(rebound.routes).toBe(layout.routes);
  }
});

it('clears stale shared port roles on restoration instead of selecting another port or instance', () => {
  const graph = makeTemplateFixture(), template = graph.templates![0]!, views = new GraphViews();
  const view = views.get('fixture', graph);
  enterSharedStructure(view, template, null);
  view.update({ selected: null, boundary: bindTemplatePortSelection(graph, template, template.instances[0]!, null,
    { kind: 'template-port', templateId: template.id, nodeRole: 'root', portRole: 'root.positions' }) });
  expect(views.get('fixture', graph).boundary?.templatePort?.portRole).toBe('root.positions');
  const changed = structuredClone(graph);
  changed.templates![0]!.instances[1]!.ports = changed.templates![0]!.instances[1]!.ports.filter((p) => p.role !== 'root.positions');
  const restored = views.get('fixture', changed);
  expect(restored.boundary).toBeUndefined(); expect(restored.selected).toBeNull(); expect(restored.shared?.instanceId).toBeNull();
  expect(restored.notice).toContain('port selection was cleared');
});

it('expands one neutral indexed layer with exact first/next/last provenance and indexed local roles', async () => {
  const graph = makeIndexedFixture(), before = structuredClone(graph);
  const view = new GraphViews().get('indexed', graph);
  view.update({ expanded: ['model'] });
  const outer = projectGraph(graph, projectionOptions(view)).nodes.find((n) => n.presentation === 'repetition')!;
  toggleComponent(view, graph, outer, 2);
  const projection = projectGraph(graph, projectionOptions(view));
  const inner = projection.nodes.find((n) => n.shared)!;
  expect(inner.label).toBe('Decoder Layer[i]'); expect(inner.sourceIds).toEqual([]);
  expect(projection.nodes.find((n) => n.id === outer.id)?.expanded).toBe(true);
  expect(projection.nodes.filter((n) => n.shared)).toHaveLength(1);
  const relation = (kind: string) => projection.edges.filter((e) => e.relationship?.kind === kind);
  expect(relation('entry')[0]!.paths.map((p) => p.map((e) => e.id))).toEqual([['entry']]);
  expect(relation('return')[0]!.paths.map((p) => p.map((e) => e.id))).toEqual([['next-0'], ['next-1']]);
  expect(relation('exit')[0]!.paths.map((p) => p.map((e) => e.id))).toEqual([['exit']]);
  expect(relation('invariant').map((e) => e.paths.map((p) => p.map((e) => e.id)))).toEqual([
    [['cos-0'], ['cos-1'], ['cos-2']], [['sin-0'], ['sin-1'], ['sin-2']], [['mask-0'], ['mask-1'], ['mask-2']],
  ]);
  expect(inner.record?.formula).toBe('out[i] = layer[i](x[i], cos, sin, mask)');
  expect(inner.ports.map((p) => p.label)).toEqual(['x[i]', 'cos', 'sin', 'mask', 'out[i]']);
  toggleComponent(view, graph, inner, 2);
  const expanded = projectGraph(graph, projectionOptions(view));
  const operation = expanded.nodes.find((n) => n.shared?.nodeRole === 'op')!;
  expect(operation.record?.formula).toBe('out[i] = transform(x[i], cos, sin, mask; weight[i])');
  expect(operation.symbolicParameters).toEqual([{ role: 'weight', label: 'weight[i]', shape: [{ kind: 'constant', value: 4 }] }]);
  expect(operation.record?.parameter_ids).toEqual([]);
  expect(graph).toEqual(before);
  toggleComponent(view, graph, projection.nodes.find((n) => n.id === outer.id)!, 2);
  expect(view.expanded).toContain(inner.id);
  toggleComponent(view, graph, outer, 2);
  expect(projectGraph(graph, projectionOptions(view)).nodes.find((n) => n.id === inner.id)?.expanded).toBe(true);
});

it.each(['broken', 'bypass', 'variant', 'nonconsecutive', 'absent', 'singleton', 'context', 'mixed-kind'] as const)('keeps a truthful window for %s stacks', (change) => {
  const graph = makeIndexedFixture(change === 'singleton' ? 1 : 3);
  if (change === 'broken') graph.edges = graph.edges.filter((e) => e.id !== 'next-1');
  if (change === 'bypass') graph.edges.push({ ...graph.edges.find((e) => e.id === 'next-0')!, id: 'bypass', target: { node_id: 'layer-2', port_id: 'x' } });
  if (change === 'variant') graph.repetitions[0]!.instances[1]!.variant = 'different';
  if (change === 'nonconsecutive') graph.templates![0]!.instances.splice(1, 1);
  if (change === 'context') {
    graph.nodes.push({ ...graph.nodes.find((n) => n.id === 'mask')!, id: 'other-mask', kind: 'context' });
    const root = graph.nodes[0]!; if (root.kind === 'group') root.children.push('other-mask');
    const edge = graph.edges.find((e) => e.id === 'mask-1')!;
    edge.source = { node_id: 'other-mask', port_id: 'out' }; edge.kind = 'context';
  }
  if (change === 'mixed-kind') graph.edges.find((e) => e.id === 'mask-1')!.kind = 'context';
  if (change === 'absent') delete graph.templates;
  const view = new GraphViews().get('fixture', graph); view.update({ expanded: ['model'] });
  const outer = projectGraph(graph, projectionOptions(view)).nodes.find((n) => n.presentation === 'repetition')!;
  toggleComponent(view, graph, outer, 2);
  expect(view.repetitions.layers).toEqual({ start: 0, count: 2 });
  expect(projectGraph(graph, projectionOptions(view)).edges.every((e) => !e.relationship)).toBe(true);
});


it.each([false, true])('expands a valid unused invariant interface with showUnused=%s', (showUnused) => {
  const graph = makeIndexedFixture();
  // Keep the declared layer input and external signal, but no operation consumes it.
  graph.edges = graph.edges.filter((e) => !['layer-0.mask', 'layer-1.mask', 'layer-2.mask'].includes(e.id));
  for (const node of graph.nodes) if (node.kind === 'operation') {
    node.ports = node.ports.filter((p) => p.id !== 'mask');
    node.formula = 'out = transform(x, cos, sin; weight)';
  }
  for (const instance of graph.templates![0]!.instances) {
    instance.ports = instance.ports.filter((p) => p.role !== 'op.mask');
    instance.edges = instance.edges.filter((e) => e.role !== 'mask');
  }
  validateArchitecture({ model_id: 'fixture', status: 'available', diagnostics: [], graph }, { modelId: 'fixture' });
  const view = new GraphViews().get('fixture', graph);
  view.update({ expanded: ['model'], showUnused });
  const outer = projectGraph(graph, projectionOptions(view)).nodes.find((n) => n.presentation === 'repetition')!;
  toggleComponent(view, graph, outer, 2);
  const projection = projectGraph(graph, projectionOptions(view));
  if (showUnused) {
    expect(projection.nodes.find((n) => n.id === outer.id)?.expanded).toBe(true);
    expect(projection.nodes.filter((n) => n.shared)).toHaveLength(1);
    expect(projection.edges.find((e) => e.relationship?.kind === 'invariant' && e.relationship.portRole === 'root.mask')!
      .paths.map((p) => p.map((e) => e.id))).toEqual([['mask-0'], ['mask-1'], ['mask-2']]);
  } else {
    expect(view.repetitions.layers).toEqual({ start: 0, count: 2 });
    expect(projection.nodes.map((n) => n.id)).toEqual(expect.arrayContaining(['layer-0', 'layer-1']));
    expect(projection.nodes.some((n) => n.shared)).toBe(false);
    expect(projection.filteredEdgeIds).toEqual(expect.arrayContaining(['mask-0', 'mask-1', 'mask-2']));
  }
});

it('keeps mathematical indices and genuine aliased weights unindexed in neutral operations', () => {
  const graph = makeIndexedFixture();
  for (const n of graph.nodes) if (n.kind === 'operation') n.formula = 'out = x[j] + x + weight';
  for (const index of [1, 2]) graph.parameters[index] = { ...graph.parameters[index]!, binding: 'alias',
    alias_of: 'weight-0', storage: [], inspection: { status: 'available', tensor_id: 'tensor-0' } };
  const projection = projectGraph(graph, { expanded: ['model', 'repeat:layers:0:2', 'indexed:layers:layer-0'] });
  const operation = projection.nodes.find((n) => n.shared?.nodeRole === 'op')!;
  expect(operation.record?.formula).toBe('out[i] = x[j] + x[i] + weight');
  expect(operation.symbolicParameters?.[0]?.label).toBe('weight');
});


it.each(['repetition', 'generation'] as const)('routes the %s return around its contents with attached endpoints', async (owner) => {
  const { generationFixture } = await import('../../tests/architecture-generation-fixture');
  const graph = owner === 'generation' ? generationFixture() : makeIndexedFixture();
  const expanded = owner === 'generation' ? ['generation', 'model'] : ['model', 'repeat:layers:0:2', 'indexed:layers:layer-0'];
  const layout = await layoutGraph(graph, { expanded });
  const loop = layout.projection.edges.find((e) => e.relationship?.owner === owner && e.relationship.kind === 'return')!;
  const route = layout.routes.find((r) => r.id === loop.id)!;
  const inner = layout.boxes.find((b) => b.id === (owner === 'generation' ? 'model' : loop.source.node_id))!;
  const outer = layout.boxes.find((b) => b.id === inner.parentId)!;
  const points = route.sections.flat();
  expect(points.some((p) => owner === 'generation' ? p.y > inner.absoluteY + inner.height : p.y < inner.absoluteY)).toBe(true);
  for (const p of points) {
    expect(p.x <= inner.absoluteX || p.x >= inner.absoluteX + inner.width || p.y < inner.absoluteY || p.y > inner.absoluteY + inner.height).toBe(true);
    expect(p.x).toBeGreaterThanOrEqual(outer.absoluteX);
    expect(p.x).toBeLessThanOrEqual(outer.absoluteX + outer.width);
    expect(p.y).toBeGreaterThanOrEqual(outer.absoluteY);
    expect(p.y).toBeLessThanOrEqual(outer.absoluteY + outer.height);
  }
  for (const [endpoint, point] of [[loop.source, points[0]!], [loop.target, points.at(-1)!]] as const) {
    const port = layout.ports.find((p) => p.nodeId === endpoint.node_id && p.portId === endpoint.port_id)!;
    expect(point).toEqual({ x: port.absoluteX, y: port.absoluteY });
  }
});
