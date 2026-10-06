import { expect, it } from 'vitest';
import { createElement } from 'react';
import { render } from '@testing-library/react';
import { ConnectionInspection } from './Connection';
import { displayLabel } from './presentation';
import { assertTraceability } from '../../tests/architecture-invariants';
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

it.each(['broken', 'bypass', 'nonconsecutive', 'absent', 'singleton', 'context', 'mixed-kind'] as const)('keeps a truthful window for %s stacks', (change) => {
  const graph = makeIndexedFixture(change === 'singleton' ? 1 : 3);
  if (change === 'broken') graph.edges = graph.edges.filter((e) => e.id !== 'next-1');
  if (change === 'bypass') graph.edges.push({ ...graph.edges.find((e) => e.id === 'next-0')!, id: 'bypass', target: { node_id: 'layer-2', port_id: 'x' } });
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
  if (change !== 'nonconsecutive') expect(view.repetitions.layers).toEqual({ start: 0, count: 2 });
  expect(projectGraph(graph, projectionOptions(view)).edges.every((e) => !e.relationship)).toBe(true);
});


it.each([{ showUnused: false, bound: true }, { showUnused: true, bound: true },
  { showUnused: false, bound: false }, { showUnused: true, bound: false }])('expands an unused optional interface (%j)', ({ showUnused, bound }) => {
  const graph = makeIndexedFixture();
  // Keep the declared layer input and external signal, but no operation consumes it.
  graph.edges = graph.edges.filter((e) => !['layer-0.mask', 'layer-1.mask', 'layer-2.mask'].includes(e.id));
  if (!bound) graph.edges = graph.edges.filter((e) => !['mask-0', 'mask-1', 'mask-2'].includes(e.id));
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
  expect(projection.nodes.find((n) => n.id === outer.id)?.expanded).toBe(true);
  expect(projection.nodes.filter((n) => n.shared)).toHaveLength(1);
  if (showUnused && bound) {
    expect(projection.edges.find((e) => e.relationship?.kind === 'invariant' && e.relationship.portRole === 'root.mask')!
      .paths.map((p) => p.map((e) => e.id))).toEqual([['mask-0'], ['mask-1'], ['mask-2']]);
  } else if (!showUnused) {
    if (bound) expect(projection.filteredEdgeIds).toEqual(expect.arrayContaining(['mask-0', 'mask-1', 'mask-2']));
    expect(projection.nodes.find((n) => n.shared)?.ports.some((p) => p.id === 'mask')).toBe(false);
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


it('decomposes a mixed stack into maximal ordered ranges and retains internal state', () => {
  const graph = makeIndexedFixture(6);
  graph.repetitions[0]!.instances.forEach((i, at) => { i.variant = at === 2 ? 'full' : 'linear'; });
  // Internal state is part of the exact mapped layer body; it never becomes an invariant input.
  for (const instance of graph.templates![0]!.instances) {
    const operation = graph.nodes.find((n) => n.id === instance.nodes.find((m) => m.role === 'op')!.node_id)!;
    operation.kind = 'state';
  }
  const view = new GraphViews().get('hybrid', graph); view.update({ expanded: ['model'] });
  const outer = projectGraph(graph, projectionOptions(view)).nodes.find((n) => n.presentation === 'repetition')!;
  toggleComponent(view, graph, outer, 2);
  let projection = projectGraph(graph, projectionOptions(view));
  const ranges = projection.nodes.filter((n) => n.parentId === outer.id);
  expect(ranges.map((n) => n.instances?.map((i) => i.index) ?? n.sourceIds)).toEqual([[0, 1], ['layer-2'], [3, 4, 5]]);
  for (const range of ranges.filter((n) => n.presentation === 'repetition')) toggleComponent(view, graph, range, 2);
  projection = projectGraph(graph, projectionOptions(view));
  expect(projection.nodes.filter((n) => n.shared)).toHaveLength(2);
  const returns = projection.edges.filter((e) => e.relationship?.kind === 'return');
  expect(returns.map((e) => e.paths.flat().map((p) => p.id))).toEqual([['next-0'], ['next-3', 'next-4']]);
  expect(returns.map((e) => e.relationship?.owner === 'repetition' && e.relationship.instances.map((i) => i.index)))
    .toEqual([[0, 1], [3, 4, 5]]);
});


it.each([false, true])('proves indexed side outputs independently of activation (swapped=%s)', (swapped) => {
  const graph = makeIndexedFixture(4);
  graph.repetitions[0]!.instances[0]!.variant = 'dense';
  for (const i of graph.repetitions[0]!.instances.slice(1)) i.variant = 'moe';
  const shape = [{ kind: 'constant' as const, value: 4 }];
  const provenance = graph.nodes[0]!.provenance;
  const bankShape = [{ kind: 'constant' as const, value: 4 }, ...shape];
  graph.nodes.push({ id: 'prior-bank', parent_id: 'model', kind: 'state', operation: 'prior_state_bank',
    label: 'Prior state bank', ports: [{ id: 'out', label: 'out', direction: 'output', shape: bankShape }],
    attributes: [], parameter_ids: [], references: [], provenance });
  const bank = { ...graph.nodes.find((n) => n.id === 'result')!, id: 'next-bank',
    ports: [0, 1, 2, 3].flatMap((i) => ['key', 'value'].map((role) => ({
      id: `${role}-${i}`, label: `${role}-${i}`, direction: 'input' as const, shape }))) };
  graph.nodes.push(bank);
  const model = graph.nodes[0]!; if (model.kind === 'group') model.children.push(bank.id, 'prior-bank');
  graph.repetitions[0]!.side_ports = ['key', 'value'].map((role) => ({
    port_id: `next_${role}`, direction: 'output', bindings: [0, 1, 2, 3].map((index) => ({
      index, endpoint: { node_id: bank.id, port_id: `${role}-${index}` } })) }));
  for (const instance of graph.templates![0]!.instances) {
    const i = Number(instance.node_id.split('-')[1]);
    const root = graph.nodes.find((n) => n.id === instance.node_id)!;
    const op = graph.nodes.find((n) => n.id === `${root.id}.op`)!;
    // Exact forwarding gives the formerly interleaved selector its truthful
    // layer ownership. Its source array position is deliberately before the root.
    root.attributes.push({ name: 'layer_index', value: i, provenance });
    root.ports.push({ id: 'bank', label: 'bank', direction: 'input', shape: bankShape });
    op.ports.push({ id: 'prior', label: 'prior', direction: 'input', shape });
    const selector = { id: `select-${i}`, parent_id: root.id, kind: 'operation' as const,
      operation: 'select_layer_state', label: 'Select layer state', formula: 'out = bank[owner.layer_index]',
      ports: [{ id: 'bank', label: 'bank', direction: 'input' as const, shape: bankShape },
        { id: 'out', label: 'out', direction: 'output' as const, shape }],
      attributes: [], parameter_ids: [], references: [], provenance };
    graph.nodes.splice(graph.nodes.indexOf(root), 0, selector);
    if (root.kind === 'group') root.children.unshift(selector.id);
    instance.nodes.push({ role: 'selector', node_id: selector.id });
    instance.ports.push({ role: 'root.bank', node_id: root.id, port_id: 'bank' },
      { role: 'op.prior', node_id: op.id, port_id: 'prior' },
      ...selector.ports.map((p) => ({ role: `selector.${p.id}`, node_id: selector.id, port_id: p.id })));
    graph.edges.push({ id: `bank-${i}`, source: { node_id: 'prior-bank', port_id: 'out' },
      target: { node_id: root.id, port_id: 'bank' }, kind: 'state', provenance });
    for (const [role, source, sourcePort, target, targetPort] of [
      ['bank-forward', root.id, 'bank', selector.id, 'bank'], ['selected-state', selector.id, 'out', op.id, 'prior'],
    ]) {
      graph.edges.push({ id: `${role}-${i}`, source: { node_id: source!, port_id: sourcePort! },
        target: { node_id: target!, port_id: targetPort! }, kind: 'state', provenance });
      instance.edges.push({ role: role!, edge_id: `${role}-${i}` });
    }
    op.kind = 'state';
    for (const role of ['key', 'value']) {
      const port = `next_${role}`;
      for (const [node, nodeRole] of [[root, 'root'], [op, 'op']] as const) {
        node.ports.push({ id: port, label: port, direction: 'output', shape });
        instance.ports.push({ role: `${nodeRole}.${port}`, node_id: node.id, port_id: port });
      }
      const id = `${root.id}.${port}`;
      graph.edges.push({ id, source: { node_id: op.id, port_id: port }, target: { node_id: root.id, port_id: port }, kind: 'state', provenance });
      instance.edges.push({ role: port, edge_id: id });
      graph.edges.push({ id: `${role}-bank-${i}`, source: { node_id: root.id, port_id: port },
        target: { node_id: bank.id, port_id: `${swapped && i === 2 ? role === 'key' ? 'value' : 'key' : role}-${i}` }, kind: 'state', provenance });
    }
  }
  const projection = projectGraph(graph, { expanded: ['model', 'repeat:layers:0:3', 'repeat:layers:1:3'] });
  const returns = projection.edges.filter((e) => e.relationship?.kind === 'return');
  if (swapped) { expect(returns).toHaveLength(0); return; }
  expect(returns.map((e) => e.originalEdgeIds)).toEqual([['next-1', 'next-2']]);
  expect(projection.edges.filter((e) => e.relationship?.kind === 'side-output').map((e) => e.originalEdgeIds))
    .toEqual([['key-bank-1', 'key-bank-2', 'key-bank-3'], ['value-bank-1', 'value-bank-2', 'value-bank-3']]);
  expect(projection.nodes.find((n) => n.shared)?.record?.formula)
    .toBe('out[i], next_key[i], next_value[i] = layer[i](x[i], cos, sin, mask, bank)');
  expect(projection.nodes.some((n) => n.sourceIds.some((id) => id.startsWith('select-')))).toBe(false);
  expect(projection.nodes.find((n) => n.shared)?.record?.attributes.find((a) => a.name === 'layer_index')?.value).toBe('i');
});

it('projects one verified 3+1 body with two scoped depth returns and exact transition coverage', async () => {
  const { makeNestedIndexedFixture } = await import('../../tests/architecture-template-fixture');
  const graph = makeNestedIndexedFixture();
  // A genuine external group forwarding prefix must stay in each ordered path.
  const cos = graph.nodes.find((n) => n.id === 'cos')!;
  cos.parent_id = 'cos-relay';
  const model = graph.nodes[0]!;
  if (model.kind === 'group') model.children[model.children.indexOf('cos')] = 'cos-relay';
  graph.nodes.push({ ...cos, id: 'cos-relay', parent_id: 'model', kind: 'group', children: ['cos'] });
  for (const edge of graph.edges.filter((e) => e.source.node_id === 'cos')) edge.source.node_id = 'cos-relay';
  graph.edges.push({ id: 'cos-start', source: { node_id: 'cos', port_id: 'out' },
    target: { node_id: 'cos-relay', port_id: 'out' }, kind: 'data', provenance: [] });
  const before = structuredClone(graph);
  const view = new GraphViews().get('nested', graph);
  view.update({ expanded: ['model', 'repeat:layers:0:23'] });
  let projection = projectGraph(graph, projectionOptions(view));
  expect(projection.nodes.filter((n) => n.label === 'Hybrid block[j]')).toHaveLength(1);
  expect(projection.nodes.filter((n) => n.label === 'Linear decoder ×3')).toHaveLength(1);
  expect(projection.nodes.find((n) => n.label === 'Full attention layer[4j+3]')?.shared?.templateId).toBe('full');
  const inner = projection.nodes.find((n) => n.label === 'Linear decoder ×3')!;
  expect(displayLabel(inner, graph)).toBe('Linear decoder ×3');
  assertTraceability(graph, projectGraph(graph, { expanded: [], exhaustive: true }), true);
  toggleComponent(view, graph, inner, 1);
  projection = projectGraph(graph, projectionOptions(view));
  expect(projection.nodes.find((n) => n.label === 'Linear layer[4j+k]')).toBeDefined();
  const firstIndices = [0, 4, 8, 12, 16, 20], finalIndices = [2, 6, 10, 14, 18, 22];
  const linearIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22];
  const fullIndices = [3, 7, 11, 15, 19, 23];
  for (const [target, indices] of [[inner.id, linearIndices], ['indexed:body:layers:0:slot:3:layer-3', fullIndices]] as const) {
    for (const port of ['cos', 'sin', 'mask']) {
      const branch = projection.edges.find((e) => e.source.node_id === 'body:layers:0' &&
        e.target.node_id === target && e.target.port_id === port)!;
      expect.soft(branch.paths.map((path) => path.map((e) => e.id))).toEqual(indices.map((i) =>
        port === 'cos' ? ['cos-start', `cos-${i}`] : [`${port}-${i}`]));
      expect.soft(branch.relationship).toMatchObject({ instances: indices.map((index) => ({ nodeId: `layer-${index}`, index })) });
    }
  }
  for (const [port, label, indices] of [['x', 'x[4j]', firstIndices], ['out', 'out[4j+2]', finalIndices]] as const) {
    const boundary = projection.nodes.find((n) => n.id === inner.id)!.ports.find((p) => p.id === port)!;
    expect.soft(boundary.label).toBe(label);
    expect.soft(boundary.endpoints).toEqual(indices.map((i) => ({ node_id: `layer-${i}`, port_id: port })));
  }
  for (const [kind, offset, phase, indices, text] of [
    ['entry', 0, 'initial', firstIndices, 'Initial entry: x[4j], at the first layer of this range in each block (0 ≤ j < 6).'],
    ['exit', 2, 'final', finalIndices, 'Final exit: out[4j+2], only after the last layer of this range in each block (0 ≤ j < 6).'],
  ] as const) {
    const edge = projection.edges.find((e) => e.relationship?.owner === 'repetition' && e.relationship.scopeId === inner.id && e.relationship.kind === kind)!;
    expect.soft(edge.relationship).toMatchObject({ indexPhase: { phase, base: 0, width: 4, offset, count: 6 },
      instances: indices.map((index) => ({ nodeId: `layer-${index}`, index })) });
    const inspection = render(createElement(ConnectionInspection, { graph, edge, trigger: document.createElement('button'),
      onClose: () => {}, inspectNode: () => {} }));
    expect.soft(inspection.getByRole('dialog')).toHaveTextContent(text);
    inspection.unmount();
  }
  const returns = projection.edges.filter((e) => e.relationship?.kind === 'return');
  expect(returns).toHaveLength(2);
  expect(returns.map((e) => e.paths.flat().map((p) => p.id))).toEqual([
    ['next-3', 'next-7', 'next-11', 'next-15', 'next-19'],
    ['next-0', 'next-1', 'next-4', 'next-5', 'next-8', 'next-9', 'next-12', 'next-13', 'next-16', 'next-17', 'next-20', 'next-21'],
  ]);
  expect(projection.edges.find((e) => e.id === 'body:layers:0:connection:1')!.originalEdgeIds)
    .toEqual(['next-2', 'next-6', 'next-10', 'next-14', 'next-18', 'next-22']);
  const transitionIds = projection.edges.flatMap((e) => e.originalEdgeIds).filter((id) => id.startsWith('next-'));
  expect(transitionIds).toHaveLength(23);
  expect(new Set(transitionIds).size).toBe(23);
  expect(projection.nodes.find((n) => n.id === 'repeat:layers:0:23')!.ports.some((p) => p.label === 'out[23]')).toBe(true);
  const layer = projection.nodes.find((n) => n.label === 'Linear layer[4j+k]')!;
  toggleComponent(view, graph, layer, 1);
  const detailed = projectGraph(graph, projectionOptions(view));
  const op = detailed.nodes.find((n) => n.id === 'indexed:body:layers:0:slot:0:layer-0.op')!;
  expect(op.record!.formula).toBe('out[4j+k] = transform(x[4j+k], cos, sin, mask; weight[4j+k])');
  expect(op.record!.parameter_ids).toEqual([]);
  expect(op.symbolicParameters![0]!.label).toBe('weight[4j+k]');
  // Existing Shared concrete selection resolves the complete global index, not
  // an inner offset which could silently bind the first block.
  for (const [familyId, index] of [['linear', 9], ['full', 11]] as const) {
    const family = graph.templates!.find((t) => t.id === familyId)!;
    const anchor = family.instances[0]!, chosen = family.instances.find((i) => i.node_id === `layer-${index}`)!;
    const local = await layoutGraph(templateGraph(graph, anchor), { expanded: [anchor.node_id], scope: anchor.node_id });
    const concrete = bindTemplateLayout(local, graph, family, anchor, chosen);
    expect(concrete.projection.nodes.find((n) => n.record?.id === `layer-${index}.op`)!.record!.parameter_ids).toEqual([`weight-${index}`]);
    expect(bindTemplateLayout(local, graph, family, anchor, null).projection.nodes.every((n) => !n.record!.parameter_ids.length)).toBe(true);
  }
  const outer = detailed.nodes.find((n) => n.id === 'repeat:layers:0:23')!;
  toggleComponent(view, graph, outer, 1);
  expect(projectGraph(graph, projectionOptions(view)).nodes.some((n) => n.id === inner.id)).toBe(false);
  toggleComponent(view, graph, outer, 1);
  expect(projectGraph(graph, projectionOptions(view)).nodes.find((n) => n.id === inner.id)!.expanded).toBe(true);
  const layout = await layoutGraph(graph, projectionOptions(view));
  expect(layout.projection.edges.filter((e) => e.relationship?.kind === 'return')).toHaveLength(2);
  expect(graph).toEqual(before);
});


it('preserves nested component repetitions in a bounded Shared scope and rebinds their source endpoints', async () => {
  const graph = makeTemplateFixture(), family = graph.templates![0]!;
  for (const index of [0, 2]) graph.repetitions.push({ id: `projections-${index}`, parent_id: `layer-${index}.attention`,
    label: 'Projections', instances: ['Q', 'K', 'V'].map((role, indexInLayer) => ({ index: indexInLayer,
      node_id: `layer-${index}.attention.${role}`, variant: 'projection' })) });
  const [anchor, chosen] = family.instances;
  const scoped = templateGraph(graph, anchor!);
  expect(scoped.repetitions.map((r) => r.id)).toEqual(['projections-0']);
  const layout = await layoutGraph(scoped, { scope: anchor!.node_id, expanded: [anchor!.node_id] });
  const neutral = bindTemplateLayout(layout, graph, family, anchor!, null);
  expect(neutral.projection.nodes.find((n) => n.repetitionId)!.sourceIds).toEqual([]);
  const bound = bindTemplateLayout(layout, graph, family, anchor!, chosen!);
  const range = bound.projection.nodes.find((n) => n.repetitionId)!;
  expect(range.sourceIds).toEqual(['layer-2.attention.Q', 'layer-2.attention.K', 'layer-2.attention.V']);
  expect(range.ports.flatMap((p) => p.endpoints).every((e) => e.node_id.startsWith('layer-2.'))).toBe(true);
  const view = new GraphViews().get('scoped', graph);
  enterSharedStructure(view, family, null);
  view.update({ shared: { templateId: family.id, anchorId: anchor!.node_id, instanceId: chosen!.node_id } });
  toggleComponent(view, graph, range, 2);
  expect(view.repetitions['projections-0']).toEqual({ start: 0, count: 2 });
});
