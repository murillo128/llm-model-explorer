import { expect, it } from 'vitest';
import { makeTemplateFixture } from '../../tests/architecture-template-fixture';
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
