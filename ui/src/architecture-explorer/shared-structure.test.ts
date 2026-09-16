import { expect, it } from 'vitest';
import { makeTemplateFixture } from '../../tests/architecture-template-fixture';
import { contractInventory } from '../../tests/architecture-fixtures';
import { validateArchitecture } from '../api/architecture-validation';
import { layoutGraph } from './auto-layout';
import { GraphViews } from './graph';
import { connectionSet, projectGraph } from './projection';
import { bindTemplateLayout, enterSharedStructure, remapNode, templateGraph } from './shared-structure';
import { backFromComponent, enterComponent, projectionOptions, returnToModel, snapshotView } from './scope-navigation';

it('uses an independently authored complete correspondence; ordinary projections ignore optional annotations', () => {
  const graph = makeTemplateFixture();
  validateArchitecture({ model_id: 'fixture', status: 'available', diagnostics: [], graph }, { modelId: 'fixture', inventory: contractInventory, tokenizerAvailable: true });
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
    expect(node.ports.every((p) => p.endpoints.every((e) => e.node_id === target.node_id))).toBe(true);
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
