import { expect, it } from 'vitest';
import { makeTemplateFixture } from '../../tests/architecture-template-fixture';
import { validateArchitecture } from '../api/architecture-validation';
import type { Graph } from './graph';
import { GraphViews } from './graph';
import { browserExpansionId, browserIndex } from './browser-model';
import { layoutGraph } from './auto-layout';
import { bindTemplateLayout, enterSharedStructure, templateGraph } from './shared-structure';

// The importer is covered by backend tests. This consumer fixture deliberately
// uses the existing API records with model-supplied (not reviewed-source) origin.
function modelDefined(): Graph {
  const graph = makeTemplateFixture();
  graph.scope = 'model_defined';
  graph.graph_id = 'model-defined-shared-fixture';
  const provenance = [{ kind: 'description' as const, source: 'model-supplied architecture.json',
    revision: 'shared-example-v1', rule: 'Author-declared structure; not verified against forward().' }];
  for (const template of graph.templates!) template.provenance = provenance;
  graph.nodes.push({ id: 'origin', kind: 'context', label: 'Model-supplied definition',
    ports: [], parameter_ids: [], references: [], attributes: [
      { name: 'definition_origin', value: 'model', provenance },
      { name: 'semantic_verification', value: 'not_verified', provenance },
    ], provenance });
  return graph;
}

it('retains Shared families and repetition navigation for model-owned provenance', () => {
  const graph = modelDefined();
  const response = validateArchitecture({ model_id: 'owned', status: 'available', diagnostics: [], graph },
    { modelId: 'owned' });
  expect(response.status).toBe('available');
  const view = new GraphViews().get('owned', graph), template = graph.templates![0]!;
  const second = template.instances[1]!;
  expect(browserIndex(graph).find((n) => n.node.id === second.node_id)!.path).toContain('Instance 2');
  enterSharedStructure(view, template, null);
  view.update({ shared: { ...view.shared!, instanceId: second.node_id } });
  expect(view.shared!.instanceId).toBe(second.node_id);
  expect(browserExpansionId(graph, view, second.node_id)).toBe(template.instances[0]!.node_id);
  expect(graph.nodes.find((n) => n.id === 'origin')!.attributes[1]!.value).toBe('not_verified');
});

it('keeps structure-only inspection unbound and uses nonzero instance weights after selection', async () => {
  const graph = modelDefined(), before = structuredClone(graph), template = graph.templates![0]!;
  const first = template.instances[0]!, second = template.instances[1]!;
  const layout = await layoutGraph(templateGraph(graph, first), {
    scope: first.node_id, expanded: first.nodes.map((n) => n.node_id), showUnused: true,
  });
  const neutral = bindTemplateLayout(layout, graph, template, first, null);
  expect(neutral.projection.nodes.every((n) => n.sourceIds.length === 0 && !n.record?.parameter_ids.length)).toBe(true);
  const selected = bindTemplateLayout(layout, graph, template, first, second);
  expect(selected.boxes).toBe(layout.boxes);
  expect(selected.ports).toBe(layout.ports);
  expect(selected.routes).toBe(layout.routes);
  expect(selected.projection.nodes.find((n) => n.id === 'layer-0.attention.Q')!.record!.parameter_ids)
    .toEqual(['parameter.model.layers.2.self_attn.q_proj.weight']);
  expect(graph).toEqual(before);
});
