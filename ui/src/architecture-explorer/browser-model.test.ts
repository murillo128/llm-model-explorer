import { expect, it } from 'vitest';
import { makeTemplateFixture } from '../../tests/architecture-template-fixture';
import { browserExpansionId, browserIndex } from './browser-model';
import { GraphViews } from './graph';
import { enterSharedStructure } from './shared-structure';
import { selectComponent } from './component-actions';

it('follows ordered containment even when transport records are shuffled, and searches source keys', () => {
  const graph = makeTemplateFixture(), expected = browserIndex(graph);
  const shuffled = { ...graph, nodes: [...graph.nodes.filter((node) => !node.parent_id), ...graph.nodes.filter((node) => node.parent_id).reverse()] };
  expect(browserIndex(shuffled).map((entry) => entry.node.id)).toEqual(expected.map((entry) => entry.node.id));
  const q = expected.find((entry) => entry.node.id === 'layer-2.attention.Q')!;
  expect(expected.find((entry) => entry.node.id === 'layer-2.silu')!.search).toContain('model.layers.2.mlp.activation');
  expect(q.path).toContain('Instance 2');
  expect(expected.filter((entry) => entry.label === q.label).map((entry) => entry.path).length).toBeGreaterThan(1);
});
it('maps a nonzero shared instance expansion through verified roles without substituting selection', () => {
  const graph = makeTemplateFixture(), view = new GraphViews().get('model', graph);
  enterSharedStructure(view, graph.templates![0]!, null);
  view.update({ shared: { ...view.shared!, instanceId: 'layer-2.attention' } });
  expect(browserExpansionId(graph, view, 'layer-2.attention')).toBe('layer-0.attention');
  const options = view.getProjectionOptions();
  selectComponent(view, 'layer-2.attention.Q');
  expect(view.selected).toBe('layer-2.attention.Q');
  expect(view.getProjectionOptions()).toBe(options);
  expect(browserExpansionId(graph, view, 'layer-1.mlp')).toBe('layer-1.mlp');
});
it('clears family selection on a source selection without invalidating layout and prunes stale browser IDs', () => {
  const graph = makeTemplateFixture(), views = new GraphViews(), view = views.get('model', graph);
  const options = view.getProjectionOptions();
  view.browser = { ...view.browser, query: 'Q', families: ['shared-full-attention', 'stale'], selectedFamily: 'stale' };
  views.get('model', graph);
  expect(view.browser.families).toEqual(['shared-full-attention']);
  expect(view.browser.selectedFamily).toBeNull();
  view.browser.selectedFamily = 'shared-full-attention'; selectComponent(view, 'layer-2.attention');
  expect(view.browser.selectedFamily).toBeNull(); expect(view.browser.query).toBe('Q');
  expect(view.getProjectionOptions()).toBe(options);
  expect(views.get('model', { ...graph, graph_id: 'replacement' }).browser.query).toBe('');
});

it('keeps neutral common selection distinct from concrete instance zero, without changing scope or layout', () => {
  const graph = makeTemplateFixture(), view = new GraphViews().get('model', graph);
  enterSharedStructure(view, graph.templates![0]!, null);
  expect(view.selectionMode).toBe('structure'); expect(view.shared!.instanceId).toBeNull();
  const options = view.getProjectionOptions();
  selectComponent(view, 'layer-0.attention');
  expect(view.selectionMode).toBe('source'); expect(view.selected).toBe('layer-0.attention');
  expect(view.shared!.instanceId).toBeNull(); expect(view.getProjectionOptions()).toBe(options);
});
