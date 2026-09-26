import { expect, it } from 'vitest';
import { makeProjectionFixture } from '../../tests/architecture-projection-fixture';
import { makeExplicitFixture } from '../../tests/architecture-explicit-fixture';
import { GraphViews } from './graph';
import { backFromComponent, enterComponent, expandComponent, projectionOptions, returnToModel, snapshotView } from './scope-navigation';
import { makeTemplateFixture } from '../../tests/architecture-template-fixture';
import { enterSharedStructure } from './shared-structure';
import { projectGraph } from './projection';

it('round-trips nested component views with exact camera, selection, filters and independent stack windows', () => {
  const graph = makeExplicitFixture({ count: 3, secondStack: 2 }), views = new GraphViews();
  const view = views.get('visual', graph);
  view.update({ expanded: ['model', 'encoder', 'predictor', 'encoder.layer-1'], selected: 'encoder.layer-1',
    repetitions: { 'encoder-layers': { start: 1, count: 2 }, 'predictor-layers': { start: 0, count: 1 } },
    focus: 'encoder.layer-1', activeStack: 'encoder-layers', edge: 'some-valid-pinned-route',
    dimensions: true, showUnused: true, showContext: false, viewport: { x: -500, y: 34, zoom: 0.9 } });
  const global = snapshotView(view);
  const globalProjection = projectGraph(graph, projectionOptions(view));
  enterComponent(view, graph, 'encoder.layer-1');
  view.viewport = { x: 51, y: 82, zoom: 1.2 }; view.selected = 'encoder.layer-1.attention';
  const layer = snapshotView(view);
  enterComponent(view, graph, 'encoder.layer-1.attention');
  expect(view.history).toHaveLength(2);
  expect(view.expanded).toEqual(['encoder.layer-1.attention']);
  expect(view.viewport).toBeUndefined();
  backFromComponent(view); expect(snapshotView(view)).toEqual(layer);
  backFromComponent(view); expect(snapshotView(view)).toEqual(global);
  expect(projectGraph(graph, projectionOptions(view))).toEqual(globalProjection);
  expect(view.history).toEqual([]); expect(view.globalView).toBeUndefined();
});

it('bounds copied local state, retains the global fallback, and invalidates replaced graphs/models', () => {
  const graph = makeProjectionFixture({ count: 4 }), views = new GraphViews();
  const view = views.get('model', graph);
  const global = snapshotView(view);
  for (let i = 0; i < 40; i++) enterComponent(view, graph, `layer-${i % 4}.attention`);
  expect(view.history).toHaveLength(16);
  expect(view.globalView).toEqual(global);
  expect(JSON.stringify(view.history)).not.toContain('parameter_ids');
  returnToModel(view); expect(snapshotView(view)).toEqual(global);
  enterComponent(view, graph, 'layer-3');
  const other = views.get('other-model', graph); expect(other.scope).toBeUndefined();
  const replacement = views.get('model', { ...graph, graph_id: 'new-graph' });
  expect(replacement.scope).toBeUndefined(); expect(replacement.history).toEqual([]);
  expect(views.get('model', graph)).not.toBe(view);
  for (let i = 0; i < 9; i++) views.get(`other-${i}`, graph);
  expect(views.get('other-model', graph)).not.toBe(other);
});

it('expands only a source or derived component, including its concrete repetitions', () => {
  const graph = makeProjectionFixture({ count: 3, secondStack: 2 }), views = new GraphViews();
  const view = views.get('model', graph);
  enterComponent(view, graph, 'encoder');
  const options = expandComponent(view, graph), projection = projectGraph(graph, options);
  expect(options.exhaustive).toBe(false); expect(options.repetitions).toEqual({});
  expect(options.expanded.every((id) => id === 'encoder' || id.startsWith('encoder.'))).toBe(true);
  expect(projection.nodes.filter((node) => node.instances)).toEqual([]);
  expect(projection.nodes.filter((node) => node.record?.kind === 'operation').every((node) => node.id.startsWith('encoder.'))).toBe(true);
  enterComponent(view, graph, 'mlp:encoder.layer-1.gate');
  const mlp = projectGraph(graph, expandComponent(view, graph));
  expect(mlp.nodes.filter((node) => node.record?.kind === 'operation').map((node) => node.id)).toEqual([
    'encoder.layer-1.gate', 'encoder.layer-1.up', 'encoder.layer-1.silu', 'encoder.layer-1.multiply', 'encoder.layer-1.down']);
  expect(mlp.nodes.some((node) => node.id === 'mlp:encoder.layer-1.gate')).toBe(true);
});

it('does not corrupt previous views on an invalid isolation request or later option edits', () => {
  const graph = makeProjectionFixture({ count: 4 }), views = new GraphViews(), view = views.get('model', graph);
  view.repetitions = { 'decoder-layers': { start: 2, count: 1 } };
  const before = snapshotView(view);
  expect(() => enterComponent(view, graph, 'missing')).toThrow('no longer available');
  expect(snapshotView(view)).toEqual(before); expect(view.history).toEqual([]);
  enterComponent(view, graph, 'layer-2');
  view.repetitions['decoder-layers']!.count = 3; view.expanded.push('layer-2.attention');
  expect(view.history[0]).toEqual(before);
  returnToModel(view); expect(snapshotView(view)).toEqual(before);
});

it('explores the exact rebound child and restores the preceding shared instance with Back', () => {
  const graph = makeTemplateFixture(), view = new GraphViews().get('model', graph);
  enterSharedStructure(view, graph.templates![0]!, null);
  view.shared = { ...view.shared!, instanceId: 'layer-2.attention' };
  view.viewport = { x: -30, y: 52, zoom: 1.1 };
  view.selected = 'layer-2.attention.K';
  const shared = snapshotView(view);
  enterComponent(view, graph, 'layer-2.attention.Q');
  expect(view.shared).toBeUndefined();
  expect(view.scope).toBe('layer-2.attention.Q');
  const projection = projectGraph(graph, projectionOptions(view));
  expect(projection.nodes.filter((node) => node.record).map((node) => node.record!.id)).toEqual(['layer-2.attention.Q']);
  backFromComponent(view);
  expect(snapshotView(view)).toEqual(shared);
});
