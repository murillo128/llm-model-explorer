import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { makeProjectionFixture } from '../../tests/architecture-projection-fixture';
import { GraphViews, type GraphView } from './graph';
import { projectGraph } from './projection';
import { selectComponent, toggleComponent } from './component-actions';
import { backFromComponent, enterComponent, snapshotView } from './scope-navigation';
import { useGraphView } from './useGraphView';

const graph = makeProjectionFixture({ variants: ['full_attention', 'full_attention', 'full_attention', 'full_attention'] });
const node = (view: GraphView, id: string) => projectGraph(graph, view.getProjectionOptions()).nodes.find((n) => n.id === id)!;
const toggle = (view: GraphView, id: string) => toggleComponent(view, graph, node(view, id), 2);

it('shares selection and expansion immediately between two consumers, with stable selection-only layout inputs', () => {
  const view = new GraphViews().get('model', graph);
  view.update({ expanded: ['model'], repetitions: { 'decoder-layers': { start: 0, count: 2 } } });
  function Consumer({ name }: { name: string }) {
    const options = useGraphView(view);
    const layer = projectGraph(graph, options).nodes.find((n) => n.id === 'layer-0')!;
    return <section aria-label={name}>
      <button onClick={() => selectComponent(view, 'layer-0')}>{name} select</button>
      <button aria-expanded={layer.expanded} onClick={() => toggleComponent(view, graph, layer, 2)}>{name} toggle</button>
      <output aria-label={`${name} selected`}>{view.selected}</output>
    </section>;
  }
  const mounted = render(<><Consumer name="Canvas" /><Consumer name="Navigator" /></>);
  const before = view.getProjectionOptions();
  fireEvent.click(screen.getByText('Navigator select'));
  expect(screen.getByLabelText('Canvas selected')).toHaveTextContent('layer-0');
  expect(view.getProjectionOptions()).toBe(before);
  fireEvent.click(screen.getByText('Canvas toggle'));
  expect(screen.getByText('Navigator toggle')).toHaveAttribute('aria-expanded', 'true');
  fireEvent.click(screen.getByText('Navigator toggle'));
  expect(screen.getByText('Canvas toggle')).toHaveAttribute('aria-expanded', 'false');
  expect(view.selected).toBe('layer-0');
  mounted.unmount();
});

it('retains nested, derived, sibling and hidden source selection state through contraction and Back', () => {
  const original = JSON.stringify(graph), views = new GraphViews(), view = views.get('model', graph);
  view.update({ expanded: ['model'], repetitions: { 'decoder-layers': { start: 0, count: 2 } } });
  for (const id of ['layer-0', 'layer-0.attention', 'mlp:layer-0.gate', 'layer-1']) toggle(view, id);
  selectComponent(view, 'layer-0.attention.Q');
  const expanded = [...view.expanded];
  toggle(view, 'layer-0');
  expect(node(view, 'layer-0.attention')).toBeUndefined();
  expect(view.expanded).toEqual(expanded.filter((id) => id !== 'layer-0'));
  expect(views.get('model', graph).selected).toBe('layer-0.attention.Q');
  toggle(view, 'layer-0');
  expect(node(view, 'layer-0.attention').expanded).toBe(true);
  expect(node(view, 'mlp:layer-0.gate').expanded).toBe(true);
  expect(node(view, 'layer-1').expanded).toBe(true);
  const global = snapshotView(view);
  enterComponent(view, graph, 'layer-0');
  toggle(view, 'layer-0.attention');
  backFromComponent(view);
  expect(snapshotView(view)).toEqual(global);
  expect(JSON.stringify(graph)).toBe(original);
  expect(views.get('other-model', graph).selected).toBeNull();
  const replacement = views.get('model', { ...graph, graph_id: 'replacement' });
  expect(replacement.selected).toBeNull(); expect(replacement.expanded).toEqual([]);
});

it('reveals a range window without choosing a source instance or erasing expanded siblings', () => {
  const view = new GraphViews().get('model', graph);
  view.update({ expanded: ['model', 'layer-3', 'layer-3.attention'] });
  const range = projectGraph(graph, view.getProjectionOptions()).nodes.find((n) => n.repetitionId)!;
  selectComponent(view, range.id); toggleComponent(view, graph, range, 2);
  expect(view.selected).toBe(range.id);
  expect(view.repetitions['decoder-layers']).toEqual({ start: 0, count: 2 });
  expect(view.expanded).toEqual(['model', 'layer-3', 'layer-3.attention']);
  expect(node(view, 'layer-3.attention').expanded).toBe(true);
  expect(node(view, 'layer-0').expanded).toBe(false);
});

it('toggles current authoritative state even before a consumer receives a replacement projection', () => {
  const view = new GraphViews().get('model', graph);
  view.update({ expanded: ['model'], repetitions: { 'decoder-layers': { start: 0, count: 2 } } });
  const collapsed = node(view, 'layer-0');
  toggleComponent(view, graph, collapsed, 2);
  expect(view.expanded).toContain('layer-0');
  toggleComponent(view, graph, collapsed, 2);
  expect(view.expanded).not.toContain('layer-0');
});

it('contracts only the target after exhaustive expansion and restores the complete projection on reopening', () => {
  const view = new GraphViews().get('model', graph);
  view.update({ exhaustive: true, expanded: [], showContext: false, showUnused: false });
  const before = projectGraph(graph, view.getProjectionOptions());
  toggle(view, 'layer-0.attention');
  const contracted = projectGraph(graph, view.getProjectionOptions());
  expect(contracted.nodes.map((n) => n.id)).toEqual(before.nodes.filter((n) => n.parentId !== 'layer-0.attention').map((n) => n.id));
  toggle(view, 'layer-0.attention');
  expect(projectGraph(graph, view.getProjectionOptions())).toEqual(before);
});

it('ignores leaf/empty-group toggles and preserves projection references on remount and camera updates', () => {
  const views = new GraphViews(), view = views.get('model', graph);
  view.update({ expanded: ['model', 'layer-0', 'layer-0.attention'] });
  const before = view.getProjectionOptions(), revision = view.getRevision();
  toggle(view, 'layer-0.attention.Q');
  const empty = { ...node(view, 'layer-0'), record: { ...graph.nodes.find((n) => n.id === 'layer-0')!, kind: 'group' as const, children: [] } };
  toggleComponent(view, graph, empty, 2);
  expect(view.getRevision()).toBe(revision);
  views.get('model', graph); view.update({ viewport: { x: 1, y: 2, zoom: 0.8 } });
  expect(view.getProjectionOptions()).toBe(before);
});
