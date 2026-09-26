import { expect, it } from 'vitest';
import { interfaceFixture } from '../../tests/architecture-interface-fixture';
import { overviewFixture } from '../../tests/architecture-overview-fixture';
import { makeProjectionFixture } from '../../tests/architecture-projection-fixture';
import { GraphViews } from './graph';
import { interfaceIndex } from './interfaces';
import { projectGraph } from './projection';
import { layoutGraph } from './auto-layout';
import { initialViewport, minimumOverviewScale, overviewExpansion, overviewScale, visibleBounds } from './overview';

it.each([
  overviewFixture('compact'), overviewFixture('synthetic'), overviewFixture('training'),
  interfaceFixture('visual'), interfaceFixture('hybrid'), makeProjectionFixture({ secondStack: 2 }),
])('opens exactly the projected outer boundary for $graph_id', (graph) => {
  const before = structuredClone(graph), view = new GraphViews().get('model', graph);
  const outer = interfaceIndex(graph).outer;
  expect(view.expanded).toEqual(outer.kind === 'source' ? [outer.id] : []);
  const projection = projectGraph(graph, view.getProjectionOptions());
  expect(projection.nodes.filter((n) => n.expanded).map((n) => n.id)).toEqual([outer.id]);
  expect(projection.nodes.every((n) => n.id === outer.id || n.parentId === outer.id)).toBe(true);
  expect(projection.nodes.some((n) => interfaceIndex(graph).declarations.has(n.id))).toBe(false);
  expect(graph).toEqual(before);
});

it('keeps exact hybrid multiplicities without expanding representative interiors', () => {
  const graph = makeProjectionFixture({ variants: ['linear_attention', 'full_attention', 'linear_attention'] });
  const root = graph.nodes.find((n) => n.id === 'model')!;
  if (root.kind !== 'group') throw new Error('Fixture root is a group');
  graph.nodes.find((n) => n.id === 'head')!.parent_id = root.id; root.children.push('head');
  const projection = projectGraph(graph, { expanded: overviewExpansion(graph) });
  const stacks = projection.nodes.filter((n) => n.presentation === 'repetition');
  expect(stacks.flatMap((n) => n.instances!.map((i) => [i.index, i.variant]))).toEqual([[0, 'linear_attention'], [1, 'full_attention'], [2, 'linear_attention']]);
  expect(stacks.every((n) => !n.expanded)).toBe(true);
});

it('uses a nonzero viewport and an inclusive, exact readable threshold', () => {
  const bounds = { x: 24, y: 60, width: 1000, height: 500 };
  expect(overviewScale(bounds, 0, 600)).toBeUndefined();
  expect(overviewScale(bounds, 900, 0)).toBeUndefined();
  expect(overviewScale(bounds, 832, 432)).toBe(minimumOverviewScale);
  expect(overviewScale(bounds, 831, 432)).toBeLessThan(minimumOverviewScale);
  expect(overviewScale(bounds, 832, 431)).toBeLessThan(minimumOverviewScale);
  expect(initialViewport(bounds, 832, 432)).toEqual({ x: -3.200000000000003, y: -32, zoom: 0.8 });
  const tiny = initialViewport({ x: 24, y: 24, width: 180, height: 84 }, 1142, 771);
  expect(tiny).toEqual({ x: 457, y: -8, zoom: 1 });
});

it('measures compact and wide candidates from actual visible layout bounds', async () => {
  const compact = overviewFixture('compact'), wide = overviewFixture('wide');
  const small = await layoutGraph(compact, { expanded: overviewExpansion(compact) });
  const large = await layoutGraph(wide, { expanded: overviewExpansion(wide) });
  expect(overviewScale(visibleBounds(small), 388, 493)).toBeGreaterThanOrEqual(minimumOverviewScale);
  expect(visibleBounds(large).width).toBeGreaterThan(1142 * 4);
  expect(overviewScale(visibleBounds(large), 1142, 771)).toBeLessThan(minimumOverviewScale);
  const closed = await layoutGraph(wide, { expanded: [], modelCollapsed: true });
  expect(closed.boxes.map((b) => b.id)).toEqual(['Model']);
  expect(overviewScale(visibleBounds(closed), 388, 493)).toBe(1);
});

it('retains saved expansion/camera and bounds the graph identity cache', () => {
  const graph = overviewFixture('compact'), views = new GraphViews(), view = views.get('model', graph);
  view.update({ expanded: [], viewport: { x: -70, y: 100, zoom: 2 } });
  expect(view.initialOverview).toBe(false);
  expect(views.get('model', graph)).toBe(view);
  expect(view.viewport).toEqual({ x: -70, y: 100, zoom: 2 });
  expect(views.get('model', { ...graph, graph_id: 'replacement' }).initialOverview).toBe(true);
  for (let i = 0; i < 9; i++) views.get(`other-${i}`, graph);
  expect(views.get('model', graph)).not.toBe(view);
});
