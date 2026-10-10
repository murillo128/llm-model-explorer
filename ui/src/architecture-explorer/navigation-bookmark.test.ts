import { expect, it } from 'vitest';
import fixture from '../../tests/fixtures/architecture-refresh.json';
import { validateSchema } from '../api/validation';
import { GraphView, GraphViews } from './graph';
import type { Graph } from './graph';
import { captureBookmark, restoreBookmark, restoreInspection } from './navigation-bookmark';
import { backFromComponent, enterComponent, projectionOptions } from './scope-navigation';
import { projectGraph } from './projection';
import { enterSharedStructure } from './shared-structure';
import { indexedNodeId } from './indexed-repetition';
import { interfaceIndex } from './interfaces';
import { makeIndexedFixture, makeTemplateFixture } from '../../tests/architecture-template-fixture';
import { makeProjectionFixture } from '../../tests/architecture-projection-fixture';
import type { ArchitectureSelection } from './ArchitectureCanvas';

const before = validateSchema('ArchitectureGraph', fixture.before), after = validateSchema('ArchitectureGraph', fixture.after);
const linear = before.nodes.find((n) => n.operation === 'linear')!;
const replacement = after.nodes.find((n) => n.operation === 'linear')!;
const captureSelection = (parameter = false): ArchitectureSelection => ({ node: linear, graphId: before.graph_id, modelId: 'model', sessionId: 'old-session', trigger: document.body,
  ...(parameter ? { parameterId: linear.parameter_ids[0]! } : {}) });

it('restores real importer revisions with every runtime ID changed, renamed content and Back', () => {
  const views = new GraphViews(), view = views.get('model', before);
  view.cameraAnchor = { target: interfaceIndex(before).outer.id, x: 0.3, y: 0.4, zoom: 0.9 };
  enterComponent(view, before, linear.parent_id!);
  view.selected = linear.id;
  view.cameraAnchor = { target: linear.id, x: 0.35, y: 0.6, zoom: 1.2 };
  view.browser.query = 'projection'; view.browser.treeScroll = 140;
  const bookmark = captureBookmark(before, view);
  expect(JSON.stringify(bookmark)).not.toContain(before.graph_id);
  for (const node of before.nodes) expect(JSON.stringify(bookmark)).not.toContain(node.id);
  const restored = views.get('model', after, bookmark);
  expect(restored.selected).toBe(replacement.id);
  expect(restored.scope).toBe(replacement.parent_id);
  expect(replacement.label).toBe('Renamed projection');
  expect(restored.restoreCamera).toEqual({ target: replacement.id, x: 0.35, y: 0.6, zoom: 1.2 });
  expect(restored.initialOverview).toBe(false);
  expect(restored.browser).toMatchObject({ query: 'projection', treeScroll: 140 });
  expect(restored.history).toHaveLength(1);
  backFromComponent(restored); expect(restored.scope).toBeUndefined();
  expect(restored.restoreCamera).toEqual({ target: interfaceIndex(after).outer.id, x: 0.3, y: 0.4, zoom: 0.9 });
  expect(views.get('unrelated-model', after).selected).toBeNull();
});

it.each(['removed', 'retyped', 'ambiguous'] as const)('falls back to a surviving ancestor for a %s child without using duplicate labels', (change) => {
  const view = new GraphView(); view.selected = linear.id; view.scope = linear.parent_id;
  const next = structuredClone(after);
  const other = next.nodes.find((n) => n.id !== replacement.id && n.kind !== 'group')!;
  other.label = replacement.label;
  if (change === 'removed') {
    next.nodes = next.nodes.filter((n) => n.id !== replacement.id);
    for (const node of next.nodes) if (node.kind === 'group') node.children = node.children.filter((id) => id !== replacement.id);
    next.edges = next.edges.filter((e) => e.source.node_id !== replacement.id && e.target.node_id !== replacement.id);
  } else if (change === 'retyped') next.nodes.find((n) => n.id === replacement.id)!.operation = 'unrelated';
  else other.navigation_key = replacement.navigation_key!;
  // Duplicate-key negative control also has the same kind, operation and scope.
  if (change === 'ambiguous') Object.assign(other, { kind: replacement.kind, operation: replacement.operation, parent_id: replacement.parent_id });
  const restored = restoreBookmark(next, captureBookmark(before, view));
  expect(restored.selected).toBe(replacement.parent_id);
  expect(restored.notice).toContain('nearest');
  expect(restored.restoreCamera?.target).toBe(replacement.parent_id);
});

it('incompatible or absent namespaces keep ordinary overview behavior', () => {
  const view = new GraphView(); view.selected = linear.id;
  const bookmark = captureBookmark(before, view);
  for (const graph of [{ ...after, navigation_namespace: 'incompatible' }, { ...after, navigation_namespace: undefined }]) {
    const restored = restoreBookmark(graph as Graph, bookmark);
    expect(restored.selected).toBeNull(); expect(restored.initialOverview).toBe(true);
  }
});

it('remaps endpoints by node-local port identity and clears only an obsolete port selection', () => {
  const view = new GraphView(); view.selected = linear.id;
  const port = linear.ports[0]!;
  view.boundary = { kind: 'boundary', owner: { kind: 'source', id: linear.id }, endpoints: [{ node_id: linear.id, port_id: port.id }] };
  const edge = before.edges.find((e) => e.target.node_id === linear.id)!; view.edge = edge.id;
  const bookmark = captureBookmark(before, view), restored = restoreBookmark(after, bookmark);
  expect(restored.boundary?.endpoints).toEqual([{ node_id: replacement.id, port_id: port.id }]);
  expect(restored.edge).not.toBe(edge.id); expect(restored.edge).not.toBeNull();
  const next = structuredClone(after);
  delete next.nodes.find((n) => n.id === replacement.id)!.ports.find((p) => p.id === port.id)!.navigation_key;
  const missing = restoreBookmark(next, bookmark);
  expect(missing.selected).toBe(replacement.id); expect(missing.boundary).toBeUndefined();
});

it('reopens only exact current parameter and inventory bindings with a new session', () => {
  const view = new GraphView(); view.selected = linear.id;
  const bookmark = captureBookmark(before, view, captureSelection(true));
  const parameter = after.parameters.find((p) => p.id === replacement.parameter_ids[0])!;
  const tensorId = parameter.inspection.status === 'available' ? parameter.inspection.tensor_id : '';
  const inventory = { coverage: 'complete' as const, diagnostics: [], tensors: [{ id: tensorId, name: parameter.name, shape: [4, 3], rank: 2, logical_dtype: 'float32' as const, storage_dtype: 'F32', numel: 12, path: ['encoder', 'proj', 'weight'] }] };
  const result = restoreInspection(after, bookmark, inventory, 'new-session', 'model', document.body);
  expect(result).toMatchObject({ sessionId: 'new-session', graphId: after.graph_id, node: replacement, parameterId: parameter.id });
  expect(result?.parameterId).not.toBe(linear.parameter_ids[0]);
  expect(restoreInspection(after, bookmark, { ...inventory, tensors: [] }, 'new-session', 'model', document.body)).toBeUndefined();
  expect(restoreInspection(after, bookmark, { ...inventory, tensors: [{ ...inventory.tensors[0]!, name: 'another.weight' }] }, 'new-session', 'model', document.body)).toBeUndefined();
});

/** Authored identity metadata; changing runtime identifiers is independent of the remapper. */
function revision(graph: Graph, suffix: string): Graph {
  const copy = structuredClone(graph);
  copy.navigation_namespace = 'fixture-domain';
  for (const record of [...copy.nodes, ...copy.repetitions, ...copy.templates ?? []]) record.navigation_key = `semantic/${record.id}`;
  for (const node of copy.nodes) for (const port of node.ports) port.navigation_key = `port/${port.id}`;
  const ids = new Set([copy.graph_id, ...copy.nodes.map((n) => n.id), ...copy.edges.map((e) => e.id), ...copy.parameters.map((p) => p.id),
    ...copy.repetitions.map((r) => r.id), ...copy.templates?.map((t) => t.id) ?? []]);
  function rewrite(value: unknown, field = ''): unknown {
    if (typeof value === 'string') return ['id', 'graph_id', 'node_id', 'parent_id', 'edge_id', 'parameter_id', 'children', 'parameter_ids'].includes(field) && ids.has(value) ? value + suffix : value;
    if (Array.isArray(value)) return value.map((item) => rewrite(item, field));
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, rewrite(item, key === 'id' && 'direction' in value ? 'port' : key)]));
    return value;
  }
  return rewrite(copy) as Graph;
}

it.each([null, 2])('keeps Shared structure neutral or exactly bound to nonzero instance %s', (instance) => {
  const old = revision(makeTemplateFixture(), '-old'), next = revision(makeTemplateFixture(), '-new');
  const view = new GraphView();
  enterSharedStructure(view, old.templates![0]!, instance === null ? null : `layer-${instance}.attention-old`);
  const restored = restoreBookmark(next, captureBookmark(old, view));
  expect(restored.shared?.instanceId).toBe(instance === null ? null : 'layer-2.attention-new');
  expect(restored.selectionMode).toBe(instance === null ? 'structure' : 'source');
  next.templates![0]!.instances = next.templates![0]!.instances.slice(0, 1);
  const missing = restoreBookmark(next, captureBookmark(old, view));
  if (instance !== null) {
    expect(missing.shared).toBeUndefined();
    // The source still exists even though it no longer belongs to the family.
    expect(missing.selected).toBe('layer-2.attention-new');
    expect(missing.scope).toBe('layer-2.attention-new');
  }
});

it('retains the surviving ancestor when a concrete Shared subtree disappears', () => {
  const old = revision(makeTemplateFixture(), '-old'), next = revision(makeTemplateFixture(), '-new');
  const view = new GraphView();
  enterSharedStructure(view, old.templates![0]!, 'layer-2.attention-old');
  const removed = new Set(next.templates![0]!.instances.find((i) => i.node_id === 'layer-2.attention-new')!.nodes.map((n) => n.node_id));
  next.nodes = next.nodes.filter((n) => !removed.has(n.id));
  for (const node of next.nodes) if (node.kind === 'group') node.children = node.children.filter((id) => !removed.has(id));
  next.edges = next.edges.filter((e) => !removed.has(e.source.node_id) && !removed.has(e.target.node_id));
  next.templates = [];
  expect(next.nodes.some((n) => n.id === 'layer-2-new')).toBe(true);
  const restored = restoreBookmark(next, captureBookmark(old, view));
  expect(restored.shared).toBeUndefined();
  expect(restored.selected).toBe('layer-2-new');
  expect(restored.scope).toBe('layer-2-new');
  expect(restored.selectionMode).toBe('source');
  expect(restored.restoreCamera?.target).toBe('layer-2-new');
  expect(restored.notice).toContain('nearest');
});

it('does not promote a neutral Shared view when its family disappears', () => {
  const old = revision(makeTemplateFixture(), '-old'), next = revision(makeTemplateFixture(), '-new');
  const view = new GraphView();
  enterSharedStructure(view, old.templates![0]!, null);
  next.templates = [];
  const restored = restoreBookmark(next, captureBookmark(old, view));
  expect(restored.shared).toBeUndefined(); expect(restored.scope).toBeUndefined();
  expect(restored.selected).toBeNull(); expect(restored.initialOverview).toBe(true);
});

it.each(['0:1', '4:5'])('preserves compressed range %s beside an open repetition window', (range) => {
  const old = revision(makeIndexedFixture(6), '-old'), next = revision(makeIndexedFixture(6), '-new');
  const view = new GraphView(['model-old']);
  view.repetitions = { 'layers-old': { start: 2, count: 2 } };
  view.selected = `repeat:layers-old:${range}`;
  expect(projectGraph(old, projectionOptions(view)).nodes.some((n) => n.id === view.selected)).toBe(true);
  const restored = restoreBookmark(next, captureBookmark(old, view));
  expect(projectGraph(next, projectionOptions(restored)).nodes.some((n) => n.id === `repeat:layers-new:${range}`)).toBe(true);
  expect(restored.selected).toBe(`repeat:layers-new:${range}`);
  expect(restored.repetitions).toEqual({ 'layers-new': { start: 2, count: 2 } });
  expect(restored.notice).toBeUndefined();
});

it('restores an indexed role and clamps a shrinking concrete repetition window', () => {
  const old = revision(makeIndexedFixture(4), '-old'), next = revision(makeIndexedFixture(3), '-new');
  const view = new GraphView(['model-old', 'repeat:layers-old:0:3', indexedNodeId('layers-old', 'layer-0-old')]);
  view.selected = indexedNodeId('layers-old', 'layer-0.op-old');
  const restored = restoreBookmark(next, captureBookmark(old, view));
  expect(restored.selected).toBe(indexedNodeId('layers-new', 'layer-0.op-new'));
  view.selected = 'layer-2.op-old'; view.repetitions = { 'layers-old': { start: 2, count: 2 } };
  const concrete = restoreBookmark(next, captureBookmark(old, view));
  expect(concrete.selected).toBe('layer-2.op-new');
  expect(concrete.repetitions).toEqual({ 'layers-new': { start: 2, count: 1 } });
});

it('maps a derived MLP through its source owner and falls back to that owner if its topology changes', () => {
  const old = revision(makeProjectionFixture({ count: 2 }), '-old');
  const next = revision(makeProjectionFixture({ count: 2 }), '-new');
  const view = new GraphView(['model-old', 'layer-1-old']);
  view.selected = 'mlp:layer-1.gate-old';
  const bookmark = captureBookmark(old, view);
  expect(restoreBookmark(next, bookmark).selected).toBe('mlp:layer-1.gate-new');
  next.nodes.find((node) => node.id === 'layer-1.gate-new')!.operation = 'changed';
  expect(restoreBookmark(next, bookmark).selected).toBe('layer-1-new');
});

it('bounds repeated replacements and history without retaining old graph records', () => {
  const views = new GraphViews(); let view = views.get('model', before), graph = before;
  for (let i = 0; i < 20; i++) {
    view.selected = graph.nodes.find((n) => n.operation === 'linear')!.id;
    const bookmark = captureBookmark(graph, view);
    graph = i % 2 ? before : after; view = views.get('model', graph, bookmark);
    expect(view.selected).toBe(graph.nodes.find((n) => n.operation === 'linear')!.id);
    expect(view.history.length).toBeLessThanOrEqual(16);
    expect(Object.values(view).includes(before)).toBe(false);
  }
});
