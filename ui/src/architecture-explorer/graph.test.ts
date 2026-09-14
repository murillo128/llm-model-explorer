import { expect, it, vi } from 'vitest';
import fixture from '../../../api/fixtures/architecture.json';
import { validateSchema } from '../api/validation';
import { formatShape, GraphViews, layoutGraph } from './graph';
import { requestLayout } from './layout';
import { referenceFixture, references } from '../../tests/architecture-fixtures';
import { validateArchitecture } from '../api/architecture-validation';

const graph = validateSchema('ArchitectureAvailableResponse', fixture.response).graph;
it.each(references)('validates full-size $name fixtures and preserves exact configured instance order', (reference) => {
  const response = referenceFixture(reference.name);
  expect(() => validateArchitecture(response, { modelId: reference.model, tokenizerAvailable: !reference.visual,
    inventory: { tensors: [], coverage: 'partial', diagnostics: [] } })).not.toThrow();
  const layout = layoutGraph(response.graph, response.graph.nodes.filter((n) => n.kind === 'group').map((n) => n.id));
  expect(layout.boxes).toHaveLength(response.graph.nodes.length);
  expect(layout.edgeIds).toEqual(response.graph.edges.map((e) => e.id));
  expect(response.graph.repetitions.map((r) => r.instances.length)).toEqual(reference.stacks);
  if (reference.hybrid) expect(response.graph.repetitions[0]!.instances.map((i) => i.variant)).toEqual(
    Array.from({ length: 6 }, () => ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']).flat());
});
it('retains every expanded record and exact crossing edge, with parent-before-child layout', () => {
  const all = layoutGraph(graph, graph.nodes.filter((n) => n.kind === 'group').map((n) => n.id));
  expect(all.boxes.map((b) => b.id).sort()).toEqual(graph.nodes.map((n) => n.id).sort());
  expect(all.edgeIds).toEqual(graph.edges.map((e) => e.id));
  const compact = layoutGraph(graph, ['root']);
  expect(compact.boxes.map((b) => b.id)).toEqual(['root', 'layer0', 'layer1', 'tokens']);
  const crossing = graph.edges.find((e) => e.source.node_id === 'layer0' && e.target.node_id === 'layer1')!;
  expect(compact.edgeIds).toContain(crossing.id);
  for (const box of all.boxes) {
    if (!box.parentId) continue;
    const parent = all.boxes.find((p) => p.id === box.parentId)!;
    expect(all.boxes.indexOf(parent)).toBeLessThan(all.boxes.indexOf(box));
    expect(box.x + box.width).toBeLessThanOrEqual(parent.width);
    expect(box.y + box.height).toBeLessThanOrEqual(parent.height);
  }
});
it('keeps graph/model camera and selection independent, clears obsolete selections', () => {
  const views = new GraphViews();
  const first = views.get('a', graph);
  first.update({ selected: 'linear1', expanded: ['root', 'layer1'], viewport: { x: 10, y: 20, zoom: 2 } });
  expect(views.get('a', graph)).toBe(first);
  expect(views.get('b', graph).selected).toBeNull();
  expect(views.get('a', { ...graph, graph_id: 'replacement' }).selected).toBeNull();
  first.update({ selected: 'nonexistent' }); expect(views.get('a', graph).selected).toBeNull();
});
it('renders dimensions as text, distinguishing scalar, unknown, symbol and expression', () => {
  expect(formatShape(null)).toContain('unknown rank'); expect(formatShape([])).toBe('scalar');
  expect(formatShape([{ kind: 'constant', value: 3 }, { kind: 'symbol', name: 'B' },
    { kind: 'expression', text: 'window.alert(1)', symbols: [] }, { kind: 'unknown', reason: 'unresolved' }]))
    .toBe('[3 × B × (window.alert(1)) × ? (unresolved)]');
});
it('terminates obsolete and timed-out layout workers', async () => {
  vi.useFakeTimers();
  try {
    const worker = { postMessage: vi.fn(), terminate: vi.fn(), onmessage: null, onerror: null };
    const create = () => worker as unknown as Worker;
    const abort = new AbortController();
    const pending = requestLayout(graph, [], abort.signal, create);
    abort.abort(); await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(worker.terminate).toHaveBeenCalledOnce();
    const timeout = requestLayout(graph, [], new AbortController().signal, create);
    const assertion = expect(timeout).rejects.toThrow('exceeded');
    await vi.advanceTimersByTimeAsync(10_001); await assertion;
    expect(worker.terminate).toHaveBeenCalledTimes(2);
  } finally { vi.useRealTimers(); }
});
