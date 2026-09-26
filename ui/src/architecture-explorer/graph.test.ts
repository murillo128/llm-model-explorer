import { layoutGraph } from './auto-layout';
import { expect, it, vi } from 'vitest';
import fixture from '../../../api/fixtures/architecture.json';
import { validateSchema } from '../api/validation';
import { formatShape, GraphViews } from './graph';
import { projectGraph } from './projection';
import { requestLayout } from './layout';
import { referenceFixture, references } from '../../tests/architecture-fixtures';
import { validateArchitecture } from '../api/architecture-validation';

const graph = validateSchema('ArchitectureAvailableResponse', fixture.response).graph;
it.each(references)('validates full-size $name fixtures and preserves exact configured instance order', async (reference) => {
  const response = referenceFixture(reference.name);
  expect(() => validateArchitecture(response, { modelId: reference.model })).not.toThrow();
  const layout = await layoutGraph(response.graph, { expanded: [], exhaustive: true });
  expect(layout.boxes).toHaveLength(response.graph.nodes.length);
  expect(layout.edgeIds).toEqual(response.graph.edges.map((e) => e.id));
  expect(response.graph.repetitions.map((r) => r.instances.length)).toEqual(reference.stacks);
  if (reference.hybrid) expect(response.graph.repetitions[0]!.instances.map((i) => i.variant)).toEqual(
    Array.from({ length: 6 }, () => ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']).flat());
}, 60_000);
it('projects and lays out the GLM layer stack and a focused routed-expert window', async () => {
  const response = referenceFixture('glm4');
  expect(() => validateArchitecture(response, { modelId: response.model_id })).not.toThrow();
  const source = response.graph;
  const layers = source.repetitions.find((item) => item.id === 'repeat-0')!;
  expect(layers.instances).toHaveLength(47);
  expect(layers.instances.map((item) => item.variant)).toEqual(['dense', ...Array(46).fill('sparse')]);
  const compact = projectGraph(source, { expanded: ['root', 'stack-0'] });
  expect(compact.nodes.find((node) => node.repetitionId === layers.id)?.summary).toBe('1 dense · 46 sparse');

  const expertRepetition = source.repetitions.find((item) => item.id === 'repeat-experts-0-1')!;
  expect(expertRepetition.instances).toHaveLength(64);
  const options = { expanded: ['root', 'stack-0', 'layer-0-1', 'layer-0-1-routed-experts'],
    repetitions: { 'repeat-0': { start: 1, count: 1 }, [expertRepetition.id]: { start: 61, count: 3 } } };
  const focused = projectGraph(source, options);
  expect(focused.nodes.map((item) => item.id)).toEqual(expect.arrayContaining(
    [61, 62, 63].map((index) => `layer-0-1-expert-${index}`)));
  const layout = await layoutGraph(source, options);
  expect(layout.boxes.map((box) => box.id)).toEqual(expect.arrayContaining(
    [61, 62, 63].map((index) => `layer-0-1-expert-${index}`)));
}, 60_000);
it('retains every expanded record and source connection while compacting repeated siblings', async () => {
  const all = await layoutGraph(graph, { expanded: [], exhaustive: true });
  expect(all.boxes.map((b) => b.id).sort()).toEqual(graph.nodes.map((n) => n.id).sort());
  expect(all.edgeIds).toEqual(graph.edges.map((e) => e.id));
  const compact = await layoutGraph(graph, { expanded: ['root'] });
  expect(compact.boxes.map((b) => b.id)).toEqual(['root', 'repeat:layers:0:1', 'tokens']);
  expect(compact.projection.nodes.find((n) => n.presentation === 'repetition')!.sourceIds).toEqual(['layer0', 'layer1']);
  const crossing = graph.edges.find((e) => e.source.node_id === 'layer0' && e.target.node_id === 'layer1')!;
  expect(compact.edgeIds).not.toContain(crossing.id);
  expect(compact.projection.hiddenEdgeIds).toContain(crossing.id);
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
it('retains an exact repetition presentation across remounts, but clears obsolete ranges and replacement graphs', () => {
  const views = new GraphViews(), view = views.get('a', graph);
  view.update({ selected: 'repeat:layers:0:1' });
  expect(views.get('a', graph).selected).toBe('repeat:layers:0:1');
  expect(views.get('b', graph).selected).toBeNull();
  expect(views.get('a', graph).selected).toBe('repeat:layers:0:1');
  view.update({ expanded: ['root', 'layer1'] });
  expect(views.get('a', graph).selected).toBeNull();
  view.update({ selected: 'repeat:layers:0:99' });
  expect(views.get('a', graph).selected).toBeNull();
  view.update({ expanded: ['root'], selected: 'repeat:layers:0:1' });
  expect(views.get('a', { ...graph, graph_id: 'replacement' }).selected).toBeNull();
});
function layoutWorker() {
  return { postMessage: vi.fn(), terminate: vi.fn(),
    onmessage: null as ((event: MessageEvent) => void) | null, onerror: null as (() => void) | null };
}
it.each(['abort', 'timeout'])('rejects %s immediately and terminates after the nested-worker teardown acknowledgement', async (cause) => {
  vi.useFakeTimers();
  try {
    const worker = layoutWorker();
    const create = () => worker as unknown as Worker;
    const abort = new AbortController();
    const pending = requestLayout(graph, { expanded: [] }, abort.signal, create);
    const assertion = cause === 'abort' ? expect(pending).rejects.toMatchObject({ name: 'AbortError' }) : expect(pending).rejects.toThrow('exceeded');
    if (cause === 'abort') abort.abort(); else await vi.advanceTimersByTimeAsync(10_000);
    await assertion;
    expect(worker.postMessage).toHaveBeenLastCalledWith({ cancel: true });
    expect(worker.terminate).not.toHaveBeenCalled();
    worker.onmessage!({ data: { cancelled: true } } as MessageEvent);
    expect(worker.terminate).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(worker.terminate).toHaveBeenCalledOnce();
  } finally { vi.useRealTimers(); }
});
it.each(['abort', 'timeout'])('bounds %s worker teardown to 1000ms when no acknowledgement arrives', async (cause) => {
  vi.useFakeTimers();
  try {
    const worker = layoutWorker(), abort = new AbortController();
    const pending = requestLayout(graph, { expanded: [] }, abort.signal, () => worker as unknown as Worker);
    const assertion = cause === 'abort' ? expect(pending).rejects.toMatchObject({ name: 'AbortError' }) : expect(pending).rejects.toThrow('exceeded');
    if (cause === 'abort') abort.abort(); else await vi.advanceTimersByTimeAsync(10_000);
    await assertion;
    await vi.advanceTimersByTimeAsync(999); expect(worker.terminate).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1); expect(worker.terminate).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
  } finally { vi.useRealTimers(); }
});
it('rejects late layout replies after cancellation instead of publishing obsolete geometry', async () => {
  vi.useFakeTimers();
  try {
    const worker = layoutWorker(), abort = new AbortController(), published = vi.fn(), rejected = vi.fn();
    const pending = requestLayout(graph, { expanded: [] }, abort.signal, () => worker as unknown as Worker);
    const originalReply = worker.onmessage!;
    const outcome = pending.then(published, rejected);
    abort.abort(); await outcome;
    expect(rejected).toHaveBeenCalledWith(expect.objectContaining({ name: 'AbortError' }));
    const late = { data: { layout: { boxes: [{ id: 'obsolete' }] } } } as MessageEvent;
    originalReply(late); // An already queued completion callback also sees the settled request.
    worker.onmessage!(late); // The active callback reaps the owner even when completion races its ACK.
    await Promise.resolve();
    expect(published).not.toHaveBeenCalled(); expect(worker.terminate).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
  } finally { vi.useRealTimers(); }
});
