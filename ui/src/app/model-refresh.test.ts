import { afterEach, expect, it, vi } from 'vitest';
import { ApiClient } from '../api/client';
import { deferred, json, memoryStorage, models, sessionA, tensors } from '../test/shell-fixtures';
import { SessionController } from './session-controller';

class Events extends EventTarget {
  static instances: Events[] = [];
  closed = false;
  constructor(readonly url: string) { super(); Events.instances.push(this); }
  close() { this.closed = true; }
  state(revision: string | null, sequence = 1, epoch = '11111111-1111-4111-8111-111111111111', model = sessionA.model_id) {
    this.dispatchEvent(new MessageEvent('model-state', { lastEventId: `${epoch}:${sequence}`, data: JSON.stringify({
      epoch, sequence, model_id: model, status: revision === null ? 'unavailable' : 'present', model_revision: revision,
    }) }));
  }
}
const tick = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
const controllers: SessionController[] = [];
afterEach(() => { controllers.forEach(c => c.dispose()); controllers.length = 0; vi.unstubAllGlobals(); vi.useRealTimers(); });
async function setup() {
  Events.instances = []; vi.stubGlobal('EventSource', Events);
  const fetcher = vi.fn<typeof fetch>();
  const client = new ApiClient({ backendBaseUrl: 'https://backend.example' }, fetcher);
  vi.spyOn(client, 'listModels').mockResolvedValue({ models, diagnostics: [] });
  vi.spyOn(client, 'listTensors').mockResolvedValue({ tensors, coverage: 'complete', diagnostics: [] });
  vi.spyOn(client, 'createSession').mockResolvedValue(sessionA);
  vi.spyOn(client, 'deleteSession').mockResolvedValue();
  const controller = new SessionController(client, 'https://backend.example', memoryStorage()); controllers.push(controller);
  controller.start(); await tick(); controller.chooseModel(sessionA.model_id); await tick();
  return { controller, client, fetcher };
}
it('automatically replaces only changed active content and resolves the exact tensor in fresh inventory', async () => {
  const { controller, client } = await setup();
  controller.selectTensor(tensors[0]!);
  const old = controller.getSnapshot().view;
  expect(Events.instances).toHaveLength(1);
  const events = Events.instances[0]!;
  events.state('snapshot_A'); events.state('other', 2, undefined, models[1]!.id);
  expect(controller.getSnapshot().view).toBe(old);
  expect(client.createSession).toHaveBeenCalledTimes(1);
  const replacement = { ...sessionA, id: 'cccccccc-cccc-4ccc-8ccc-cccccccccccc', model_revision: 'snapshot_B' };
  vi.mocked(client.createSession).mockResolvedValue(replacement);
  const changed = { ...tensors[0]!, shape: [6], rank: 1 };
  vi.mocked(client.listTensors).mockResolvedValue({ tensors: [tensors[1]!, changed], coverage: 'complete', diagnostics: [] });
  events.state('snapshot_B', 3); events.state('snapshot_B', 3); await tick();
  Events.instances.at(-1)!.state('snapshot_B', 4); await tick();
  expect(old.signal.aborted).toBe(true);
  expect(controller.getSnapshot().session).toEqual(replacement);
  expect(controller.getSnapshot().selected).toEqual(changed);
  expect(client.createSession).toHaveBeenCalledTimes(2);
  expect(client.deleteSession).toHaveBeenCalledWith(sessionA.id);
});

it('routes genuine HTTP content invalidation through the active client observer', async () => {
  const { controller, client, fetcher } = await setup();
  const pending = deferred<typeof sessionA>(); vi.mocked(client.createSession).mockReturnValue(pending.promise);
  fetcher.mockResolvedValue(json({ code: 'model_content_changed', message: '/private' }, 409));
  const active = controller.explorerClient(controller.getSnapshot().view);
  await expect(active.tokenize(sessionA.id, { text: '', add_special_tokens: true })).rejects.toThrow();
  expect(client.createSession).toHaveBeenCalledTimes(2);
  expect(controller.getSnapshot().session).toBeNull();
  expect(controller.getSnapshot().message).not.toContain('/private');
  pending.resolve({ ...sessionA, id: 'cccccccc-cccc-4ccc-8ccc-cccccccccccc', model_revision: 'snapshot_B' }); await tick();
});

it('coalesces a burst and reconciles a POST that pinned newer content than every old event', async () => {
  const { controller, client } = await setup();
  const pending = deferred<typeof sessionA>(); vi.mocked(client.createSession).mockReturnValueOnce(pending.promise);
  const events = Events.instances[0]!;
  events.state('B'); events.state('C', 2); events.state('D', 3); events.state('B', 1);
  expect(client.createSession).toHaveBeenCalledTimes(2);
  pending.resolve({ ...sessionA, id: 'cccccccc-cccc-4ccc-8ccc-cccccccccccc', model_revision: 'E' }); await tick();
  expect(controller.getSnapshot().session).toBeNull();
  expect(events.closed).toBe(true);
  events.state('obsolete', 9);
  Events.instances.at(-1)!.state('E', 10); await tick();
  expect(client.createSession).toHaveBeenCalledTimes(2);
  expect(controller.getSnapshot().session?.model_revision).toBe('E');
});

it('discards a superseded pinned snapshot and performs one follow-up for current state', async () => {
  const { controller, client } = await setup();
  const pending = deferred<typeof sessionA>(); vi.mocked(client.createSession).mockReturnValueOnce(pending.promise);
  Events.instances[0]!.state('B'); Events.instances[0]!.state('C', 2);
  const intermediate = { ...sessionA, id: 'cccccccc-cccc-4ccc-8ccc-cccccccccccc', model_revision: 'B' };
  pending.resolve(intermediate); await tick();
  const final = { ...intermediate, id: 'dddddddd-dddd-4ddd-8ddd-dddddddddddd', model_revision: 'C' };
  vi.mocked(client.createSession).mockResolvedValueOnce(final);
  Events.instances.at(-1)!.state('C', 3); await tick();
  expect(controller.getSnapshot().session).toBeNull();
  expect(client.listTensors).toHaveBeenCalledTimes(1);
  Events.instances.at(-1)!.state('C', 4); await tick();
  expect(controller.getSnapshot().session).toEqual(final);
  expect(client.createSession).toHaveBeenCalledTimes(3);
  expect(client.deleteSession).toHaveBeenCalledWith(intermediate.id);
});

it('keeps a valid session usable during transport loss and detects a restarted epoch', async () => {
  const { controller, client } = await setup();
  const events = Events.instances[0]!; events.state('snapshot_A');
  const view = controller.getSnapshot().view;
  events.dispatchEvent(new Event('error'));
  expect(controller.getSnapshot()).toMatchObject({ session: sessionA, observation: 'reconnecting', toasts: [] });
  events.state('snapshot_A', 2);
  expect(controller.getSnapshot().view).toBe(view);
  const pending = deferred<typeof sessionA>(); vi.mocked(client.createSession).mockReturnValueOnce(pending.promise);
  events.state('snapshot_A', 1, '22222222-2222-4222-8222-222222222222');
  expect(client.createSession).toHaveBeenCalledTimes(2);
  controller.closeSession(); pending.resolve({ ...sessionA, id: 'cccccccc-cccc-4ccc-8ccc-cccccccccccc' }); await tick();
  expect(client.deleteSession).toHaveBeenCalledWith('cccccccc-cccc-4ccc-8ccc-cccccccccccc');
  expect(controller.getSnapshot()).toMatchObject({ session: null, selectedModelId: null, sessionStatus: 'idle' });
});

it('waits through removal and repairs the same identity without choosing another model', async () => {
  const { controller, client } = await setup(); controller.selectTensor(tensors[0]!);
  const events = Events.instances[0]!; events.state(null);
  expect(controller.getSnapshot()).toMatchObject({ selectedModelId: sessionA.model_id, session: null, refreshStatus: 'waiting' });
  expect(events.closed).toBe(false);
  expect(client.createSession).toHaveBeenCalledTimes(1);
  vi.mocked(client.createSession).mockResolvedValueOnce({ ...sessionA, id: 'cccccccc-cccc-4ccc-8ccc-cccccccccccc', model_revision: 'repaired' });
  vi.mocked(client.listTensors).mockResolvedValueOnce({ tensors: [tensors[1]!], coverage: 'complete', diagnostics: [] });
  events.state('repaired', 2); await tick(); Events.instances.at(-1)!.state('repaired', 3); await tick();
  expect(controller.getSnapshot()).toMatchObject({ sessionStatus: 'ready', selected: null });
  expect(controller.getSnapshot().toasts.some(t => t.message.includes('selected tensor'))).toBe(true);
});

it('honors model and explorer intent during replacement and deletes abandoned POST sessions', async () => {
  const { controller, client } = await setup();
  const pending = deferred<typeof sessionA>(); vi.mocked(client.createSession).mockReturnValueOnce(pending.promise);
  Events.instances[0]!.state('B');
  controller.switchExplorer('Tokenizer Explorer');
  const second = { ...sessionA, id: 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb', model_id: models[1]!.id };
  vi.mocked(client.createSession).mockResolvedValueOnce(second);
  controller.chooseModel(second.model_id); await tick();
  pending.resolve({ ...sessionA, id: 'cccccccc-cccc-4ccc-8ccc-cccccccccccc', model_revision: 'B' }); await tick();
  expect(controller.getSnapshot()).toMatchObject({ session: second, explorer: 'Tokenizer Explorer' });
  expect(client.deleteSession).toHaveBeenCalledWith('cccccccc-cccc-4ccc-8ccc-cccccccccccc');
  expect(client.deleteSession).not.toHaveBeenCalledWith(second.id);
});

it('bounds transient retries at 1/2/4 seconds and lets revision and close cancel timers', async () => {
  vi.useFakeTimers();
  const { controller, client } = await setup();
  const { ApiFailure } = await import('../api/errors');
  vi.mocked(client.createSession).mockRejectedValue(new ApiFailure('transport', 'private'));
  Events.instances[0]!.state('B'); await tick();
  for (const [delay, calls] of [[1000, 3], [2000, 4], [4000, 5]]) {
    await vi.advanceTimersByTimeAsync(delay! - 1); expect(client.createSession).toHaveBeenCalledTimes(calls! - 1);
    await vi.advanceTimersByTimeAsync(1); expect(client.createSession).toHaveBeenCalledTimes(calls!);
  }
  await vi.advanceTimersByTimeAsync(60_000); expect(client.createSession).toHaveBeenCalledTimes(5);
  Events.instances[0]!.state('C', 2); await tick(); expect(client.createSession).toHaveBeenCalledTimes(6);
  controller.closeSession(); await vi.advanceTimersByTimeAsync(60_000); expect(client.createSession).toHaveBeenCalledTimes(6);
});

it('does not refresh for capability/conflict errors or requests from disposed views', async () => {
  const { controller, client, fetcher } = await setup();
  const old = controller.explorerClient(controller.getSnapshot().view);
  controller.switchExplorer('Tokenizer Explorer');
  fetcher.mockResolvedValue(json({ code: 'model_content_changed', message: 'private' }, 409));
  await expect(old.tokenize(sessionA.id, { text: '', add_special_tokens: true })).rejects.toThrow();
  fetcher.mockResolvedValue(json({ code: 'unsupported_representation', message: 'private' }, 409));
  await expect(controller.explorerClient(controller.getSnapshot().view).tokenize(sessionA.id, { text: '', add_special_tokens: true })).rejects.toThrow();
  expect(client.createSession).toHaveBeenCalledTimes(1);
});

it('separates malformed and observation errors from unavailability with a retryable subscription', async () => {
  const { controller, client } = await setup(); const events = Events.instances[0]!;
  const epoch = '11111111-1111-4111-8111-111111111111';
  events.dispatchEvent(new MessageEvent('observation-error', { lastEventId: `${epoch}:1`, data: JSON.stringify({ epoch, sequence: 1, model_id: sessionA.model_id, code: 'observation_failed' }) }));
  expect(controller.getSnapshot()).toMatchObject({ session: sessionA, observation: 'observation-error' });
  events.dispatchEvent(new MessageEvent('model-state', { data: '{private invalid' }));
  expect(events.closed).toBe(true);
  expect(controller.getSnapshot()).toMatchObject({ session: sessionA, observation: 'protocol-error' });
  expect(client.createSession).toHaveBeenCalledTimes(1);
  controller.retryObservation(); Events.instances.at(-1)!.state('snapshot_A');
  expect(controller.getSnapshot().observation).toBe('live');
});

it.each([
  { sequence: -1 }, { sequence: Number.MAX_SAFE_INTEGER + 1 }, { epoch: 'private' },
  { status: 'present', model_revision: null }, { extra: '/private' }, { model_revision: 'x'.repeat(65_537) },
].map(patch => ({ patch, field: Object.keys(patch)[0] })))('rejects invalid event field $field safely', async ({ patch }) => {
  const { controller, client } = await setup();
  const value = { epoch: '11111111-1111-4111-8111-111111111111', sequence: 1, model_id: sessionA.model_id, status: 'present', model_revision: 'B', ...patch };
  Events.instances[0]!.dispatchEvent(new MessageEvent('model-state', { data: JSON.stringify(value), lastEventId: `${value.epoch}:${value.sequence}` }));
  expect(controller.getSnapshot()).toMatchObject({ session: sessionA, observation: 'protocol-error' });
  expect(client.createSession).toHaveBeenCalledTimes(1);
});

it('retains only presentation primitives through one accepted attempt and clears them on navigation', async () => {
  const { controller, client } = await setup();
  controller.switchExplorer('Architecture Explorer');
  const bookmark = { expanded: ['semantic:block'], focus: null, selection: 'semantic:weight', camera: { x: 1, y: 2, zoom: 3 } };
  controller.presentation.register(controller.getSnapshot().view, () => bookmark);
  vi.mocked(client.createSession).mockResolvedValueOnce({ ...sessionA, id: 'cccccccc-cccc-4ccc-8ccc-cccccccccccc', model_revision: 'B' });
  Events.instances[0]!.state('B'); await tick(); Events.instances.at(-1)!.state('B', 2); await tick();
  const key = { backend: controller.backend, modelId: sessionA.model_id, attempt: controller.getSnapshot().refreshAttempt };
  expect(controller.presentation.consume({ ...key, modelId: models[1]!.id })).toBeUndefined();
  expect(controller.presentation.consume(key)).toEqual(bookmark);
  expect(controller.presentation.consume(key)).toBeUndefined();
  controller.presentation.register(controller.getSnapshot().view, () => bookmark);
  Events.instances.at(-1)!.state('C', 3);
  controller.switchExplorer('Tensor Explorer');
  expect(controller.presentation.consume({ ...key, attempt: key.attempt + 1 })).toBeUndefined();
});

it('routes a supported LMEX terminal content error through the same coordinator', async () => {
  const { controller, client, fetcher } = await setup();
  const pending = deferred<typeof sessionA>(); vi.mocked(client.createSession).mockReturnValueOnce(pending.promise);
  const payload = new TextEncoder().encode(JSON.stringify({ code: 'model_content_changed', message: 'private' }));
  const bytes = new Uint8Array(12 + payload.length); bytes.set([76, 77, 69, 88, 5]);
  new DataView(bytes.buffer).setUint32(8, payload.length, true); bytes.set(payload, 12);
  fetcher.mockResolvedValueOnce(new Response(bytes, { headers: { 'Content-Type': 'application/vnd.llm-model-explorer.stream', 'X-Operation-Id': sessionA.id } }));
  const active = controller.explorerClient(controller.getSnapshot().view);
  expect((await active.streamTensor(sessionA.id, tensors[0]!.id).done).kind).toBe('backend');
  expect(client.createSession).toHaveBeenCalledTimes(2);
  expect(controller.getSnapshot().refreshStatus).toBe('updating');
  controller.closeSession(); pending.resolve(sessionA); await tick();
});

it('keeps stable admission failures waiting and bounds failed retired-session cleanup', async () => {
  const { controller, client } = await setup();
  const { ApiFailure } = await import('../api/errors');
  vi.mocked(client.createSession).mockRejectedValueOnce(new ApiFailure('http', 'private', 422, { code: 'unsupported_representation', message: 'private' }));
  vi.mocked(client.deleteSession).mockRejectedValue(new ApiFailure('transport', 'private'));
  Events.instances[0]!.state('B'); await tick(); Events.instances[0]!.state('B', 2); await tick();
  expect(client.createSession).toHaveBeenCalledTimes(2);
  expect(client.deleteSession).toHaveBeenCalledTimes(2);
  expect(controller.getSnapshot()).toMatchObject({ refreshStatus: 'waiting', selectedModelId: sessionA.model_id });
  expect(controller.getSnapshot().toasts.at(-1)?.message).toContain('Could not release an old session');
  controller.retrySession(); await tick(); expect(client.createSession).toHaveBeenCalledTimes(3);
});

it('resolves a changed public ID by unique full name and uses new unsupported geometry', async () => {
  const { controller, client } = await setup(); controller.selectTensor(tensors[0]!);
  const changed = { ...tensors[0]!, id: 'new-public-id', shape: [1, 2, 3], rank: 3 };
  vi.mocked(client.listTensors).mockResolvedValueOnce({ tensors: [tensors[1]!, changed], coverage: 'complete', diagnostics: [] });
  vi.mocked(client.createSession).mockResolvedValueOnce({ ...sessionA, id: 'cccccccc-cccc-4ccc-8ccc-cccccccccccc', model_revision: 'B' });
  Events.instances[0]!.state('B'); await tick(); Events.instances.at(-1)!.state('B', 2); await tick();
  expect(controller.getSnapshot().selected).toEqual(changed);
});

it('preserves configured backend prefixes and exact encoded logical subscription IDs', async () => {
  await setup();
  const client = new ApiClient({ backendBaseUrl: 'https://backend.example/prefix/' });
  const close = client.watchModel('lab/model@revision+adapter name', { state: vi.fn(), status: vi.fn() });
  expect(Events.instances.at(-1)!.url).toBe('https://backend.example/prefix/models/events?model_id=lab%2Fmodel%40revision%2Badapter%20name');
  close(); expect(Events.instances.at(-1)!.closed).toBe(true);
});
