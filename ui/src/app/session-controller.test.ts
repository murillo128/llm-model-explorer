import { describe, expect, it, vi } from 'vitest';
import { ApiClient } from '../api/client';
import { ApiFailure } from '../api/errors';
import { deferred, memoryStorage, models, sessionA, sessionB, tensors } from '../test/shell-fixtures';
import { Lifetime, RequestChannel } from './lifetime';
import { SessionController, sessionStorageKey } from './session-controller';
import type { Session } from './session-controller';

const backend = 'https://backend.example';
function setup(storage = memoryStorage(), url = backend) {
  const client = new ApiClient({ backendBaseUrl: url }, vi.fn());
  vi.spyOn(client, 'listModels').mockResolvedValue({ models });
  vi.spyOn(client, 'listTensors').mockResolvedValue({ tensors });
  vi.spyOn(client, 'createSession').mockResolvedValue(sessionA);
  vi.spyOn(client, 'getSession').mockImplementation(async (id) => id === sessionB.id ? sessionB : sessionA);
  vi.spyOn(client, 'deleteSession').mockResolvedValue();
  const controller = new SessionController(client, url, storage);
  controller.start();
  return { client, controller, storage };
}
const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 0));
const expired = new ApiFailure('http', 'private error /srv/models', 404, { code: 'session_not_found', message: '/srv/models' });

it('creates immutable sessions, persists per backend, recovers, and explicitly deletes only the active session', async () => {
  const { controller, client, storage } = setup();
  await tick();
  controller.chooseModel(models[0]!.id);
  await tick();
  expect(controller.getSnapshot().session).toEqual(sessionA);
  expect(storage.getItem(sessionStorageKey(backend))).toBe(sessionA.id);
  vi.mocked(client.createSession).mockResolvedValue(sessionB);
  controller.chooseModel(models[1]!.id);
  await tick();
  expect(client.createSession).toHaveBeenLastCalledWith({ model_id: models[1]!.id });
  expect(client.deleteSession).not.toHaveBeenCalled();
  expect(controller.getSnapshot().session).toEqual(sessionB);
  controller.dispose();
  const recovered = setup(storage);
  await tick();
  expect(recovered.client.getSession).toHaveBeenCalledWith(sessionB.id, expect.any(AbortSignal));
  recovered.controller.closeSession();
  await tick();
  expect(recovered.client.deleteSession).toHaveBeenCalledTimes(1);
  recovered.controller.dispose();
});

it('distinguishes expiration from transport errors and permits fresh creation or retry', async () => {
  const { controller, client, storage } = setup();
  await tick();
  vi.mocked(client.getSession).mockRejectedValue(expired);
  controller.recover(sessionA.id);
  await tick();
  expect(controller.getSnapshot()).toMatchObject({ sessionStatus: 'expired-session', session: null });
  expect(controller.getSnapshot().message).not.toContain('/srv');
  expect(storage.getItem(sessionStorageKey(backend))).toBeNull();
  vi.mocked(client.createSession).mockRejectedValueOnce(new ApiFailure('transport', 'internal'));
  controller.chooseModel(models[0]!.id);
  await tick();
  expect(controller.getSnapshot().sessionStatus).toBe('failed');
  controller.retrySession();
  await tick();
  expect(controller.getSnapshot().sessionStatus).toBe('ready');
  vi.mocked(client.listTensors).mockRejectedValue(expired);
  controller.loadInventory();
  await tick();
  expect(controller.getSnapshot().sessionStatus).toBe('expired-session');
  controller.dispose();
});

it('fences late inventories across session changes even when abort is ignored', async () => {
  const { controller, client } = setup();
  const first = deferred<{ tensors: typeof tensors }>();
  vi.mocked(client.listTensors).mockReturnValueOnce(first.promise);
  await tick();
  controller.chooseModel(models[0]!.id);
  await tick();
  const oldSignal = vi.mocked(client.listTensors).mock.calls[0]![1];
  vi.mocked(client.createSession).mockResolvedValue(sessionB);
  vi.mocked(client.listTensors).mockResolvedValue({ tensors: [tensors[1]!] });
  controller.chooseModel(models[1]!.id);
  await tick();
  first.resolve({ tensors: [tensors[0]!] });
  await tick();
  expect(oldSignal?.aborted).toBe(true);
  expect(controller.getSnapshot().tensors).toEqual([tensors[1]]);
  expect(controller.getSnapshot().session).toEqual(sessionB);
  controller.dispose();
});

it('releases a superseded newly created session without touching the winning one', async () => {
  const { controller, client } = setup();
  const first = deferred<Session>();
  vi.mocked(client.createSession).mockReturnValueOnce(first.promise).mockResolvedValueOnce(sessionB);
  await tick();
  controller.chooseModel(models[0]!.id);
  controller.chooseModel(models[1]!.id);
  await tick();
  first.resolve(sessionA);
  await tick();
  expect(controller.getSnapshot().session).toEqual(sessionB);
  expect(client.deleteSession).toHaveBeenCalledExactlyOnceWith(sessionA.id);
  controller.dispose();
});

it('disposes A → B → A selections and tool changes, and accepts only current status callbacks', async () => {
  const { controller } = setup();
  await tick(); controller.chooseModel(models[0]!.id); await tick();
  controller.selectTensor(tensors[0]!);
  const firstA = controller.getSnapshot().view;
  const cancel = vi.fn();
  firstA.onDispose(cancel);
  controller.selectTensor(tensors[1]!);
  controller.selectTensor(tensors[0]!);
  controller.reportStatus(firstA, 'failed');
  expect(controller.getSnapshot().viewStatus).toBe('idle');
  expect(cancel).toHaveBeenCalledTimes(1);
  const nextA = controller.getSnapshot().view;
  for (const status of ['loading', 'streaming', 'complete', 'cancelled', 'failed'] as const) {
    controller.reportStatus(nextA, status);
    expect(controller.getSnapshot().viewStatus).toBe(status);
  }
  controller.switchExplorer('Tokenizer Explorer');
  expect(nextA.signal.aborted).toBe(true);
  expect(controller.getSnapshot().session).toEqual(sessionA);
  controller.reportStatus(nextA, 'expired-session');
  expect(controller.getSnapshot().sessionStatus).toBe('ready');
  controller.reportStatus(controller.getSnapshot().view, 'expired-session');
  expect(controller.getSnapshot().sessionStatus).toBe('expired-session');
  controller.dispose();
});

it('isolates tab and backend state, preserves other backend recovery records, and cancels only its own resources', async () => {
  const one = setup(); const two = setup();
  await tick();
  one.controller.chooseModel(models[0]!.id);
  vi.mocked(two.client.createSession).mockResolvedValue(sessionB);
  two.controller.chooseModel(models[1]!.id);
  await tick();
  const cancelOne = vi.fn(); const cancelTwo = vi.fn();
  one.controller.getSnapshot().view.onDispose(cancelOne);
  two.controller.getSnapshot().view.onDispose(cancelTwo);
  one.controller.closeSession(); await tick();
  expect(cancelOne).toHaveBeenCalledTimes(1);
  expect(cancelTwo).not.toHaveBeenCalled();
  expect(two.client.deleteSession).not.toHaveBeenCalled();
  expect(two.storage.getItem(sessionStorageKey(backend))).toBe(sessionB.id);
  const otherBackend = setup(two.storage, 'https://other.example'); await tick();
  expect(otherBackend.client.getSession).not.toHaveBeenCalled();
  otherBackend.controller.dispose(); one.controller.dispose(); two.controller.dispose();
});

it('retains the recoverable ID on recovery failure and leaves a failed deletion retryable', async () => {
  const { controller, client, storage } = setup(); await tick();
  storage.setItem(sessionStorageKey(backend), sessionA.id);
  vi.mocked(client.getSession).mockRejectedValueOnce(new ApiFailure('transport', 'offline'));
  controller.recover(); await tick();
  expect(storage.getItem(sessionStorageKey(backend))).toBe(sessionA.id);
  controller.retrySession(); await tick();
  vi.mocked(client.deleteSession).mockRejectedValueOnce(new ApiFailure('transport', 'offline'));
  controller.closeSession(); await tick();
  expect(controller.getSnapshot().sessionStatus).toBe('ready');
  expect(storage.getItem(sessionStorageKey(backend))).toBe(sessionA.id);
  controller.closeSession(); await tick();
  expect(controller.getSnapshot().session).toBeNull();
  expect(storage.getItem(sessionStorageKey(backend))).toBeNull();
  controller.dispose();
});

describe('request ownership', () => {
  it('fences repeated requests while independent channels remain active', () => {
    const selection = new Lifetime();
    const data = new RequestChannel(selection); const stats = new RequestChannel(selection);
    const requestA = data.begin(); const independent = stats.begin();
    const cancelA = vi.fn(); const cancelStats = vi.fn();
    requestA.onDispose(cancelA); independent.onDispose(cancelStats);
    const callback = vi.fn(); const oldCallback = requestA.guard(callback);
    const requestB = data.begin(); oldCallback('late');
    expect(callback).not.toHaveBeenCalled(); expect(cancelA).toHaveBeenCalledTimes(1);
    expect(independent.isCurrent()).toBe(true);
    expect(cancelStats).not.toHaveBeenCalled();
    requestB.guard(callback)('current'); expect(callback).toHaveBeenCalledWith('current');
    selection.dispose(); expect(cancelStats).toHaveBeenCalledTimes(1);
    expect(data.begin().isCurrent()).toBe(false);
    data.dispose(); stats.dispose();
  });
});

it('ignores late recovery and catalogue results after replacement', async () => {
  const { controller, client } = setup(); await tick();
  const recovery = deferred<Session>();
  vi.mocked(client.getSession).mockReturnValueOnce(recovery.promise);
  controller.recover(sessionA.id);
  vi.mocked(client.createSession).mockResolvedValue(sessionB);
  controller.chooseModel(models[1]!.id); await tick();
  recovery.resolve(sessionA); await tick();
  expect(controller.getSnapshot().session).toEqual(sessionB);
  const catalogue = deferred<{ models: typeof models }>();
  vi.mocked(client.listModels).mockReturnValueOnce(catalogue.promise).mockResolvedValueOnce({ models: [] });
  controller.loadModels(); controller.loadModels(); await tick();
  catalogue.resolve({ models }); await tick();
  expect(controller.getSnapshot().models).toEqual([]);
  controller.dispose();
});

it('remains usable when tab storage is denied', async () => {
  const denied = { getItem: () => { throw new Error('denied'); }, setItem: () => { throw new Error('denied'); }, removeItem: () => { throw new Error('denied'); } };
  const { controller } = setup(denied);
  await tick(); controller.chooseModel(models[0]!.id); await tick();
  expect(controller.getSnapshot()).toMatchObject({ storageAvailable: false, sessionStatus: 'ready', session: sessionA });
  controller.closeSession(); await tick();
  expect(controller.getSnapshot().session).toBeNull();
  controller.dispose();
});
