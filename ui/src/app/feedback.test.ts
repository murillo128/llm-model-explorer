import { expect, it, vi } from 'vitest';
import { ApiClient } from '../api/client';
import { ApiFailure } from '../api/errors';
import { deferred, json, models } from '../test/shell-fixtures';
import { Feedback } from './feedback';
import type { FeedbackState } from './feedback';
import { Lifetime } from './lifetime';

function setup() {
  let state: FeedbackState = { connection: 'connecting', toasts: [] };
  const feedback = new Feedback(value => { state = value; });
  const lifetime = new Lifetime();
  return { feedback, lifetime, state: () => state };
}

it('reports actual attempts, deduplicates failures and never toasts routine connection events', async () => {
  const { feedback, lifetime, state } = setup();
  const transport = new ApiFailure('transport', 'private');
  expect(state().connection).toBe('connecting');
  await feedback.track(lifetime, () => Promise.reject(transport)).catch(() => {});
  expect(state()).toEqual({ connection: 'disconnected', toasts: [] });
  for (let attempt = 0; attempt < 2; attempt++) {
    const request = deferred<void>();
    const done = feedback.track(lifetime, () => request.promise).catch(() => {});
    expect(state().connection).toBe('reconnecting');
    request.reject(transport); await done;
    expect(state()).toEqual({ connection: 'disconnected', toasts: [] });
  }
  await feedback.track(lifetime, () => Promise.resolve());
  expect(state()).toEqual({ connection: 'connected', toasts: [] });
  const observer = feedback.observe(lifetime, true);
  const event = { id: Symbol(), phase: 'end' as const, failure: transport };
  observer(event); observer(event);
  expect(state().toasts).toHaveLength(1);
  feedback.dismiss(state().toasts[0]!.id);
  observer(event);
  expect(state().toasts).toHaveLength(0);
  expect(state().connection).toBe('disconnected');
});

it.each(['http', 'protocol', 'backend', 'cancelled'] as const)('%s failures do not mean disconnection or create operation toasts', async kind => {
  const { feedback, lifetime, state } = setup();
  await feedback.track(lifetime, () => Promise.resolve());
  await feedback.track(lifetime, () => Promise.reject(new ApiFailure(kind, 'private'))).catch(() => {});
  expect(state()).toEqual({ connection: 'connected', toasts: [] });
});

it('bounds the stack and fences old model, request and disposed backend feedback', () => {
  const { feedback, lifetime, state } = setup();
  const observe = feedback.observe(lifetime, true);
  for (let i = 0; i < 5; i++) observe({ id: Symbol(), phase: 'end', failure: new ApiFailure('transport', 'private') });
  expect(state().toasts).toHaveLength(3);
  feedback.reset();
  observe({ id: Symbol(), phase: 'end', failure: new ApiFailure('transport', 'late') });
  expect(state().toasts).toHaveLength(0);
  const current = feedback.observe(lifetime, true);
  current({ id: Symbol(), phase: 'start' });
  expect(state().connection).toBe('reconnecting');
  lifetime.dispose();
  expect(state().connection).toBe('disconnected');
  current({ id: Symbol(), phase: 'response' });
  expect(state().connection).toBe('disconnected');
});

it('observes real HTTP/protocol responses as reachable and fences an aborted JSON request even if fetch ignores abort', async () => {
  const { feedback, lifetime, state } = setup();
  const pending = deferred<Response>();
  const fetcher = vi.fn().mockResolvedValueOnce(json({ code: 'unsupported_representation', message: 'private' }, 422))
    .mockResolvedValueOnce(json({ invalid: true })).mockReturnValueOnce(pending.promise);
  const client = new ApiClient({ backendBaseUrl: 'https://backend.example' }, fetcher).observe(feedback.observe(lifetime, true));
  await expect(client.listModels()).rejects.toMatchObject({ kind: 'http' });
  expect(state()).toEqual({ connection: 'connected', toasts: [] });
  await expect(client.listModels()).rejects.toMatchObject({ kind: 'protocol' });
  expect(state()).toEqual({ connection: 'connected', toasts: [] });
  const abort = new AbortController();
  const done = client.listModels(abort.signal);
  abort.abort();
  pending.reject(new Error('late failure'));
  await expect(done).rejects.toMatchObject({ kind: 'cancelled' });
  expect(state()).toEqual({ connection: 'connected', toasts: [] });
});

it('does not mistake an existing stream for a recovery attempt; cancel cleanup failures are quiet', async () => {
  const { feedback, lifetime, state } = setup();
  const observer = feedback.observe(lifetime, true);
  observer({ id: Symbol(), phase: 'response' });
  observer({ id: Symbol(), phase: 'start' }); // another long-running operation
  observer({ id: Symbol(), phase: 'end', failure: new ApiFailure('transport', 'lost') });
  expect(state().connection).toBe('disconnected');
  feedback.reset();
  const client = new ApiClient({ backendBaseUrl: 'https://backend.example' }, vi.fn().mockRejectedValue(new Error('lost')))
    .observe(feedback.observe(lifetime, true));
  await expect(client.cancelOperation('id')).rejects.toMatchObject({ kind: 'transport' });
  expect(state().toasts).toHaveLength(0);
  const recovery = new ApiClient({ backendBaseUrl: 'https://backend.example' }, vi.fn().mockResolvedValue(json({ models, diagnostics: [] })))
    .observe(feedback.observe(lifetime));
  await recovery.listModels();
  expect(state().connection).toBe('connected');
});
