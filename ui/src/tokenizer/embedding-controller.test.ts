import { expect, it, vi } from 'vitest';
import type { StreamOptions } from '../api/client';
import type { StreamOutcome } from '../api/lmex-decoder';
import type { Metadata } from '../api/validation';
import type { MatrixUpdates } from '../matrix-explorer';
import { deferred } from '../test/shell-fixtures';
import { EmbeddingController } from './embedding-controller';
import type { EmbeddingState } from './embedding-controller';

const metadata: Metadata = { kind: 'input_embeddings', token_ids: [2, 0, 2], shape: [3, 2], dtype: 'float32', byte_order: 'little', layout: 'c', byte_length: 24 };
function harness() {
  const requests: { options: StreamOptions; done: ReturnType<typeof deferred<StreamOutcome>>; cancel: ReturnType<typeof vi.fn> }[] = [];
  const client = { streamInputEmbeddings: vi.fn((_session, _body, options: StreamOptions = {}) => {
    const done = deferred<StreamOutcome>(), cancel = vi.fn(async () => {});
    requests.push({ options, done, cancel });
    return { operationId: undefined, done: done.promise, cancel };
  }) };
  return { client, requests };
}
it('streams exact split words with one allocation, restarts at zero on remount and fences retained callbacks', async () => {
  const { client, requests } = harness();
  const signal = new AbortController();
  const states: EmbeddingState[] = [];
  const updates: MatrixUpdates = { values: vi.fn(), transfer: vi.fn(), distribution: vi.fn(), distributionDomain: vi.fn() };
  let detach!: () => void;
  const controller = new EmbeddingController(client, 's', [2, 0, 2], signal.signal, (state, allocate) => {
    states.push(state);
    if (allocate) detach = state.source!.subscribe(updates);
  });
  controller.start();
  requests[0]!.options.onMetadata!(metadata);
  const bytes = new Uint8Array(new Float32Array([-7.5, 8, 1, -2, -7.5, 8]).buffer);
  requests[0]!.options.onData!(bytes.subarray(0, 3), 0);
  expect(updates.values).not.toHaveBeenCalled();
  requests[0]!.options.onData!(bytes.subarray(3), 3);
  expect(updates.values).toHaveBeenCalledExactlyOnceWith(new Float32Array([-7.5, 8, 1, -2, -7.5, 8]), 0);
  const source = states.at(-1)!.source!;
  detach(); const secondDetach = source.subscribe(updates);
  expect(requests[0]!.cancel).toHaveBeenCalled();
  requests[0]!.options.onData!(bytes, 24); // Ignore even invalid stale offsets.
  requests[0]!.done.resolve({ kind: 'backend', error: { code: 'internal_error', message: 'Old error' } });
  requests[1]!.options.onMetadata!(metadata);
  requests[1]!.options.onData!(bytes, 0);
  expect(updates.values).toHaveBeenCalledTimes(2);
  signal.abort();
  requests[1]!.options.onData!(bytes, 24);
  requests[1]!.done.resolve({ kind: 'complete', metadata, byteLength: 24 });
  await Promise.resolve();
  expect(updates.values).toHaveBeenCalledTimes(2);
  expect(states.at(-1)!.status).toBe('streaming');
  expect(requests[1]!.cancel).toHaveBeenCalled();
  secondDetach(); controller.dispose();
});
it.each(['unsupported_representation', 'internal_error'] as const)('removes partial data on terminal %s', async code => {
  const { client, requests } = harness();
  const changed = vi.fn();
  const controller = new EmbeddingController(client, 's', [2, 0, 2], new AbortController().signal, changed);
  controller.start();
  requests[0]!.options.onMetadata!(metadata);
  requests[0]!.done.resolve({ kind: 'backend', error: { code, message: 'Fixture' } });
  await Promise.resolve();
  expect(changed).toHaveBeenLastCalledWith({ status: code === 'unsupported_representation' ? 'unsupported' : 'failed' });
  controller.dispose();
});
