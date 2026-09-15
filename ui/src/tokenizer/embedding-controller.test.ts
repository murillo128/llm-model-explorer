import { expect, it, vi } from 'vitest';
import { ApiFailure } from '../api/errors';
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
  const auxiliary: typeof requests = [];
  const analysis = vi.fn((_session, _body, options: StreamOptions = {}) => {
    const done = deferred<StreamOutcome>(), cancel = vi.fn(async () => {});
    auxiliary.push({ options, done, cancel });
    return { operationId: undefined, done: done.promise, cancel };
  });
  const client = { streamInputEmbeddingsStatistics: analysis, streamInputEmbeddingsDistributions: analysis, streamInputEmbeddings: vi.fn((_session, _body, options: StreamOptions = {}) => {
    const done = deferred<StreamOutcome>(), cancel = vi.fn(async () => {});
    requests.push({ options, done, cancel });
    return { operationId: undefined, done: done.promise, cancel };
  }) };
  return { client, requests, auxiliary };
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
  expect(changed.mock.lastCall![0]).toMatchObject({ status: code === 'unsupported_representation' ? 'unsupported' : 'failed', source: undefined });
  controller.dispose();
});


it('fences completion and cancels transport when a replacement renderer fails', async () => {
  const { client, requests } = harness();
  const changed = vi.fn();
  const controller = new EmbeddingController(client, 's', [2, 0, 2], new AbortController().signal, changed);
  controller.start();
  requests[0]!.options.onMetadata!(metadata);
  controller.renderingFailed();
  expect(changed.mock.lastCall![0]).toMatchObject({ status: 'failed', source: undefined });
  expect(requests[0]!.cancel).toHaveBeenCalled();
  requests[0]!.options.onData!(new Uint8Array(24), 0);
  requests[0]!.done.resolve({ kind: 'complete', metadata, byteLength: 24 });
  await Promise.resolve();
  expect(changed.mock.lastCall![0]).toMatchObject({ status: 'failed', source: undefined });
  controller.dispose();
});

it.each([
  ['http', 'unsupported_representation', 'unsupported'],
  ['http', 'internal_error', 'failed'],
  ['protocol', undefined, 'failed'],
  ['transport', undefined, 'failed'],
] as const)('classifies %s/%s without hiding failures as unsupported', async (kind, code, status) => {
  const { client, requests } = harness();
  const changed = vi.fn();
  const controller = new EmbeddingController(client, 's', [2, 0, 2], new AbortController().signal, changed);
  controller.start();
  requests[0]!.done.reject(new ApiFailure(kind, 'Fixture failure', code === 'unsupported_representation' ? 422 : 500,
    code ? { code, message: 'Fixture' } : undefined));
  await Promise.resolve(); await Promise.resolve();
  expect(changed.mock.lastCall![0]).toMatchObject({ status, source: undefined });
  controller.dispose();
});

const statistics: Extract<Metadata, { kind: 'input_embeddings_statistics' }> = {
  kind: 'input_embeddings_statistics', token_ids: [2, 0, 2], shape: [3, 2], count: 6,
  finite_count: 6, non_finite_count: 0, minimum: -7.5, maximum: 8, mean: 0, stddev: Math.sqrt(245.5 / 6),
  percentiles: { p01: -7.5, p05: -7.5, p50: -0.5, p95: 8, p99: 8 }, byte_length: 0,
};
const distributions: Extract<Metadata, { kind: 'input_embeddings_distributions' }> = {
  kind: 'input_embeddings_distributions', token_ids: [2, 0, 2], rows: 3, columns: 2,
  bin_count: 100, binning: 'linear-full-range', domain_minimum: -7.5, domain_maximum: 8,
  dtype: 'uint32', byte_order: 'little', byte_length: 2000,
  sections: [{ name: 'row_counts', shape: [3, 100], offset: 0, byte_length: 1200 },
    { name: 'column_counts', shape: [100, 2], offset: 1200, byte_length: 800 }],
};
// Hand-authored bins for rows [-7.5, 8], [1, -2], [-7.5, 8].
const histogram = new Uint32Array(500);
for (const index of [0, 99, 135, 154, 200, 299]) histogram[index] = 1;
histogram[300] = 2; histogram[300 + 35 * 2 + 1] = 1;
histogram[300 + 54 * 2] = 1; histogram[300 + 99 * 2 + 1] = 2;
const scalarBytes = new Uint8Array(new Float32Array([-7.5, 8, 1, -2, -7.5, 8]).buffer);
const countBytes = new Uint8Array(histogram.buffer);
const tick = async () => { await Promise.resolve(); await Promise.resolve(); };

it.each([false, true])('flushes the final values before promotion and respects presentation failure=%s', async fail => {
  const { client, requests } = harness();
  const events: string[] = [];
  const updates: MatrixUpdates = { values: vi.fn(), transfer: vi.fn(), distribution: vi.fn(), distributionDomain: vi.fn(),
    flush: () => { events.push('present'); if (fail) controller.renderingFailed(); } };
  const controller = new EmbeddingController(client, 's', [2, 0, 2], new AbortController().signal, (state, allocate) => {
    events.push(state.status);
    if (allocate) state.source!.subscribe(updates);
  });
  controller.start(); requests[0]!.options.onMetadata!(metadata);
  requests[0]!.options.onData!(scalarBytes, 0);
  requests[0]!.done.resolve({ kind: 'complete', metadata, byteLength: 24 });
  await tick();
  expect(events.slice(-2)).toEqual(['present', fail ? 'failed' : 'complete']);
  expect(updates.values).toHaveBeenCalledExactlyOnceWith(new Float32Array([-7.5, 8, 1, -2, -7.5, 8]), 0);
  controller.dispose();
});

it('uploads every progressive chunk without republishing unchanged result status', async () => {
  const { client, requests, auxiliary } = harness();
  const updates: MatrixUpdates = { values: vi.fn(), transfer: vi.fn(), distribution: vi.fn(), distributionDomain: vi.fn() };
  const changed = vi.fn((state: EmbeddingState, allocate?: boolean) => {
    if (allocate) state.source!.subscribe(updates);
  });
  const controller = new EmbeddingController(client, 's', [2, 0, 2], new AbortController().signal, changed);
  controller.start(); requests[0]!.options.onMetadata!(metadata);
  const allocated = changed.mock.calls.length;
  for (let offset = 0; offset < scalarBytes.length; offset += 4) {
    requests[0]!.options.onData!(scalarBytes.subarray(offset, offset + 4), offset);
    expect(updates.values).toHaveBeenLastCalledWith(new Float32Array(scalarBytes.slice(offset, offset + 4).buffer), offset / 4);
  }
  expect(updates.values).toHaveBeenCalledTimes(6);
  expect(changed).toHaveBeenCalledTimes(allocated);
  auxiliary[1]!.options.onMetadata!(distributions);
  for (let offset = 0; offset < countBytes.length; offset += 100) {
    auxiliary[1]!.options.onData!(countBytes.subarray(offset, offset + 100), offset);
  }
  expect(updates.distribution).toHaveBeenCalledTimes(20);
  expect(changed).toHaveBeenCalledTimes(allocated + 1);
  requests[0]!.done.resolve({ kind: 'complete', metadata, byteLength: 24 });
  auxiliary[0]!.done.resolve({ kind: 'complete', metadata: statistics, byteLength: 0 });
  auxiliary[1]!.done.resolve({ kind: 'complete', metadata: distributions, byteLength: 2000 });
  await tick();
  expect(changed.mock.lastCall![0]).toMatchObject({ status: 'complete', statistics: 'complete', distributions: 'complete', statisticsMetadata: statistics });
  expect(changed).toHaveBeenCalledTimes(allocated + 4);
  controller.dispose();
});

it.each(['vsd', 'vds', 'svd', 'sdv', 'dvs', 'dsv'])('keeps values progressive and auxiliary identity exact in %s completion order', async order => {
  const { client, requests, auxiliary } = harness();
  const states: EmbeddingState[] = [];
  const uploads: { axis: string; data: number[]; offset: number }[] = [];
  const updates: MatrixUpdates = { values: vi.fn(), transfer: vi.fn(), distributionDomain: vi.fn(),
    distribution: (axis, counts, offset) => uploads.push({ axis, data: [...counts], offset }) };
  const controller = new EmbeddingController(client, 's', [2, 0, 2], new AbortController().signal, (state, allocate) => {
    states.push(state);
    if (allocate) {
      expect(state.source!.distributions).toBe(true);
      state.source!.subscribe(updates);
    }
  });
  controller.start();
  expect(client.streamInputEmbeddings).toHaveBeenCalledOnce();
  expect(auxiliary).toHaveLength(0);
  requests[0]!.options.onMetadata!(metadata);
  expect(auxiliary).toHaveLength(2);
  for (const step of order) {
    if (step === 'v') {
      requests[0]!.options.onData!(scalarBytes, 0);
      expect(updates.values).toHaveBeenCalledExactlyOnceWith(new Float32Array([-7.5, 8, 1, -2, -7.5, 8]), 0);
      expect(states.at(-1)!.status).toBe('streaming');
      requests[0]!.done.resolve({ kind: 'complete', metadata, byteLength: 24 });
    } else if (step === 's') {
      auxiliary[0]!.options.onMetadata!(statistics);
      expect(updates.transfer).not.toHaveBeenCalled(); // META alone is not successful statistics.
      auxiliary[0]!.done.resolve({ kind: 'complete', metadata: statistics, byteLength: 0 });
    } else {
      auxiliary[1]!.options.onMetadata!(distributions);
      // Network delivery crosses both word and row/column section boundaries.
      auxiliary[1]!.options.onData!(countBytes.subarray(0, 1199), 0);
      auxiliary[1]!.options.onData!(countBytes.subarray(1199), 1199);
      auxiliary[1]!.done.resolve({ kind: 'complete', metadata: distributions, byteLength: 2000 });
    }
    await tick();
    if (step === 'v') expect(states.at(-1)!.status).toBe('complete');
  }
  expect(states.at(-1)).toMatchObject({ status: 'complete', statistics: 'complete', distributions: 'complete', statisticsMetadata: statistics });
  expect(updates.transfer).toHaveBeenCalledWith({ statistics });
  expect(updates.distributionDomain).toHaveBeenCalledWith({ minimum: -7.5, maximum: 8 });
  for (const [axis, expected] of [['rows', histogram.slice(0, 300)], ['columns', histogram.slice(300)]] as const) {
    const actual = new Uint32Array(expected.length);
    for (const upload of uploads.filter(upload => upload.axis === axis)) actual.set(upload.data, upload.offset);
    expect(actual).toEqual(expected);
  }
  const last = states.at(-1)!;
  const transferred = vi.mocked(updates.transfer).mock.calls.length;
  controller.dispose();
  auxiliary[1]!.options.onData!(countBytes, 99999);
  auxiliary[0]!.options.onMetadata!({ ...statistics, token_ids: [0, 2, 0] });
  expect(states.at(-1)).toBe(last);
  expect(updates.transfer).toHaveBeenCalledTimes(transferred);
});

it.each(['statistics', 'distributions'] as const)('a failed or cancelled %s operation preserves successful values and other consumers', async result => {
  const { client, requests, auxiliary } = harness();
  const changed = vi.fn();
  const controller = new EmbeddingController(client, 's', [2, 0, 2], new AbortController().signal, changed);
  controller.start(); requests[0]!.options.onMetadata!(metadata);
  const source = changed.mock.lastCall![0].source;
  source.subscribe({ values: vi.fn(), transfer: vi.fn(), distribution: vi.fn(), distributionDomain: vi.fn() });
  requests[0]!.options.onData!(scalarBytes, 0);
  requests[0]!.done.resolve({ kind: 'complete', metadata, byteLength: 24 });
  await tick();
  const index = result === 'statistics' ? 0 : 1;
  if (index === 0) auxiliary[index]!.done.reject(new ApiFailure('protocol', 'Invalid auxiliary metadata'));
  else controller.cancel(result);
  await tick();
  expect(changed.mock.lastCall![0]).toMatchObject({ source, status: 'complete', [result]: index === 0 ? 'failed' : 'cancelled' });
  expect(requests[0]!.cancel).not.toHaveBeenCalled();
  expect(auxiliary[1 - index]!.cancel).not.toHaveBeenCalled();
  controller.dispose();
});

it('fences equal-shaped early auxiliary results and releases staging on abort and source replacement', async () => {
  const { client, requests, auxiliary } = harness();
  const updates: MatrixUpdates = { values: vi.fn(), transfer: vi.fn(), distribution: vi.fn(), distributionDomain: vi.fn() };
  const oldSignal = new AbortController();
  const oldChanged = vi.fn();
  const old = new EmbeddingController(client, 'old-session', [2, 0, 2], oldSignal.signal, oldChanged);
  old.start();
  requests[0]!.options.onMetadata!(metadata); // No renderer has subscribed yet.
  auxiliary[1]!.options.onMetadata!(distributions);
  auxiliary[1]!.options.onData!(countBytes, 0);
  auxiliary[0]!.done.resolve({ kind: 'complete', metadata: statistics, byteLength: 0 });
  await tick(); oldSignal.abort();
  const next = new EmbeddingController(client, 'new-session', [0, 2, 0], new AbortController().signal, (state, allocate) => {
    if (allocate) state.source!.subscribe(updates);
  });
  next.start();
  requests[1]!.options.onMetadata!({ ...metadata, token_ids: [0, 2, 0] });
  requests[0]!.options.onMetadata!(metadata);
  auxiliary[1]!.done.resolve({ kind: 'complete', metadata: distributions, byteLength: 2000 });
  auxiliary[1]!.options.onData!(countBytes, 0);
  await tick();
  expect(updates.distribution).not.toHaveBeenCalled();
  expect(updates.distributionDomain).not.toHaveBeenCalled();
  expect(updates.transfer).not.toHaveBeenCalled();
  expect(auxiliary.slice(0, 2).every(request => request.cancel.mock.calls.length > 0)).toBe(true);
  expect(() => auxiliary[2]!.options.onMetadata!(statistics)).toThrow('Embedding ordered identity mismatch');
  expect(() => auxiliary[3]!.options.onMetadata!(distributions)).toThrow('Embedding ordered identity mismatch');
  next.dispose();
});

it('rejects inconsistent auxiliary shapes before uploading them, while values remain usable', async () => {
  const { client, requests, auxiliary } = harness();
  const changed = vi.fn();
  const controller = new EmbeddingController(client, 's', [2, 0, 2], new AbortController().signal, changed);
  controller.start(); requests[0]!.options.onMetadata!(metadata);
  const updates: MatrixUpdates = { values: vi.fn(), transfer: vi.fn(), distribution: vi.fn(), distributionDomain: vi.fn() };
  changed.mock.lastCall![0].source.subscribe(updates);
  for (const [index, wrong] of [[0, { ...statistics, shape: [3, 3] }], [1, { ...distributions, columns: 3 }]] as const) {
    // Simulate the typed transport propagating a rejected metadata callback.
    try { auxiliary[index]!.options.onMetadata!(wrong as Metadata); }
    catch (error) { auxiliary[index]!.done.reject(error); }
  }
  await tick();
  expect(changed.mock.lastCall![0]).toMatchObject({ status: 'streaming', statistics: 'failed', distributions: 'failed' });
  expect(updates.distributionDomain).not.toHaveBeenCalled();
  expect(updates.transfer).not.toHaveBeenCalled();
  requests[0]!.options.onData!(scalarBytes, 0);
  expect(updates.values).toHaveBeenCalledOnce();
  controller.dispose();
});

it('does not request auxiliary operations for an unavailable input table and can cancel before META', async () => {
  const { client, requests, auxiliary } = harness();
  const changed = vi.fn();
  const unavailable = new EmbeddingController(client, 's', [2, 0, 2], new AbortController().signal, changed);
  unavailable.start();
  requests[0]!.done.resolve({ kind: 'backend', error: { code: 'unsupported_representation', message: 'No input table' } });
  await tick();
  expect(auxiliary).toHaveLength(0);
  expect(changed.mock.lastCall![0].status).toBe('unsupported');
  unavailable.dispose();
  const cancelled = new EmbeddingController(client, 's', [2, 0, 2], new AbortController().signal, changed);
  cancelled.start(); cancelled.cancel('values');
  expect(requests[1]!.cancel).toHaveBeenCalled();
  expect(auxiliary).toHaveLength(0);
  expect(changed.mock.lastCall![0].status).toBe('cancelled');
  cancelled.dispose();
});

it('stages counts and completed statistics only until the matching renderer subscribes', async () => {
  const { client, requests, auxiliary } = harness();
  const changed = vi.fn();
  const controller = new EmbeddingController(client, 's', [2, 0, 2], new AbortController().signal, changed);
  controller.start(); requests[0]!.options.onMetadata!(metadata);
  auxiliary[1]!.options.onMetadata!(distributions);
  auxiliary[1]!.options.onData!(countBytes, 0);
  auxiliary[0]!.done.resolve({ kind: 'complete', metadata: statistics, byteLength: 0 });
  await tick();
  const updates: MatrixUpdates = { values: vi.fn(), transfer: vi.fn(), distribution: vi.fn(), distributionDomain: vi.fn() };
  changed.mock.lastCall![0].source.subscribe(updates);
  expect(updates.values).not.toHaveBeenCalled();
  expect(updates.transfer).toHaveBeenCalledWith({ statistics });
  expect(updates.distribution).toHaveBeenCalledWith('rows', histogram.subarray(0, 300), 0);
  expect(updates.distribution).toHaveBeenCalledWith('columns', histogram.subarray(300), 0);
  requests[0]!.options.onData!(scalarBytes, 0);
  expect(updates.values).toHaveBeenCalledOnce();
  controller.dispose();
});
