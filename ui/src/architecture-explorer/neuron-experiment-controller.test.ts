// @vitest-environment node
import { expect, it, vi } from 'vitest';
import fixtures from '../../../api/fixtures/neurons.json';
import { ApiClient } from '../api/client';
import type { StreamOptions } from '../api/client';
import type { StreamOutcome } from '../api/lmex-decoder';
import { validateSchema } from '../api/validation';
import { deferred } from '../test/shell-fixtures';
import { NeuronExperimentController } from './neuron-experiment-controller';
import type { ExperimentSnapshot } from './neuron-experiment';

function setup() {
  const client = new ApiClient({ backendBaseUrl: 'https://fixture.example' });
  const metadata = validateSchema('NeuronActivationsMetadata', fixtures.bases.activations);
  const snapshot: ExperimentSnapshot = { session: { id: metadata.session_id, model_id: metadata.model_id, model_revision: metadata.model_revision },
    capabilities: validateSchema('NeuronCapabilities', fixtures.bases.capabilities), mode: 'activations',
    request: validateSchema('NeuronActivationsRequest', fixtures.bases.observation), input: { text: 'x x', add_special_tokens: false, tokens: [] } };
  const pending = deferred<StreamOutcome>();
  let callbacks: StreamOptions = {};
  const cancel = vi.fn(async () => {});
  const stream = vi.spyOn(client, 'streamNeuronActivations').mockImplementation((_session, _cap, _body, options) => {
    callbacks = options!; return { done: pending.promise, cancel, operationId: undefined };
  });
  const token = new AbortController(), owner = new AbortController();
  const controller = new NeuronExperimentController(client);
  const bytes = Uint8Array.from(Buffer.from('000080bf0000004000000000000040c0', 'hex'));
  const deliver = () => { callbacks.onOperationId?.('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'); callbacks.onMetadata?.(metadata); callbacks.onData?.(bytes.subarray(0, 3), 0); callbacks.onData?.(bytes.subarray(3), 3); };
  const complete = () => pending.resolve({ kind: 'complete', metadata, byteLength: 16 });
  return { controller, snapshot, token, owner, pending, cancel, stream, deliver, complete };
}
it('prevents double runs, accepts split float32 words only after complete EOF, and preserves the snapshot', async () => {
  const f = setup(); f.controller.run(f.snapshot, f.token.signal, f.owner.signal);
  f.controller.run(f.snapshot, f.token.signal, f.owner.signal); expect(f.stream).toHaveBeenCalledOnce();
  f.snapshot.input.text = 'changed after run';
  f.deliver(); expect(f.controller.getSnapshot().status).toBe('running'); expect(f.controller.getSnapshot().result).toBeUndefined();
  f.complete(); await vi.waitFor(() => expect(f.controller.getSnapshot().status).toBe('complete'));
  expect([...f.controller.getSnapshot().result!.values]).toEqual([-1, 2, 0, -3]);
  expect(f.controller.getSnapshot().result!.snapshot.input.text).toBe('x x');
  f.controller.run(f.snapshot, f.token.signal, f.owner.signal); expect(f.stream).toHaveBeenCalledTimes(2);
  f.controller.dispose();
});
it.each(['cancel', 'prompt', 'session'] as const)('rejects late completion after %s and releases only its consumer', async reason => {
  const f = setup(); f.controller.run(f.snapshot, f.token.signal, f.owner.signal);
  f.deliver();
  if (reason === 'cancel') await f.controller.cancel();
  else if (reason === 'prompt') f.token.abort(); else f.owner.abort();
  f.deliver(); f.complete();
  await vi.waitFor(() => expect(f.controller.getSnapshot().status).toBe('cancelled'));
  expect(f.cancel).toHaveBeenCalledOnce(); expect(f.controller.getSnapshot().result).toBeUndefined();
});
it('does not promote an incomplete body or expose arbitrary server/path messages', async () => {
  const f = setup(); f.controller.run(f.snapshot, f.token.signal, f.owner.signal); f.deliver();
  f.pending.reject(new Error('/private/model: incomplete bytes'));
  await vi.waitFor(() => expect(f.controller.getSnapshot().status).toBe('failed'));
  expect(f.controller.getSnapshot().result).toBeUndefined();
  expect(f.controller.getSnapshot().message).toContain('incomplete'); expect(f.controller.getSnapshot().message).not.toContain('/private');
});
