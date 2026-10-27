// @vitest-environment node
import { expect, it, vi } from 'vitest';
import { ApiClient } from './client';
import { neuronCapabilities, neuronSession } from '../test/neuron-fixture';
import fixtures from '../../../api/fixtures/neurons.json';
import { validateSchema } from './validation';
import { validateNeuronAssociation } from './neuron-validation';

const json = (value: unknown) => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } });
it('retrieves bounded validated capabilities for the exact session with no-store and cancellation', async () => {
  const fetcher = vi.fn(async () => json(neuronCapabilities));
  const client = new ApiClient({ backendBaseUrl: 'https://backend.example' }, fetcher);
  const signal = new AbortController().signal;
  expect(await client.getNeuronCapabilities(neuronSession, signal)).toEqual(neuronCapabilities);
  expect(fetcher).toHaveBeenCalledWith(`https://backend.example/sessions/${neuronSession.id}/neuron-capabilities`, expect.objectContaining({ cache: 'no-store', signal }));
});
it.each(['session_id', 'model_id', 'model_revision'] as const)('rejects a response with another %s', async field => {
  const value = { ...neuronCapabilities, [field]: field === 'session_id' ? 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa' : 'other' };
  const client = new ApiClient({ backendBaseUrl: 'https://backend.example' }, async () => json(value));
  await expect(client.getNeuronCapabilities(neuronSession)).rejects.toThrow('identity mismatch');
});
it('rejects control JSON beyond one MiB before parsing it', async () => {
  const client = new ApiClient({ backendBaseUrl: 'https://backend.example' }, async () => json({ padding: 'x'.repeat(1_048_576) }));
  await expect(client.getNeuronCapabilities(neuronSession)).rejects.toThrow('1 MiB');
});

const capabilities = validateSchema('NeuronCapabilities', fixtures.bases.capabilities);
const observation = validateSchema('NeuronActivationsRequest', fixtures.bases.observation);
const comparison = validateSchema('NeuronComparisonRequest', fixtures.bases.comparison);
const metadata = validateSchema('NeuronComparisonMetadata', fixtures.bases.probabilities);
const wire = (name: string) => Uint8Array.from(Buffer.from(fixtures.wire_cases.find(c => c.name === name)!.wire_hex, 'hex'));
const operationId = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const headers = { 'Content-Type': 'application/vnd.llm-model-explorer.stream', 'X-Operation-Id': operationId };

it.each(['activations', 'comparison'] as const)('posts exact %s IDs and freezes association before callers can mutate inputs', async kind => {
  const fetcher = vi.fn(async () => new Response(wire(kind === 'activations' ? 'neuron-activations' : 'neuron-probabilities'), { headers }));
  const client = new ApiClient({ backendBaseUrl: 'https://backend.example' }, fetcher);
  const body = structuredClone(kind === 'activations' ? observation : comparison);
  const cap = structuredClone(capabilities);
  const operation = kind === 'activations' ? client.streamNeuronActivations(neuronSession, cap, body)
    : client.streamNeuronComparison(neuronSession, cap, body as typeof comparison);
  body.token_ids[0] = 3; cap.modules[0]!.layer_index = 0;
  expect((await operation.done).kind).toBe('complete');
  const [url, init] = fetcher.mock.calls[0]! as unknown as [string, RequestInit];
  expect(url).toBe(`https://backend.example/sessions/${neuronSession.id}/neuron-${kind === 'activations' ? 'activations' : 'comparisons'}`);
  expect(JSON.parse(init.body as string)).toEqual(kind === 'activations' ? observation : comparison);
});

it.each(fixtures.association_cases)('checks contract originator association: $field / $valid', entry => {
  const value = { ...metadata, [entry.field]: entry.value };
  if (entry.valid) expect(() => validateNeuronAssociation(value, neuronSession, capabilities, comparison)).not.toThrow();
  else expect(() => validateNeuronAssociation(value, neuronSession, capabilities, comparison)).toThrow();
});

it('rejects a valid stream for reordered neurons before delivering metadata or values', async () => {
  const client = new ApiClient({ backendBaseUrl: 'https://backend.example' }, async () => new Response(wire('neuron-activations'), { headers }));
  const onMetadata = vi.fn(), onData = vi.fn();
  await expect(client.streamNeuronActivations(neuronSession, capabilities, { ...observation, neuron_indices: [0, 2] }, { onMetadata, onData }).done).rejects.toThrow('identity mismatch');
  expect(onMetadata).not.toHaveBeenCalled(); expect(onData).not.toHaveBeenCalled();
});

it('aborts its consumer and deletes only its known operation before META', async () => {
  let abort = false;
  const fetcher = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    if (init?.method === 'DELETE') return new Response(null, { status: 204 });
    const stream = new ReadableStream<Uint8Array>({ start(controller) { init?.signal?.addEventListener('abort', () => { abort = true; controller.error(new Error('aborted')); }); } });
    expect(String(url)).toContain('neuron-comparisons');
    return new Response(stream, { headers });
  });
  const client = new ApiClient({ backendBaseUrl: 'https://backend.example' }, fetcher);
  const id = vi.fn();
  const operation = client.streamNeuronComparison(neuronSession, capabilities, comparison, { onOperationId: id });
  await vi.waitFor(() => expect(id).toHaveBeenCalledWith(operationId));
  await Promise.all([operation.cancel(), operation.cancel()]);
  expect(await operation.done).toEqual({ kind: 'cancelled' }); expect(abort).toBe(true);
  expect(fetcher.mock.calls.filter(([, init]) => init?.method === 'DELETE').map(([url]) => url)).toEqual([`https://backend.example/operations/${operationId}`]);
});
