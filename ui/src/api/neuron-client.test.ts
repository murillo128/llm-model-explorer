// @vitest-environment node
import { expect, it, vi } from 'vitest';
import { ApiClient } from './client';
import { neuronCapabilities, neuronSession } from '../test/neuron-fixture';

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
