// @vitest-environment node
import { expect, it, vi } from 'vitest';
import fixture from '../../../api/fixtures/architecture.json';
import { ApiClient, architectureByteLimit } from './client';

const context = { modelId: fixture.context.session.model_id };
it('retrieves typed architecture under the configured base and enforces contextual identity', async () => {
  const fetcher = vi.fn<typeof fetch>(async () => new Response(JSON.stringify(fixture.response), { headers: { 'Content-Type': 'application/json' } }));
  const client = new ApiClient({ backendBaseUrl: 'https://backend.example/prefix' }, fetcher);
  expect((await client.getArchitecture('s', context)).status).toBe('available');
  expect(fetcher.mock.calls[0]![0]).toBe('https://backend.example/prefix/sessions/s/architecture');
  await expect(client.getArchitecture('s', { ...context, modelId: 'other' })).rejects.toMatchObject({ kind: 'protocol' });
});
it('stops a chunked oversized response during reading and releases its body', async () => {
  const cancel = vi.fn(); let chunks = 0;
  const block = new Uint8Array(1024 * 1024).fill(32);
  const body = new ReadableStream<Uint8Array>({ pull(controller) { chunks++; controller.enqueue(block); }, cancel });
  const client = new ApiClient({ backendBaseUrl: 'https://backend.example' }, async () => new Response(body, { headers: { 'Content-Type': 'application/json' } }));
  await expect(client.getArchitecture('s', context)).rejects.toMatchObject({ kind: 'protocol' });
  expect(cancel).toHaveBeenCalledOnce(); expect(body.locked).toBe(false);
  expect(chunks).toBeLessThanOrEqual(architectureByteLimit / block.length + 2);
});
