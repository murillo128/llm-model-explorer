// @vitest-environment node
import { expect, it, vi } from 'vitest';
import fixtures from '../../../api/fixtures/embeddings.json';
import { ApiClient, streamMediaType } from './client';

const headers = { 'Content-Type': streamMediaType, 'X-Operation-Id': '01234567-89ab-cdef-0123-456789abcdef' };
const wire = (name: string) => Uint8Array.from(Buffer.from(fixtures.wire_cases.find(f => f.name === name)!.wire_hex, 'hex'));
const config = { backendBaseUrl: 'https://backend.example/prefix/' };
it('posts generated request types, preserves duplicates and consumes split float32 bytes progressively', async () => {
  const bytes = wire('embeddings-duplicates');
  let stream!: ReadableStreamDefaultController<Uint8Array>;
  const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response(new ReadableStream({ start(c) { stream = c; } }), { headers }));
  const data = vi.fn();
  const ids = [3, 1, 3];
  const operation = new ApiClient(config, fetcher).streamInputEmbeddings('session', { token_ids: ids }, { onData: data });
  ids.reverse(); ids[0] = 9;
  for (let i = 0; i < bytes.length - 12; i++) stream.enqueue(bytes.subarray(i, i + 1));
  await vi.waitFor(() => expect(data).toHaveBeenCalled());
  const settled = vi.fn(); void operation.done.then(settled);
  expect(settled).not.toHaveBeenCalled();
  stream.enqueue(bytes.subarray(bytes.length - 12)); stream.close();
  expect((await operation.done).kind).toBe('complete');
  expect(fetcher.mock.calls[0]![0]).toBe('https://backend.example/prefix/sessions/session/embeddings');
  expect(fetcher.mock.calls[0]![1]).toMatchObject({ method: 'POST', body: '{"token_ids":[3,1,3]}', headers: { Accept: streamMediaType, 'Content-Type': 'application/json' } });
});
it.each([[0, 2], [2, 2], [2], []].map(ids => ({ ids })))('rejects mismatched ordered echo $ids before exposing metadata/data', async ({ ids }) => {
  const onMetadata = vi.fn(), onData = vi.fn();
  const client = new ApiClient(config, async () => new Response(wire('embeddings-asymmetric'), { headers }));
  await expect(client.streamInputEmbeddings('s', { token_ids: ids }, { onMetadata, onData }).done).rejects.toMatchObject({ kind: 'protocol' });
  expect(onMetadata).not.toHaveBeenCalled(); expect(onData).not.toHaveBeenCalled();
});
it('validates input before transport and handles empty sequences', async () => {
  const fetcher = vi.fn<typeof fetch>(async () => new Response(wire('embeddings-empty'), { headers }));
  const client = new ApiClient(config, fetcher);
  expect(() => client.streamInputEmbeddings('s', { token_ids: [-1] })).toThrow();
  expect(fetcher).not.toHaveBeenCalled();
  expect((await client.streamInputEmbeddings('s', { token_ids: [] }).done).kind).toBe('complete');
});
