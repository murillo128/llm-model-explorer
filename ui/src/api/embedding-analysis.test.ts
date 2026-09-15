// @vitest-environment node
import { expect, it, vi } from 'vitest';
import fixtures from '../../../api/fixtures/embedding-analysis.json';
import { ApiClient, streamMediaType } from './client';

const headers = { 'Content-Type': streamMediaType, 'X-Operation-Id': '01234567-89ab-cdef-0123-456789abcdef' };
const config = { backendBaseUrl: 'https://backend.example/prefix/' };
const endpoints = [
  ['statistics', 'streamInputEmbeddingsStatistics'],
  ['distributions', 'streamInputEmbeddingsDistributions'],
] as const;
const wire = (name: string) => Uint8Array.from(Buffer.from(fixtures.wire_cases.find(f => f.name === name)!.wire_hex, 'hex'));

for (const [suffix, method] of endpoints) {
  it(`${suffix}: posts ordered identity, validates split bytes, and waits for EOF`, async () => {
    let stream!: ReadableStreamDefaultController<Uint8Array>;
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response(new ReadableStream({ start(c) { stream = c; } }), { headers }));
    const onMetadata = vi.fn(), onData = vi.fn(), completed = vi.fn();
    const ids = [2, 0, 2];
    const operation = new ApiClient(config, fetcher)[method]('session', { token_ids: ids }, { onMetadata, onData });
    ids[0] = 0;
    void operation.done.then(completed);
    const bytes = wire(`embedding-${suffix}-duplicates`);
    for (let i = 0; i < bytes.length; i++) stream.enqueue(bytes.subarray(i, i + 1));
    await vi.waitFor(() => expect(onMetadata).toHaveBeenCalledOnce());
    expect(completed).not.toHaveBeenCalled();
    if (suffix === 'distributions') await vi.waitFor(() => expect(onData).toHaveBeenCalled());
    else expect(onData).not.toHaveBeenCalled();
    stream.close();
    expect((await operation.done).kind).toBe('complete');
    expect(fetcher.mock.calls[0]![0]).toBe(`https://backend.example/prefix/sessions/session/embeddings/${suffix}`);
    expect(fetcher.mock.calls[0]![1]).toMatchObject({ method: 'POST', body: '{"token_ids":[2,0,2]}', cache: 'no-store', headers: { Accept: streamMediaType, 'Content-Type': 'application/json' } });
  });

  it.each(fixtures.association_cases)(`${suffix}: validates ordered request $token_ids`, async ({ token_ids, valid }) => {
    const onMetadata = vi.fn(), onData = vi.fn();
    const client = new ApiClient(config, async () => new Response(wire(`embedding-${suffix}-duplicates`), { headers }));
    const operation = client[method]('s', { token_ids }, { onMetadata, onData });
    if (valid) expect((await operation.done).kind).toBe('complete');
    else {
      await expect(operation.done).rejects.toMatchObject({ kind: 'protocol' });
      expect(onMetadata).not.toHaveBeenCalled(); expect(onData).not.toHaveBeenCalled();
    }
  });

  it(`${suffix}: rejects the other result kind and invalid requests`, async () => {
    const fetcher = vi.fn<typeof fetch>(async () => new Response(wire(`embedding-${suffix === 'statistics' ? 'distributions' : 'statistics'}-duplicates`), { headers }));
    const client = new ApiClient(config, fetcher);
    const onMetadata = vi.fn(), onData = vi.fn();
    expect(() => client[method]('s', { token_ids: [-1] })).toThrow();
    expect(() => client[method]('s', { token_ids: [1.5] })).toThrow();
    expect(fetcher).not.toHaveBeenCalled();
    await expect(client[method]('s', { token_ids: [2, 0, 2] }, { onMetadata, onData }).done).rejects.toMatchObject({ kind: 'protocol' });
    expect(onMetadata).not.toHaveBeenCalled(); expect(onData).not.toHaveBeenCalled();
  });

  it(`${suffix}: handles an empty requested matrix`, async () => {
    const client = new ApiClient(config, async () => new Response(wire(`embedding-${suffix}-empty`), { headers }));
    expect((await client[method]('s', { token_ids: [] }).done).kind).toBe('complete');
  });

  it(`${suffix}: cancellation releases only its own operation`, async () => {
    let stream!: ReadableStreamDefaultController<Uint8Array>;
    const cancel = vi.fn();
    const fetcher = vi.fn<typeof fetch>(async (_url, options) => options?.method === 'DELETE'
      ? new Response(null, { status: 204 })
      : new Response(new ReadableStream({ start(c) { stream = c; }, cancel }), { headers }));
    const operation = new ApiClient(config, fetcher)[method]('s', { token_ids: [2, 0, 2] });
    await vi.waitFor(() => expect(operation.operationId).toBe(headers['X-Operation-Id']));
    await operation.cancel();
    // The fetch test double must wake the pending reader on abort.
    stream.close();
    expect(await operation.done).toEqual({ kind: 'cancelled' });
    await operation.cancel();
    expect(fetcher.mock.calls.filter(([, options]) => options?.method === 'DELETE')).toHaveLength(1);
    expect(fetcher.mock.calls[1]![0]).toContain(`/operations/${headers['X-Operation-Id']}`);
  });
}
