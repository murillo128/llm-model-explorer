// @vitest-environment node
import { expect, it, vi } from 'vitest';
import fixtures from '../../../api/fixtures/conformance.json';
import { ApiClient, streamMediaType } from './client';
import { ApiFailure } from './errors';

const id = '01234567-89ab-cdef-0123-456789abcdef';
const config = { backendBaseUrl: 'https://backend.example/prefix/' };
const wire = (name: string) => Uint8Array.from(Buffer.from(fixtures.wire_cases.find(f => f.name === name)!.wire_hex, 'hex'));
const headers = { 'Content-Type': streamMediaType, 'X-Operation-Id': id };

it('exposes headers and first data before the final network chunk; releases the reader', async () => {
  let streamController!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({ start(controller) { streamController = controller; } });
  const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response(body, { headers }));
  const client = new ApiClient(config, fetcher);
  const onOperationId = vi.fn();
  const onData = vi.fn();
  const operation = client.streamTensor('session', 't', { onOperationId, onData });
  await vi.waitFor(() => expect(onOperationId).toHaveBeenCalledWith(id));
  expect(operation.operationId).toBe(id);
  expect(onData).not.toHaveBeenCalled();
  const bytes = wire('tensor-scalar');
  streamController.enqueue(bytes.subarray(0, bytes.length - 12));
  await vi.waitFor(() => expect(onData).toHaveBeenCalled());
  const settled = vi.fn();
  void operation.done.then(settled);
  expect(settled).not.toHaveBeenCalled();
  streamController.enqueue(bytes.subarray(bytes.length - 12));
  await Promise.resolve();
  expect(settled).not.toHaveBeenCalled();
  streamController.close();
  expect((await operation.done).kind).toBe('complete');
  expect(body.locked).toBe(false);
  expect(fetcher.mock.calls[0]![0]).toBe('https://backend.example/prefix/sessions/session/tensors/t/data');
});

it.each([
  ['missing operation header', { 'Content-Type': streamMediaType }],
  ['invalid operation header', { ...headers, 'X-Operation-Id': 'bad' }],
  ['missing media type', { 'X-Operation-Id': id }],
  ['wrong media type', { ...headers, 'Content-Type': 'application/json' }],
])('rejects %s and cancels unread bodies', async (_, responseHeaders) => {
  const cancel = vi.fn();
  const body = new ReadableStream<Uint8Array>({ cancel });
  const client = new ApiClient(config, vi.fn<typeof fetch>().mockResolvedValue(new Response(body, { headers: responseHeaders })));
  await expect(client.streamTensor('s', 't').done).rejects.toMatchObject({ kind: 'protocol' });
  expect(cancel).toHaveBeenCalledOnce();
  expect(body.locked).toBe(false);
});

it('distinguishes HTTP JSON, backend, cancellation, protocol and transport outcomes', async () => {
  const fetcher = vi.fn<typeof fetch>();
  const client = new ApiClient(config, fetcher);
  const error = { code: 'tensor_not_found', message: 'Missing tensor' };
  fetcher.mockResolvedValueOnce(new Response(JSON.stringify(error), { status: 404, headers: { 'Content-Type': 'application/json' } }));
  await expect(client.streamTensor('s', 't').done).rejects.toMatchObject({ kind: 'http', status: 404, detail: error });
  fetcher.mockResolvedValueOnce(new Response(wire('early-error'), { headers }));
  expect((await client.streamTensor('s', 't').done).kind).toBe('backend');
  fetcher.mockResolvedValueOnce(new Response(wire('early-cancelled'), { headers }));
  expect((await client.streamTensor('s', 't').done).kind).toBe('cancelled');
  fetcher.mockResolvedValueOnce(new Response(wire('bad-magic'), { headers }));
  await expect(client.streamTensor('s', 't').done).rejects.toMatchObject({ kind: 'protocol' });
  fetcher.mockResolvedValueOnce(new Response(wire('truncated-payload'), { headers }));
  await expect(client.streamTensor('s', 't').done).rejects.toMatchObject({ kind: 'transport' });
  fetcher.mockRejectedValueOnce(new TypeError('Connection lost'));
  await expect(client.streamTensor('s', 't').done).rejects.toMatchObject({ kind: 'transport' });
  fetcher.mockResolvedValueOnce(new Response('{', { status: 500, headers: { 'Content-Type': 'application/json' } }));
  await expect(client.streamTensor('s', 't').done).rejects.toMatchObject({ kind: 'protocol' });
});

it('rejects metadata for the wrong capability or tensor', async () => {
  const client = new ApiClient(config, async () => new Response(wire('tensor-scalar'), { headers }));
  await expect(client.streamTensorStatistics('s', 't').done).rejects.toMatchObject({ kind: 'protocol' });
  await expect(client.streamTensor('s', 'other').done).rejects.toMatchObject({ kind: 'protocol' });
});

it('shares explicit cancellation, aborts consumption and releases listeners/readers', async () => {
  const external = new AbortController();
  const removeListener = vi.spyOn(external.signal, 'removeEventListener');
  let body!: ReadableStream<Uint8Array>;
  const cancel = vi.fn();
  const fetcher = vi.fn<typeof fetch>(async (_url, init) => {
    if (init?.method === 'DELETE') return new Response(null, { status: 204 });
    body = new ReadableStream({ start(controller) { init?.signal?.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')), { once: true }); }, cancel });
    return new Response(body, { headers });
  });
  const client = new ApiClient(config, fetcher);
  const operation = client.streamTensor('s', 't', { signal: external.signal });
  await vi.waitFor(() => expect(operation.operationId).toBe(id));
  const first = operation.cancel();
  expect(operation.cancel()).toBe(first);
  await first;
  expect((await operation.done).kind).toBe('cancelled');
  await operation.cancel();
  expect(fetcher.mock.calls.filter(([, init]) => init?.method === 'DELETE')).toHaveLength(1);
  expect(fetcher.mock.calls[1]![0]).toBe(`https://backend.example/prefix/operations/${id}`);
  expect(removeListener).toHaveBeenCalled();
  expect(body.locked).toBe(false);
});

it('pre-header abort needs no operation DELETE', async () => {
  const fetcher = vi.fn<typeof fetch>(async (_url, init) => new Promise((_resolve, reject) => {
    init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
  }));
  const operation = new ApiClient(config, fetcher).streamTensor('s', 't');
  await operation.cancel();
  expect((await operation.done).kind).toBe('cancelled');
  expect(fetcher).toHaveBeenCalledOnce();
});

it('uses explicit typed endpoint methods and checks JSON responses', async () => {
  const fetcher = vi.fn<typeof fetch>();
  const client = new ApiClient(config, fetcher);
  const json = (value: unknown, status = 200) => fetcher.mockResolvedValueOnce(new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json; charset=utf-8' } }));
  json({ models: [], diagnostics: [] });
  expect(await client.listModels()).toEqual({ models: [], diagnostics: [] });
  json({ id, model_id: 'model' }, 201);
  expect(await client.createSession({ model_id: 'model' })).toEqual({ id, model_id: 'model' });
  json({ id, model_id: 'model' });
  await client.getSession(id);
  json({ coverage: 'complete', diagnostics: [], tensors: [] });
  await client.listTensors(id);
  const tokenized = fixtures.schema_cases.find(c => c.name === 'unicode-overlapping-specials')!.value;
  json(tokenized);
  await client.tokenize(id, { text: 'A😀é<special>' });
  fetcher.mockResolvedValueOnce(new Response(null, { status: 204 }));
  await client.deleteSession(id);
  fetcher.mockResolvedValueOnce(new Response(null, { status: 204 }));
  await client.cancelOperation(id);
  expect(fetcher.mock.calls.map(([url, init]) => [url, init?.method])).toEqual([
    [`${config.backendBaseUrl}models`, 'GET'], [`${config.backendBaseUrl}sessions`, 'POST'],
    [`${config.backendBaseUrl}sessions/${id}`, 'GET'], [`${config.backendBaseUrl}sessions/${id}/tensors`, 'GET'],
    [`${config.backendBaseUrl}sessions/${id}/tokenize`, 'POST'], [`${config.backendBaseUrl}sessions/${id}`, 'DELETE'],
    [`${config.backendBaseUrl}operations/${id}`, 'DELETE'],
  ]);
  json({ models: [null], diagnostics: [] });
  await expect(client.listModels()).rejects.toBeInstanceOf(ApiFailure);
  expect(() => client.createSession({ model_id: '' })).toThrow(ApiFailure);
});

it('reports explicit DELETE failure separately from local cancellation', async () => {
  const fetcher = vi.fn<typeof fetch>(async (_url, init) => {
    if (init?.method === 'DELETE') return new Response(JSON.stringify({ code: 'resource_exhausted', message: 'Busy' }), { status: 503, headers: { 'Content-Type': 'application/json' } });
    const body = new ReadableStream<Uint8Array>({ start(controller) {
      init?.signal?.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')), { once: true });
    } });
    return new Response(body, { headers });
  });
  const operation = new ApiClient(config, fetcher).streamTensor('s', 't');
  await vi.waitFor(() => expect(operation.operationId).toBe(id));
  await expect(operation.cancel()).rejects.toMatchObject({ kind: 'http', status: 503 });
  expect((await operation.done).kind).toBe('cancelled');
});

it('rejects a missing body and a transport failure even after COMPLETE', async () => {
  const fetcher = vi.fn<typeof fetch>().mockResolvedValueOnce(new Response(null, { headers }));
  const client = new ApiClient(config, fetcher);
  await expect(client.streamTensor('s', 't').done).rejects.toMatchObject({ kind: 'protocol' });
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({ start(c) { controller = c; } });
  fetcher.mockResolvedValueOnce(new Response(body, { headers }));
  const onData = vi.fn();
  const operation = client.streamTensor('s', 't', { onData });
  controller.enqueue(wire('tensor-scalar'));
  await vi.waitFor(() => expect(onData).toHaveBeenCalled());
  controller.error(new TypeError('Connection lost'));
  await expect(operation.done).rejects.toMatchObject({ kind: 'transport' });
  expect(body.locked).toBe(false);
});
