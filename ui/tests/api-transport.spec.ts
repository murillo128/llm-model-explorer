import { createServer } from 'node:http';
import type { Server, ServerResponse } from 'node:http';
import { readFile } from 'node:fs/promises';
import { build } from 'vite';
import { expect, test } from '@playwright/test';
import type { StreamOperation } from '../src/api/client';

// The test-only IIFE bundles the actual public adapter, without changing the app shell.
declare global {
  interface Window {
    ApiTransport: typeof import('../src/api/client');
    concurrentTest: { left: StreamOperation; right: StreamOperation };
    transportTest: {
      operation: StreamOperation;
      abort: AbortController;
      id?: string;
      dataBytes: number;
      result?: string;
      failure?: string;
    };
  }
}
let server: Server;
let base: string;
let bundle: string;
let scalar: Buffer;
const open = new Map<string, ServerResponse>();
const closed = new Set<string>();
const deleted: string[] = [];
const preflights: string[] = [];
let sequence = 0;

test.beforeAll(async ({ baseURL }) => {
  const fixtures = JSON.parse(await readFile(new URL('../../api/fixtures/conformance.json', import.meta.url), 'utf8'));
  scalar = Buffer.from(fixtures.wire_cases.find((f: { name: string }) => f.name === 'tensor-scalar').wire_hex, 'hex');
  const output = await build({ configFile: false, logLevel: 'error', build: { write: false, minify: false, lib: { entry: new URL('../src/api/client.ts', import.meta.url).pathname, name: 'ApiTransport', formats: ['iife'] } } });
  const result = Array.isArray(output) ? output[0] : output;
  if (!result || !('output' in result)) throw new Error('Expected a browser bundle');
  const chunk = result.output.find(item => item.type === 'chunk');
  if (!chunk || chunk.type !== 'chunk') throw new Error('Missing browser bundle');
  bundle = chunk.code;
  server = createServer((request, response) => {
    response.setHeader('Access-Control-Allow-Origin', new URL(baseURL!).origin);
    response.setHeader('Access-Control-Allow-Methods', 'GET, DELETE');
    response.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    response.setHeader('Cache-Control', 'no-store');
    if (request.method === 'OPTIONS') {
      preflights.push(request.url!);
      response.writeHead(204).end();
    } else if (request.method === 'DELETE') {
      deleted.push(request.url!);
      response.writeHead(204).end();
    } else {
      const key = `01234567-89ab-cdef-0123-${String(++sequence).padStart(12, '0')}`;
      open.set(key, response);
      response.on('close', () => { closed.add(key); open.delete(key); });
      response.setHeader('Content-Type', 'application/vnd.llm-model-explorer.stream');
      response.setHeader('X-Operation-Id', key);
      if (!request.url?.includes('/hidden/')) response.setHeader('Access-Control-Expose-Headers', 'X-Operation-Id');
      response.flushHeaders();
    }
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Missing server port');
  base = `http://127.0.0.1:${address.port}/backend`;
});
test.afterAll(async () => {
  if (!server) return;
  for (const response of open.values()) response.destroy();
  server.closeAllConnections();
  await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
});

test('browser fetch streams progressively, exposes CORS headers, and cancels each consumer once', async ({ page }) => {
  await page.goto('/');
  await page.addScriptTag({ content: bundle });
  const start = async (tensor = 't') => {
    await page.evaluate(({ base, tensor }) => {
      const client = new window.ApiTransport.ApiClient({ backendBaseUrl: base });
      const abort = new AbortController();
      const operation = client.streamTensor('session', tensor, {
        signal: abort.signal,
        onOperationId: id => { window.transportTest.id = id; },
        onData: bytes => { window.transportTest.dataBytes += bytes.length; },
      });
      window.transportTest = { operation, abort, dataBytes: 0 };
      void operation.done.then(result => { window.transportTest.result = result.kind; }, error => { window.transportTest.failure = error.kind; });
    }, { base, tensor });
  };
  await start();
  await expect.poll(() => page.evaluate(() => window.transportTest.id)).toBeTruthy();
  const firstId = (await page.evaluate(() => window.transportTest.id))!;
  expect(await page.evaluate(() => window.transportTest.dataBytes)).toBe(0);
  open.get(firstId)!.write(scalar.subarray(0, scalar.length - 12));
  await expect.poll(() => page.evaluate(() => window.transportTest.dataBytes)).toBe(4);
  expect(await page.evaluate(() => window.transportTest.result)).toBeUndefined();
  open.get(firstId)!.write(scalar.subarray(scalar.length - 12));
  // Terminal alone is insufficient: the response remains open until the server ends it.
  expect(await page.evaluate(() => window.transportTest.result)).toBeUndefined();
  open.get(firstId)!.end();
  await expect.poll(() => page.evaluate(() => window.transportTest.result)).toBe('complete');
  const beforeDelete = deleted.length;
  await page.evaluate(() => window.transportTest.operation.cancel());
  expect(deleted).toHaveLength(beforeDelete);

  await start();
  await expect.poll(() => page.evaluate(() => window.transportTest.id)).toBeTruthy();
  const cancelledId = (await page.evaluate(() => window.transportTest.id))!;
  await page.evaluate(async () => {
    const operation = window.transportTest.operation;
    await Promise.all([operation.cancel(), operation.cancel()]);
  });
  await expect.poll(() => page.evaluate(() => window.transportTest.result)).toBe('cancelled');
  await expect.poll(() => closed.has(cancelledId)).toBe(true);
  expect(deleted.filter(path => path === `/backend/operations/${cancelledId}`)).toHaveLength(1);
  expect(preflights).toContain(`/backend/operations/${cancelledId}`);

  await start();
  await expect.poll(() => page.evaluate(() => window.transportTest.id)).toBeTruthy();
  const abortedId = (await page.evaluate(() => window.transportTest.id))!;
  await page.evaluate(() => window.transportTest.abort.abort());
  await expect.poll(() => page.evaluate(() => window.transportTest.result)).toBe('cancelled');
  await expect.poll(() => closed.has(abortedId)).toBe(true);
  expect(deleted).not.toContain(`/backend/operations/${abortedId}`);

  await start('hidden');
  await expect.poll(() => page.evaluate(() => window.transportTest.failure)).toBe('protocol');
  expect(await page.evaluate(() => window.transportTest.id)).toBeUndefined();
  await expect.poll(() => open.size).toBe(0);
});


test('cancelling one concurrent browser consumer leaves the other stream usable', async ({ page }) => {
  await page.goto('/');
  await page.addScriptTag({ content: bundle });
  const ids = await page.evaluate(async base => {
    const client = new window.ApiTransport.ApiClient({ backendBaseUrl: base });
    let resolveLeft!: (id: string) => void;
    let resolveRight!: (id: string) => void;
    const leftHeader = new Promise<string>(resolve => { resolveLeft = resolve; });
    const rightHeader = new Promise<string>(resolve => { resolveRight = resolve; });
    const left = client.streamTensor('session', 't', { onOperationId: resolveLeft });
    const right = client.streamTensor('session', 't', { onOperationId: resolveRight });
    window.concurrentTest = { left, right };
    return Promise.all([leftHeader, rightHeader]);
  }, base);
  expect(ids[0]).not.toBe(ids[1]);
  await page.evaluate(() => window.concurrentTest.left.cancel());
  expect(await page.evaluate(async () => (await window.concurrentTest.left.done).kind)).toBe('cancelled');
  expect(open.has(ids[1]!)).toBe(true);
  expect(deleted).not.toContain(`/backend/operations/${ids[1]}`);
  open.get(ids[1]!)!.end(scalar);
  expect(await page.evaluate(async () => (await window.concurrentTest.right.done).kind)).toBe('complete');
});
