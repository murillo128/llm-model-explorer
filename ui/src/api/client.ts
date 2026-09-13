import type { components, operations } from './generated/types';
import { ApiFailure, requireProtocol } from './errors';
import { LmexDecoder } from './lmex-decoder';
import type { DecoderCallbacks, StreamOutcome } from './lmex-decoder';
import type { RuntimeConfig } from './runtime-config';
import { uuidPattern, validateResponse, validateSchema } from './validation';
import type { Metadata } from './validation';

export const streamMediaType = 'application/vnd.llm-model-explorer.stream';
type Schemas = components['schemas'];
export interface StreamOptions extends DecoderCallbacks {
  signal?: AbortSignal;
  onOperationId?: (operationId: string) => void;
}
export interface StreamOperation {
  readonly operationId: string | undefined;
  /** Resolves only after EOF. HTTP/protocol/transport failures reject with ApiFailure. */
  readonly done: Promise<StreamOutcome>;
  /** Abort consumption and explicitly release known backend work. Repeated calls share one DELETE. */
  cancel(): Promise<void>;
}

function mediaType(response: Response): string {
  return response.headers.get('Content-Type')?.split(';')[0]?.trim().toLowerCase() ?? '';
}
function transportFailure(error: unknown, signal?: AbortSignal): ApiFailure {
  if (error instanceof ApiFailure) return error;
  return new ApiFailure(signal?.aborted ? 'cancelled' : 'transport', signal?.aborted ? 'Request cancelled' : 'HTTP transport interrupted', undefined, undefined, { cause: error });
}
async function readJson(response: Response): Promise<unknown> {
  requireProtocol(mediaType(response) === 'application/json', 'Expected application/json');
  try { return JSON.parse(await response.text()); }
  catch (cause) {
    if (cause instanceof SyntaxError) throw new ApiFailure('protocol', 'Invalid HTTP JSON', undefined, undefined, { cause });
    throw cause;
  }
}
async function checkStatus(response: Response, expected: number): Promise<void> {
  if (!response.ok) {
    const detail = validateSchema('Error', await readJson(response));
    throw new ApiFailure('http', detail.message, response.status, detail);
  }
  requireProtocol(response.status === expected, `Unexpected HTTP status ${response.status}`);
}

/** Construct once from loadRuntimeConfig's result. No React or same-origin dependency. */
export class ApiClient {
  private readonly base: string;
  constructor(config: RuntimeConfig, private readonly fetcher: typeof fetch = globalThis.fetch.bind(globalThis)) {
    this.base = config.backendBaseUrl.replace(/\/+$/, '');
  }
  private sessionPath(id: string): string { return `/sessions/${encodeURIComponent(id)}`; }
  private tensorPath(session: string, tensor: string): string {
    return `${this.sessionPath(session)}/tensors/${encodeURIComponent(tensor)}`;
  }
  private async request<T>(path: string, method: string, status: number, schema?: Parameters<typeof validateResponse>[0], body?: unknown, signal?: AbortSignal): Promise<T> {
    let response: Response | undefined;
    try {
      response = await this.fetcher(this.base + path, {
        method, cache: 'no-store', ...(signal ? { signal } : {}),
        headers: { Accept: 'application/json', ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      });
      await checkStatus(response, status);
      if (!schema) return undefined as T;
      const result = await readJson(response);
      validateResponse(schema, result);
      return result as T;
    } catch (error) { throw transportFailure(error, signal); }
    finally { if (response?.body && !response.body.locked) await response.body.cancel().catch(() => {}); }
  }
  listModels(signal?: AbortSignal) {
    return this.request<operations['listModels']['responses'][200]['content']['application/json']>('/models', 'GET', 200, 'listModels', undefined, signal);
  }
  createSession(body: Schemas['CreateSessionRequest'], signal?: AbortSignal) {
    validateSchema('CreateSessionRequest', body);
    return this.request<Schemas['Session']>('/sessions', 'POST', 201, 'createSession', body, signal);
  }
  getSession(id: string, signal?: AbortSignal) {
    return this.request<Schemas['Session']>(this.sessionPath(id), 'GET', 200, 'getSession', undefined, signal);
  }
  deleteSession(id: string, signal?: AbortSignal) {
    return this.request<void>(this.sessionPath(id), 'DELETE', 204, undefined, undefined, signal);
  }
  listTensors(id: string, signal?: AbortSignal) {
    return this.request<Schemas['TensorInventory']>(`${this.sessionPath(id)}/tensors`, 'GET', 200, 'listTensors', undefined, signal);
  }
  tokenize(id: string, body: Schemas['TokenizeRequest'], signal?: AbortSignal) {
    validateSchema('TokenizeRequest', body);
    return this.request<Schemas['TokenizeResponse']>(`${this.sessionPath(id)}/tokenize`, 'POST', 200, 'tokenize', body, signal);
  }
  cancelOperation(id: string, signal?: AbortSignal) {
    return this.request<void>(`/operations/${encodeURIComponent(id)}`, 'DELETE', 204, undefined, undefined, signal);
  }
  streamTensor(session: string, tensor: string, options: StreamOptions = {}): StreamOperation {
    return this.stream(`${this.tensorPath(session, tensor)}/data`, tensor, 'tensor', options);
  }
  streamTensorStatistics(session: string, tensor: string, options: StreamOptions = {}): StreamOperation {
    return this.stream(`${this.tensorPath(session, tensor)}/statistics`, tensor, 'tensor_statistics', options);
  }
  streamTensorDistributions(session: string, tensor: string, options: StreamOptions = {}): StreamOperation {
    return this.stream(`${this.tensorPath(session, tensor)}/distributions`, tensor, 'tensor_distributions', options);
  }
  private stream(path: string, tensor: string, kind: Metadata['kind'], options: StreamOptions): StreamOperation {
    const controller = new AbortController();
    let operationId: string | undefined;
    let settled = false;
    let cancellation: Promise<void> | undefined;
    const abort = () => controller.abort();
    options.signal?.addEventListener('abort', abort, { once: true });
    if (options.signal?.aborted) abort();
    const done = (async (): Promise<StreamOutcome> => {
      let response: Response | undefined;
      let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
      try {
        response = await this.fetcher(this.base + path, { signal: controller.signal, cache: 'no-store', headers: { Accept: streamMediaType } });
        await checkStatus(response, 200);
        const id = response.headers.get('X-Operation-Id');
        requireProtocol(id && uuidPattern.test(id), 'Missing or invalid X-Operation-Id (check CORS exposure)');
        operationId = id;
        options.onOperationId?.(id);
        requireProtocol(mediaType(response) === streamMediaType, 'Invalid stream media type');
        requireProtocol(response.body, 'Missing stream body');
        const decoder = new LmexDecoder({ ...options, onMetadata: (metadata) => {
          requireProtocol(metadata.kind === kind && metadata.tensor_id === tensor, 'Unexpected stream result identity');
          options.onMetadata?.(metadata);
        } });
        reader = response.body.getReader();
        while (true) {
          controller.signal.throwIfAborted();
          const { value, done } = await reader.read();
          controller.signal.throwIfAborted();
          if (done) return decoder.finish();
          decoder.push(value);
        }
      } catch (error) {
        if (controller.signal.aborted) return { kind: 'cancelled' };
        throw transportFailure(error);
      } finally {
        settled = true;
        options.signal?.removeEventListener('abort', abort);
        if (reader) {
          try { await reader.cancel(); } catch { /* Transport may already be closed. */ }
          finally { reader.releaseLock(); }
        } else if (response?.body && !response.body.locked) await response.body.cancel().catch(() => {});
      }
    })();
    return {
      get operationId() { return operationId; },
      done,
      cancel: () => {
        if (cancellation) return cancellation;
        if (settled) return Promise.resolve();
        abort();
        cancellation = operationId ? this.cancelOperation(operationId) : Promise.resolve();
        return cancellation;
      },
    };
  }
}
