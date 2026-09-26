import type { components, operations } from './generated/types';
import { validateArchitecture } from './architecture-validation';
import type { ArchitectureContext } from './architecture-validation';
import { ApiFailure, requireProtocol } from './errors';
import { LmexDecoder } from './lmex-decoder';
import type { DecoderCallbacks, StreamOutcome } from './lmex-decoder';
import type { RuntimeConfig } from './runtime-config';
import { uuidPattern, validateResponse, validateSchema } from './validation';
import type { Metadata } from './validation';

export type RequestActivity = { id: symbol; phase: 'start' | 'response' | 'end'; failure?: ApiFailure | undefined; quiet?: boolean };
export type ActivityObserver = (activity: RequestActivity) => void;

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
export const architectureByteLimit = 33_554_432;
async function readJson(response: Response, limit?: number): Promise<unknown> {
  requireProtocol(mediaType(response) === 'application/json', 'Expected application/json');
  try {
    if (limit === undefined) return JSON.parse(await response.text());
    requireProtocol(response.body, 'Missing architecture response body');
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8', { fatal: true });
    let size = 0, text = '';
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        size += value.byteLength;
        requireProtocol(size <= limit, 'Architecture exceeds the 32 MiB limit');
        text += decoder.decode(value, { stream: true });
      }
      return JSON.parse(text + decoder.decode());
    } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
  }
  catch (cause) {
    if (cause instanceof SyntaxError) throw new ApiFailure('protocol', 'Invalid HTTP JSON', undefined, undefined, { cause });
    throw cause;
  }
}
async function checkStatus(response: Response, expected: number, limit?: number): Promise<void> {
  if (!response.ok) {
    const detail = validateSchema('Error', await readJson(response, limit));
    throw new ApiFailure('http', detail.message, response.status, detail);
  }
  requireProtocol(response.status === expected, `Unexpected HTTP status ${response.status}`);
}

/** Construct once from loadRuntimeConfig's result. No React or same-origin dependency. */
export class ApiClient {
  private readonly base: string;
  constructor(config: RuntimeConfig, private readonly fetcher: typeof fetch = globalThis.fetch.bind(globalThis), private readonly observer?: ActivityObserver) {
    this.base = config.backendBaseUrl.replace(/\/+$/, '');
  }
  /** A consumer-scoped observer; transport and operation ownership remain unchanged. */
  observe(observer: ActivityObserver) {
    return new ApiClient({ backendBaseUrl: this.base }, this.fetcher, observer);
  }
  private activity(signal?: AbortSignal, quiet = false) {
    const id = Symbol('api-request');
    let ended = false;
    const abort = () => emit('end');
    const emit = (phase: RequestActivity['phase'], failure?: ApiFailure) => {
      if (ended) return;
      if (phase === 'end') {
        ended = true;
        signal?.removeEventListener('abort', abort);
        this.observer?.({ id, phase, failure: signal?.aborted ? undefined : failure, quiet });
      } else if (!signal?.aborted) this.observer?.({ id, phase });
    };
    signal?.addEventListener('abort', abort, { once: true });
    return emit;
  }
  private sessionPath(id: string): string { return `/sessions/${encodeURIComponent(id)}`; }
  private tensorPath(session: string, tensor: string): string {
    return `${this.sessionPath(session)}/tensors/${encodeURIComponent(tensor)}`;
  }
  private async request<T>(path: string, method: string, status: number, schema?: Parameters<typeof validateResponse>[0], body?: unknown, signal?: AbortSignal): Promise<T> {
    let response: Response | undefined;
    const activity = this.activity(signal, method === 'DELETE');
    activity('start');
    try {
      response = await this.fetcher(this.base + path, {
        method, cache: 'no-store', ...(signal ? { signal } : {}),
        headers: { Accept: 'application/json', ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      });
      activity('response');
      await checkStatus(response, status, schema === 'getArchitecture' ? architectureByteLimit : undefined);
      if (!schema) return undefined as T;
      const result = await readJson(response, schema === 'getArchitecture' ? architectureByteLimit : undefined);
      validateResponse(schema, result);
      return result as T;
    } catch (error) {
      const failure = transportFailure(error, signal);
      activity('end', failure);
      throw failure;
    }
    finally { activity('end'); if (response?.body && !response.body.locked) await response.body.cancel().catch(() => {}); }
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
  async getArchitecture(id: string, context: ArchitectureContext, signal?: AbortSignal) {
    const response = await this.request<Schemas['ArchitectureResponse']>(`${this.sessionPath(id)}/architecture`, 'GET', 200, 'getArchitecture', undefined, signal);
    return validateArchitecture(response, context);
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
  streamInputEmbeddings(session: string, body: Schemas['InputEmbeddingsRequest'], options: StreamOptions = {}): StreamOperation {
    validateSchema('InputEmbeddingsRequest', body);
    // Snapshot the ordered identity: caller mutation must not alter echo validation.
    const token_ids = [...body.token_ids];
    return this.stream(`${this.sessionPath(session)}/embeddings`, token_ids, 'input_embeddings', options, { token_ids });
  }
  streamInputEmbeddingsStatistics(session: string, body: Schemas['InputEmbeddingsRequest'], options: StreamOptions = {}): StreamOperation {
    validateSchema('InputEmbeddingsRequest', body);
    const token_ids = [...body.token_ids];
    return this.stream(`${this.sessionPath(session)}/embeddings/statistics`, token_ids, 'input_embeddings_statistics', options, { token_ids });
  }
  streamInputEmbeddingsDistributions(session: string, body: Schemas['InputEmbeddingsRequest'], options: StreamOptions = {}): StreamOperation {
    validateSchema('InputEmbeddingsRequest', body);
    const token_ids = [...body.token_ids];
    return this.stream(`${this.sessionPath(session)}/embeddings/distributions`, token_ids, 'input_embeddings_distributions', options, { token_ids });
  }
  private stream(path: string, identity: string | readonly number[], kind: Metadata['kind'], options: StreamOptions, body?: Schemas['InputEmbeddingsRequest']): StreamOperation {
    const controller = new AbortController();
    let operationId: string | undefined;
    let settled = false;
    let cancellation: Promise<void> | undefined;
    const abort = () => controller.abort();
    options.signal?.addEventListener('abort', abort, { once: true });
    if (options.signal?.aborted) abort();
    const activity = this.activity(controller.signal);
    activity('start');
    const done = (async (): Promise<StreamOutcome> => {
      let response: Response | undefined;
      let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
      try {
        response = await this.fetcher(this.base + path, { signal: controller.signal, cache: 'no-store',
          headers: { Accept: streamMediaType, ...(body ? { 'Content-Type': 'application/json' } : {}) },
          ...(body ? { method: 'POST', body: JSON.stringify(body) } : {}),
        });
        activity('response');
        await checkStatus(response, 200);
        const id = response.headers.get('X-Operation-Id');
        requireProtocol(id && uuidPattern.test(id), 'Missing or invalid X-Operation-Id (check CORS exposure)');
        operationId = id;
        options.onOperationId?.(id);
        requireProtocol(mediaType(response) === streamMediaType, 'Invalid stream media type');
        requireProtocol(response.body, 'Missing stream body');
        const decoder = new LmexDecoder({ ...options, onMetadata: (metadata) => {
          requireProtocol(metadata.kind === kind, 'Unexpected stream result kind');
          requireProtocol('token_ids' in metadata
            ? Array.isArray(identity) && metadata.token_ids.length === identity.length && metadata.token_ids.every((id, i) => id === identity[i])
            : metadata.tensor_id === identity, 'Unexpected stream result identity');
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
        const failure = transportFailure(error);
        activity('end', failure);
        throw failure;
      } finally {
        activity('end');
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
