import type { ApiClient, StreamOperation, StreamOptions } from '../api/client';
import { ApiFailure, requireProtocol } from '../api/errors';
import type { Metadata } from '../api/validation';
import { StreamWords } from '../explorers/stream-words';
import type { MatrixSource, MatrixUpdates } from '../matrix-explorer';

export type EmbeddingResult = 'values' | 'statistics' | 'distributions';
type Status = 'loading' | 'streaming' | 'complete' | 'cancelled' | 'failed';
type Statistics = Extract<Metadata, { kind: 'input_embeddings_statistics' }>;
type Distributions = Extract<Metadata, { kind: 'input_embeddings_distributions' }>;
export interface EmbeddingState {
  status: Status | 'unsupported';
  statistics: Status;
  distributions: Status;
  statisticsMetadata?: Statistics | undefined;
  source?: MatrixSource | undefined;
}
type Client = Pick<ApiClient, 'streamInputEmbeddings' | 'streamInputEmbeddingsStatistics' | 'streamInputEmbeddingsDistributions'>;

/** One ordered tokenizer generation and three independent consumers. Scalars
 * go straight to the subscribed renderer. Only early auxiliary counts are staged,
 * in one bounded generation-owned buffer which is released on subscription. */
export class EmbeddingController {
  private handles = new Map<EmbeddingResult, StreamOperation>();
  private epoch = 0;
  private disposed = false;
  private updates: MatrixUpdates | undefined;
  private source?: MatrixSource | undefined;
  private subscribed = false;
  private receivedData = false;
  private stopped = false;
  private distributionMetadata?: Distributions | undefined;
  private pendingCounts?: Uint32Array | undefined;
  private pendingCountLength = 0;
  private state: EmbeddingState = { status: 'loading', statistics: 'loading', distributions: 'loading' };
  private readonly tokenIds: number[];

  constructor(private readonly client: Client,
    private readonly sessionId: string, tokenIds: number[],
    private readonly signal: AbortSignal,
    private readonly changed: (state: EmbeddingState, allocate?: boolean) => void) {
    this.tokenIds = [...tokenIds];
    signal.addEventListener('abort', this.dispose, { once: true });
  }

  private publish(allocate?: boolean) { this.changed({ ...this.state, source: this.source }, allocate); }
  private status(result: EmbeddingResult) { return result === 'values' ? this.state.status : this.state[result]; }
  private active(result: EmbeddingResult) { return ['loading', 'streaming'].includes(this.status(result)); }
  private update(result: EmbeddingResult, status: Status | 'unsupported') {
    if (this.status(result) === status) return;
    this.state = result === 'values' ? { ...this.state, status } : { ...this.state, [result]: status };
    this.publish();
  }
  private releasePending() { this.pendingCounts = undefined; this.pendingCountLength = 0; }
  private verifyIdentity(metadata: Metadata) {
    requireProtocol('token_ids' in metadata && metadata.token_ids.length === this.tokenIds.length
      && metadata.token_ids.every((id, i) => id === this.tokenIds[i]), 'Embedding ordered identity mismatch');
  }
  private verifyShape(rows: number, columns: number) {
    if (!this.source) return;
    requireProtocol(rows === this.source.descriptor.shape[0] && columns === this.source.descriptor.shape[1], 'Embedding analysis shape mismatch');
  }
  private uploadCounts(values: Uint32Array, offset: number) {
    const metadata = this.distributionMetadata;
    requireProtocol(metadata, 'Missing embedding distribution metadata');
    for (const section of metadata.sections) {
      const start = section.offset / 4;
      const from = Math.max(offset, start), to = Math.min(offset + values.length, start + section.byte_length / 4);
      if (to > from) this.updates!.distribution(section.name === 'row_counts' ? 'rows' : 'columns',
        values.subarray(from - offset, to - offset), from - start);
    }
  }
  private attachAuxiliary() {
    if (!this.updates) return;
    if (this.state.statisticsMetadata) {
      try {
        this.verifyShape(...this.state.statisticsMetadata.shape);
        this.updates.transfer({ statistics: this.state.statisticsMetadata });
      } catch { this.state.statisticsMetadata = undefined; this.update('statistics', 'failed'); }
    }
    if (this.distributionMetadata) {
      try {
        const metadata = this.distributionMetadata;
        this.verifyShape(metadata.rows, metadata.columns);
        this.updates.distributionDomain({ minimum: metadata.domain_minimum, maximum: metadata.domain_maximum });
        if (this.pendingCounts) this.uploadCounts(this.pendingCounts.subarray(0, this.pendingCountLength), 0);
      } catch {
        this.distributionMetadata = undefined;
        void this.handles.get('distributions')?.cancel().catch(() => {});
        this.update('distributions', 'failed');
      }
      this.releasePending();
    }
  }

  start = () => {
    if (this.disposed || this.signal.aborted) return;
    const epoch = ++this.epoch;
    for (const handle of this.handles.values()) void handle.cancel().catch(() => {});
    this.handles.clear();
    this.receivedData = false;
    this.stopped = false;
    this.distributionMetadata = undefined;
    this.releasePending();
    this.state = { status: 'loading', statistics: 'loading', distributions: 'loading' };
    this.publish();
    const current = (result: EmbeddingResult) => !this.disposed && !this.signal.aborted && epoch === this.epoch && this.active(result);
    const words = new StreamWords<Float32Array>('float32', (values, offset) => {
      requireProtocol(this.updates, 'Embedding renderer is unavailable');
      this.updates.values(values, offset);
      this.update('values', 'streaming');
    });
    const counts = new StreamWords<Uint32Array>('uint32', (values, offset) => {
      requireProtocol(this.distributionMetadata, 'Missing embedding distribution metadata');
      if (this.updates) this.uploadCounts(values, offset);
      else {
        this.pendingCounts ??= new Uint32Array(this.distributionMetadata.byte_length / 4);
        this.pendingCounts.set(values, offset);
        this.pendingCountLength = offset + values.length;
      }
      this.update('distributions', 'streaming');
    });
    const start = (result: EmbeddingResult, run: (options: StreamOptions) => StreamOperation) => {
      if (!current(result)) return;
      try {
        const handle = run({
          signal: this.signal,
          onMetadata: metadata => {
            if (!current(result)) return;
            this.verifyIdentity(metadata);
            if (result === 'values') {
              requireProtocol(metadata.kind === 'input_embeddings', 'Unexpected embedding result');
              if (this.source) {
                this.verifyShape(...metadata.shape);
                startAnalysis();
                return;
              }
              this.source = {
                descriptor: { shape: metadata.shape, rank: 2, numel: metadata.byte_length / 4, logical_dtype: 'float32' },
                distributions: true,
                subscribe: updates => {
                  this.updates = updates;
                  if (this.subscribed && (this.receivedData || this.stopped)) this.start();
                  else this.attachAuxiliary();
                  this.subscribed = true;
                  return () => {
                    if (this.updates !== updates) return;
                    this.updates = undefined;
                    queueMicrotask(() => {
                      if (this.updates || this.disposed) return;
                      this.stopped = true;
                      ++this.epoch;
                      for (const handle of this.handles.values()) void handle.cancel().catch(() => {});
                      this.handles.clear();
                      this.releasePending();
                    });
                  };
                },
              };
              this.state.status = 'streaming';
              // Allocate/subscribe before DATA from this same network chunk.
              this.publish(true);
              startAnalysis();
            } else if (result === 'statistics') {
              requireProtocol(metadata.kind === 'input_embeddings_statistics', 'Unexpected embedding statistics');
              this.verifyShape(...metadata.shape);
            } else {
              requireProtocol(metadata.kind === 'input_embeddings_distributions', 'Unexpected embedding distributions');
              this.verifyShape(metadata.rows, metadata.columns);
              this.distributionMetadata = metadata;
              this.attachAuxiliary();
            }
          },
          onData: (bytes, offset) => {
            if (!current(result)) return;
            if (result === 'values') { this.receivedData = true; words.push(bytes, offset); }
            else if (result === 'distributions') { this.receivedData = true; counts.push(bytes, offset); }
          },
        });
        this.handles.set(result, handle);
        void handle.done.then(outcome => {
          if (!current(result)) return;
          if (outcome.kind === 'complete') {
            if (result === 'statistics') {
              requireProtocol(outcome.metadata.kind === 'input_embeddings_statistics', 'Unexpected embedding statistics');
              this.verifyIdentity(outcome.metadata);
              this.verifyShape(...outcome.metadata.shape);
              this.state.statisticsMetadata = outcome.metadata;
              this.updates?.transfer({ statistics: outcome.metadata });
            }
            this.updates?.flush?.();
            if (!current(result)) return;
            this.update(result, 'complete');
          } else this.fail(result, outcome.kind === 'cancelled' ? 'cancelled'
            : result === 'values' && outcome.error.code === 'unsupported_representation' ? 'unsupported' : 'failed');
        }).catch(error => {
          if (current(result)) this.fail(result, error instanceof ApiFailure && error.kind === 'cancelled' ? 'cancelled'
            : result === 'values' && error instanceof ApiFailure && error.detail?.code === 'unsupported_representation' ? 'unsupported' : 'failed');
        });
      } catch (error) {
        if (current(result)) this.fail(result, result === 'values' && error instanceof ApiFailure
          && error.detail?.code === 'unsupported_representation' ? 'unsupported' : 'failed');
      }
    };
    const request = { token_ids: this.tokenIds };
    // META establishes a supported table and exact shape. Do not request
    // analysis for an unavailable input table; never wait for value completion.
    const startAnalysis = () => {
      start('statistics', options => this.client.streamInputEmbeddingsStatistics(this.sessionId, request, options));
      start('distributions', options => this.client.streamInputEmbeddingsDistributions(this.sessionId, request, options));
    };
    start('values', options => this.client.streamInputEmbeddings(this.sessionId, request, options));
  };

  private fail(result: EmbeddingResult, status: Status | 'unsupported') {
    void this.handles.get(result)?.cancel().catch(() => {});
    if (result === 'values') {
      this.source = undefined;
      this.updates = undefined;
      for (const auxiliary of ['statistics', 'distributions'] as const) {
        if (this.active(auxiliary)) this.state[auxiliary] = 'cancelled';
        void this.handles.get(auxiliary)?.cancel().catch(() => {});
      }
      this.state.statisticsMetadata = undefined;
      this.distributionMetadata = undefined;
      this.releasePending();
    } else if (result === 'distributions') this.releasePending();
    else this.state.statisticsMetadata = undefined;
    this.update(result, status);
  }

  cancel = (result: EmbeddingResult) => { if (!this.disposed && this.active(result)) this.fail(result, 'cancelled'); };

  /** A hidden replacement that cannot render must never displace a valid matrix. */
  renderingFailed = () => {
    if (this.disposed) return;
    this.fail('values', 'failed');
    this.dispose();
  };

  dispose = () => {
    if (this.disposed) return;
    this.disposed = true;
    ++this.epoch;
    this.signal.removeEventListener('abort', this.dispose);
    for (const handle of this.handles.values()) void handle.cancel().catch(() => {});
    this.handles.clear();
    this.updates = undefined;
    this.source = undefined;
    this.distributionMetadata = undefined;
    this.releasePending();
    this.state = { status: 'cancelled', statistics: 'cancelled', distributions: 'cancelled' };
  };
}
