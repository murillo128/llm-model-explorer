import type { ApiClient, StreamOperation } from '../api/client';
import { ApiFailure, requireProtocol } from '../api/errors';
import { StreamWords } from '../explorers/stream-words';
import type { MatrixSource, MatrixUpdates } from '../matrix-explorer';

export interface EmbeddingState {
  status: 'loading' | 'streaming' | 'complete' | 'cancelled' | 'failed' | 'unsupported';
  source?: MatrixSource | undefined;
}

/** One tokenizer generation. No result cache or second scalar copy: the renderer
 * consumes each borrowed chunk synchronously, retaining its own inspection copy.
 * A renderer remount restarts transport at zero instead of replaying staged data. */
export class EmbeddingController {
  private handle?: StreamOperation;
  private epoch = 0;
  private disposed = false;
  private updates: MatrixUpdates | undefined;
  private source?: MatrixSource | undefined;
  private subscribed = false;
  private receivedData = false;
  private stopped = false;

  constructor(private readonly client: Pick<ApiClient, 'streamInputEmbeddings'>,
    private readonly sessionId: string, private readonly tokenIds: number[],
    private readonly signal: AbortSignal,
    private readonly changed: (state: EmbeddingState, allocate?: boolean) => void) {
    signal.addEventListener('abort', this.dispose, { once: true });
  }

  start = () => {
    if (this.disposed || this.signal.aborted) return;
    const epoch = ++this.epoch;
    this.receivedData = false;
    this.stopped = false;
    void this.handle?.cancel().catch(() => {});
    const current = () => !this.disposed && !this.signal.aborted && epoch === this.epoch;
    const words = new StreamWords<Float32Array>('float32', (values, offset) => {
      requireProtocol(this.updates, 'Embedding renderer is unavailable');
      this.updates.values(values, offset);
      this.changed({ status: 'streaming', source: this.source });
    });
    try {
      this.handle = this.client.streamInputEmbeddings(this.sessionId, { token_ids: this.tokenIds }, {
        signal: this.signal,
        onMetadata: metadata => {
          if (!current()) return;
          requireProtocol(metadata.kind === 'input_embeddings', 'Unexpected embedding result');
          if (this.source) {
            requireProtocol(metadata.shape.every((n, i) => n === this.source!.descriptor.shape[i]), 'Embedding shape changed during reconstruction');
            return;
          }
          this.source = {
            descriptor: { shape: metadata.shape, rank: 2, numel: metadata.byte_length / 4, logical_dtype: 'float32' },
            subscribe: updates => {
              this.updates = updates;
              if (this.subscribed && (this.receivedData || this.stopped)) this.start();
              this.subscribed = true;
              return () => {
                this.updates = undefined;
                // StrictMode may reattach synchronously before the first DATA.
                // Reuse that untouched transport, but restart any populated prefix.
                queueMicrotask(() => {
                  if (this.updates || this.disposed) return;
                  this.stopped = true;
                  ++this.epoch;
                  void this.handle?.cancel().catch(() => {});
                });
              };
            },
          };
          // Allocate/subscribe before the decoder delivers DATA in this same chunk.
          this.changed({ status: 'streaming', source: this.source }, true);
        },
        onData: (bytes, offset) => { if (current()) { this.receivedData = true; words.push(bytes, offset); } },
      });
      void this.handle.done.then(outcome => {
        if (!current()) return;
        if (outcome.kind === 'complete') this.changed({ status: 'complete', source: this.source });
        else this.changed({ status: outcome.kind === 'cancelled' ? 'cancelled'
          : outcome.error.code === 'unsupported_representation' ? 'unsupported' : 'failed' });
      }).catch(error => { if (current()) this.fail(error); });
    } catch (error) { if (current()) this.fail(error); }
  };

  private fail(error: unknown) {
    this.changed({ status: error instanceof ApiFailure && error.detail?.code === 'unsupported_representation' ? 'unsupported' : 'failed' });
  }

  dispose = () => {
    if (this.disposed) return;
    this.disposed = true;
    ++this.epoch;
    this.signal.removeEventListener('abort', this.dispose);
    void this.handle?.cancel().catch(() => {});
    this.updates = undefined;
    this.source = undefined;
  };
}
