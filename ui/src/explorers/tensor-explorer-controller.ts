import type { StreamOperation, StreamOptions } from '../api/client';
import { requireProtocol } from '../api/errors';
import type { Metadata } from '../api/validation';
import type { ExplorerContextValue } from '../app/explorer-context';
import { Lifetime } from '../app/lifetime';
import { MatrixViewport } from '../rendering/matrix-viewport';
import type { RendererOptions } from '../rendering/tensor-renderer';
import { StreamWords } from './stream-words';

export type ResultState = 'loading' | 'streaming' | 'complete' | 'failed' | 'cancelled' | 'unneeded';
export interface ExplorerStatus {
  tensor: ResultState;
  statistics: ResultState;
  distributions: ResultState;
  rendering: 'ready' | 'failed';
}
type Result = 'tensor' | 'statistics' | 'distributions';

/** One selected descriptor and its three independently owned operation handles. */
export class TensorExplorerController {
  readonly viewport: MatrixViewport;
  private readonly lifetime = new Lifetime();
  private readonly handles = new Map<Result, StreamOperation>();
  private readonly detach: () => void;
  private cancelled = false;
  private status: ExplorerStatus = { tensor: 'loading', statistics: 'loading', distributions: 'loading', rendering: 'ready' };

  constructor(host: HTMLElement, context: ExplorerContextValue, private readonly changed: (status: ExplorerStatus) => void, options: RendererOptions = {}) {
    const tensor = context.selectedTensor!;
    this.viewport = new MatrixViewport(host, tensor, { ...options, onStateChange: (state) => {
      options.onStateChange?.(state);
      if (['failed', 'lost', 'needs-reconstruction'].includes(state)) {
        this.status = { ...this.status, rendering: 'failed' };
        this.changed(this.status);
      }
    } });
    this.detach = context.selection.onDispose(() => this.dispose());
    if (!this.lifetime.isCurrent()) return;
    if (tensor.rank !== 2) this.update('distributions', 'unneeded');
    const current = () => this.lifetime.isCurrent() && context.selection.isCurrent() && !this.cancelled;
    const scalar = new StreamWords<Float32Array>('float32', (values, offset) => {
      this.viewport.matrix.renderer.upload(values, offset);
      this.viewport.refresh();
      this.update('tensor', 'streaming');
    });
    let distributions: Extract<Metadata, { kind: 'tensor_distributions' }> | undefined;
    const counts = new StreamWords<Uint32Array>('uint32', (values, offset) => {
      requireProtocol(distributions, 'Missing distribution metadata');
      // Metadata offsets and section shapes preserve the two different C-order layouts.
      for (const section of distributions.sections) {
        const start = section.offset / 4;
        const from = Math.max(offset, start);
        const to = Math.min(offset + values.length, start + section.byte_length / 4);
        if (to <= from) continue;
        const renderer = section.name === 'row_counts' ? this.viewport.rows! : this.viewport.columns!;
        renderer.upload(values.subarray(from - offset, to - offset), from - start);
      }
      this.viewport.refresh();
      this.update('distributions', 'streaming');
    });
    const start = (result: Result, run: (options: StreamOptions) => StreamOperation) => {
      const handle = run({
        onMetadata: (metadata) => {
          if (!current()) return;
          if (metadata.kind === 'tensor') {
            requireProtocol(metadata.shape.length === tensor.rank && metadata.shape.every((n, i) => n === tensor.shape[i]), 'Tensor descriptor/stream shape mismatch');
          } else if (metadata.kind === 'tensor_statistics') {
            requireProtocol(metadata.count === tensor.numel, 'Statistics descriptor/count mismatch');
          } else {
            requireProtocol(metadata.rows === tensor.shape[0] && metadata.columns === tensor.shape[1], 'Distribution descriptor/shape mismatch');
            distributions = metadata;
          }
        },
        onData: (bytes, offset) => {
          if (!current()) return;
          if (result === 'tensor') scalar.push(bytes, offset);
          if (result === 'distributions') counts.push(bytes, offset);
        },
      });
      this.handles.set(result, handle);
      void handle.done.then((outcome) => {
        if (!current()) return;
        if (outcome.kind === 'complete' && result === 'statistics') {
          requireProtocol(outcome.metadata.kind === 'tensor_statistics', 'Unexpected statistics result');
          this.viewport.matrix.renderer.setTransfer({ statistics: outcome.metadata });
          this.viewport.refresh();
        }
        this.update(result, outcome.kind === 'backend' ? 'failed' : outcome.kind);
      }).catch(() => {
        if (current()) this.update(result, 'failed');
      });
    };
    try {
      start('tensor', (options) => context.client.streamTensor(context.sessionId, tensor.id, options));
      start('statistics', (options) => context.client.streamTensorStatistics(context.sessionId, tensor.id, options));
      if (tensor.rank === 2) start('distributions', (options) => context.client.streamTensorDistributions(context.sessionId, tensor.id, options));
    } catch (error) {
      this.dispose();
      throw error;
    }
    this.changed(this.status);
  }

  private update(result: Result, state: ResultState) {
    this.status = { ...this.status, [result]: state };
    this.changed(this.status);
  }

  cancel = () => {
    this.cancelled = true;
    for (const [result, handle] of this.handles) {
      if (['loading', 'streaming'].includes(this.status[result])) this.update(result, 'cancelled');
      void handle.cancel().catch(() => { /* Local cancellation is already observable. */ });
    }
  };

  dispose = () => {
    if (!this.lifetime.isCurrent()) return;
    this.lifetime.dispose();
    this.detach?.();
    for (const handle of this.handles.values()) void handle.cancel().catch(() => {});
    this.handles.clear();
    this.viewport.dispose();
  };
}
