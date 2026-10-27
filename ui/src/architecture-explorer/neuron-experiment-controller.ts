import type { ApiClient, StreamOperation, StreamOptions } from '../api/client';
import { ApiFailure, requireProtocol } from '../api/errors';
import { StreamWords } from '../explorers/stream-words';
import type { ExperimentMetadata, ExperimentSnapshot } from './neuron-experiment';

export interface ExperimentResult { snapshot: ExperimentSnapshot; metadata: ExperimentMetadata; values: Float32Array; tokenSignal: AbortSignal }
export interface ExperimentState {
  status: 'idle' | 'preparing' | 'running' | 'cancelling' | 'cancelled' | 'complete' | 'failed' | 'unsupported';
  result?: ExperimentResult; message?: string; operationId?: string;
}
export class NeuronExperimentController {
  private state: ExperimentState = { status: 'idle' };
  private listeners = new Set<() => void>();
  private operation: StreamOperation | undefined;
  private epoch = 0;
  private detach: (() => void) | undefined;
  getSnapshot = () => this.state;
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; };
  constructor(private readonly client: ApiClient) {}
  private publish(patch: Partial<ExperimentState>) {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) listener();
  }
  get busy() { return ['preparing', 'running', 'cancelling'].includes(this.state.status); }
  run(snapshot: ExperimentSnapshot, tokenSignal: AbortSignal, ownerSignal: AbortSignal) {
    if (this.busy || tokenSignal.aborted || ownerSignal.aborted) return;
    const epoch = ++this.epoch;
    this.state = { status: 'preparing', ...(this.state.result ? { result: this.state.result } : {}) };
    this.publish({});
    // Capture once; form edits never mutate the request or completed identity.
    const captured = structuredClone(snapshot);
    const controller = new AbortController();
    const current = () => this.epoch === epoch && !controller.signal.aborted && !tokenSignal.aborted && !ownerSignal.aborted;
    const cancel = () => { void this.cancel(); };
    tokenSignal.addEventListener('abort', cancel, { once: true });
    ownerSignal.addEventListener('abort', cancel, { once: true });
    this.detach = () => { controller.abort(); tokenSignal.removeEventListener('abort', cancel); ownerSignal.removeEventListener('abort', cancel); };
    let metadata: ExperimentMetadata | undefined, values: Float32Array | undefined;
    const words = new StreamWords<Float32Array>('float32', (part, offset) => { if (current()) values?.set(part, offset); });
    const options: StreamOptions = { signal: controller.signal,
      onOperationId: id => { if (current()) this.publish({ status: 'running', operationId: id }); },
      onMetadata: data => {
        if (!current()) return;
        requireProtocol(data.kind === 'neuron_activations' || data.kind === 'neuron_comparison', 'Invalid neuron result');
        metadata = data;
        values = new Float32Array(data.byte_length / 4);
        this.publish({ status: 'running' });
      }, onData: (bytes, offset) => { if (current()) words.push(bytes, offset); },
    };
    try {
      const operation = 'conditions' in captured.request
        ? this.client.streamNeuronComparison(captured.session, captured.capabilities, captured.request, options)
        : this.client.streamNeuronActivations(captured.session, captured.capabilities, captured.request, options);
      this.operation = operation;
      void operation.done.then(outcome => {
        if (!current()) return;
        if (outcome.kind === 'complete') {
          requireProtocol(metadata && values, 'Missing neuron result');
          this.publish({ status: 'complete', result: { snapshot: captured, metadata, values, tokenSignal } });
        } else if (outcome.kind === 'cancelled') this.publish({ status: 'cancelled' });
        else this.fail(new ApiFailure('backend', 'Neuron execution failed', undefined, outcome.error));
      }).catch(error => { if (current()) this.fail(error); }).finally(() => {
        if (this.epoch === epoch) { this.detach?.(); this.detach = undefined; this.operation = undefined; }
      });
    } catch (error) { this.fail(error); this.detach?.(); this.detach = undefined; }
  }
  private fail(error: unknown) {
    const code = error instanceof ApiFailure ? error.detail?.code : undefined;
    const message = code === 'model_content_changed' ? 'Model content changed. Start a fresh session before running again.'
      : code === 'resource_exhausted' ? 'Insufficient execution resources. Reduce the prompt or generation limit and retry.'
      : code === 'validation_error' || code === 'unsupported_size' ? 'The server rejected this configuration. Check the effective limits and retry.'
      : code === 'unsupported_representation' ? 'Neuron execution is unavailable for this configuration. Review the capability information.'
      : 'Execution failed or returned an incomplete or invalid result. Check the connection and run again.';
    this.publish({ status: code === 'unsupported_representation' ? 'unsupported' : 'failed', message });
  }
  cancel = async () => {
    if (!this.busy || this.state.status === 'cancelling') return;
    const epoch = ++this.epoch, operation = this.operation;
    this.publish({ status: 'cancelling' });
    // cancel() owns the explicit operation DELETE; detach aborts this consumer.
    const cancelled = operation?.cancel();
    this.detach?.(); this.detach = undefined; this.operation = undefined;
    try { await cancelled; }
    catch { /* The stream is released even if the best-effort DELETE cannot reach the server. */ }
    if (epoch === this.epoch) this.publish({ status: 'cancelled' });
  };
  dispose = () => { void this.cancel(); ++this.epoch; this.detach?.(); this.listeners.clear(); };
}
