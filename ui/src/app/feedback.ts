import type { ActivityObserver } from '../api/client';
import { ApiFailure } from '../api/errors';
import type { Lifetime } from './lifetime';

export type Connection = 'connecting' | 'connected' | 'reconnecting' | 'disconnected';
export interface Toast {
  id: number;
  message: string;
  severity: 'error' | 'info';
  action?: { label: string; run: () => void } | undefined;
}
export interface FeedbackState { connection: Connection; toasts: Toast[] }

/** Presentation only: never owns a stream, retries work, or replaces a session. */
export class Feedback {
  private nextId = 0;
  private epoch = 0;
  private observed: 'unknown' | 'reachable' | 'unreachable' = 'unknown';
  private pending = new Set<symbol>();
  private state: FeedbackState = { connection: 'connecting', toasts: [] };
  constructor(private readonly publish: (state: FeedbackState) => void) {}
  private update(patch: Partial<FeedbackState>) {
    this.state = { ...this.state, ...patch };
    this.publish(this.state);
  }
  private connection() {
    this.update({ connection: this.observed === 'unknown' ? 'connecting' : this.observed === 'reachable' ? 'connected'
      : this.pending.size ? 'reconnecting'
      : 'disconnected' });
  }
  reset() {
    this.epoch++;
    this.pending.clear();
    this.update({ toasts: [] });
    this.connection();
  }
  notify(message: string, severity: Toast['severity'], action?: Toast['action']) {
    this.update({ toasts: [...this.state.toasts, { id: ++this.nextId, message, severity, action }].slice(-3) });
  }
  dismiss = (id: number) => this.update({ toasts: this.state.toasts.filter((toast) => toast.id !== id) });
  observe(lifetime: Lifetime, consequential = false, backendScoped = false): ActivityObserver {
    const epoch = this.epoch;
    const requests = new Set<symbol>();
    const delivered = new Set<symbol>();
    lifetime.onDispose(() => {
      for (const id of requests) this.pending.delete(id);
      if (backendScoped || epoch === this.epoch) this.connection();
    });
    return ({ id, phase, failure, quiet }) => {
      if ((!backendScoped && epoch !== this.epoch) || !lifetime.isCurrent()) return;
      if (phase === 'start') {
        requests.add(id);
        if (this.observed !== 'reachable') this.pending.add(id);
      }
      if (phase === 'response') this.observed = 'reachable';
      if (phase === 'end') {
        requests.delete(id);
        this.pending.delete(id);
        if (failure?.kind === 'transport') {
          this.observed = 'unreachable';
          this.pending.clear();
          if (consequential && !quiet && !delivered.has(id)) {
            delivered.add(id);
            this.notify('The operation could not be completed because the connection was lost.', 'error');
          }
        }
      }
      this.connection();
    };
  }
  /** Controller requests retain their existing injected client and cleanup semantics. */
  track<T>(lifetime: Lifetime, operation: () => Promise<T>): Promise<T> {
    const observe = this.observe(lifetime, false, true);
    const id = Symbol('shell-request');
    observe({ id, phase: 'start' });
    return operation().then((result) => {
      observe({ id, phase: 'response' });
      observe({ id, phase: 'end' });
      return result;
    }, (error: unknown) => {
      if (error instanceof ApiFailure && error.kind !== 'transport' && error.kind !== 'cancelled') observe({ id, phase: 'response' });
      observe({ id, phase: 'end', failure: error instanceof ApiFailure ? error : undefined });
      throw error;
    });
  }
}
