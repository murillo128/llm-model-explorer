/** A selection/request identity. Abort alone cannot fence already queued callbacks. */
export class Lifetime {
  private readonly controller = new AbortController();
  private readonly cleanups = new Set<() => void>();
  readonly generation = Symbol('request-generation');
  readonly signal = this.controller.signal;

  isCurrent = () => !this.signal.aborted;

  guard<Args extends unknown[]>(callback: (...args: Args) => void) {
    return (...args: Args) => { if (this.isCurrent()) callback(...args); };
  }

  /** Register only resources owned by this consumer. Return value unregisters cleanup. */
  onDispose(cleanup: () => void) {
    if (this.signal.aborted) cleanup();
    else this.cleanups.add(cleanup);
    return () => { this.cleanups.delete(cleanup); };
  }

  dispose = () => {
    if (this.signal.aborted) return;
    this.controller.abort();
    // One failed resource cleanup must not prevent releasing the other resources.
    for (const cleanup of this.cleanups) {
      try { cleanup(); } catch { /* Resource owners report their own cleanup failures. */ }
    }
    this.cleanups.clear();
  };
}

/** Child-owned channel: begin fences its previous request, including A → B → A. */
export class RequestChannel {
  private current: Lifetime | undefined;
  private readonly detach: () => void;
  constructor(private readonly selection: Lifetime) {
    this.detach = selection.onDispose(() => this.current?.dispose());
  }
  begin() {
    this.current?.dispose();
    const request = new Lifetime();
    this.current = request;
    if (!this.selection.isCurrent()) request.dispose();
    return request;
  }
  dispose = () => { this.current?.dispose(); this.detach(); };
}
