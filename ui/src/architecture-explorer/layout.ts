import type { Graph, Layout } from './graph';
import type { ProjectionOptions } from './projection';

export const layoutTimeoutMs = 10_000;
/** A real worker can be terminated, including when its layout code is stuck. */
export function requestLayout(graph: Graph, options: ProjectionOptions, signal: AbortSignal,
  createWorker = () => new Worker(new URL('./layout.worker.ts', import.meta.url), { type: 'module' }), timeout = layoutTimeoutMs): Promise<Layout> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(new DOMException('Cancelled', 'AbortError')); return; }
    const worker = createWorker();
    let finished = false;
    const finish = (error?: Error, layout?: Layout, cancel = false) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer); signal.removeEventListener('abort', abort);
      if (cancel) {
        // Let the owner terminate ELK before reaping it. The caller rejects now;
        // this bounded teardown cannot publish a stale layout or retain a worker.
        const reap = setTimeout(() => worker.terminate(), 1_000);
        const stop = () => { clearTimeout(reap); worker.terminate(); };
        worker.onmessage = stop; worker.onerror = stop;
        try { worker.postMessage({ cancel: true }); } catch { stop(); }
      } else worker.terminate();
      if (error) reject(error); else resolve(layout!);
    };
    const abort = () => finish(new DOMException('Cancelled', 'AbortError'), undefined, true);
    const timer = setTimeout(() => finish(new Error('Layout exceeded 10 seconds. Retry or collapse groups.'), undefined, true), timeout);
    signal.addEventListener('abort', abort, { once: true });
    worker.onmessage = (event: MessageEvent<{ layout?: Layout; error?: string }>) => finish(event.data.error ? new Error(event.data.error) : undefined, event.data.layout);
    worker.onerror = () => finish(new Error('The graph layout worker failed. Retry or collapse groups.'));
    try { worker.postMessage({ graph, options }); } catch { finish(new Error('The graph could not be sent to layout.')); }
  });
}
