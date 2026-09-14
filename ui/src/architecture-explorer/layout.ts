import type { Graph, Layout } from './graph';

export const layoutTimeoutMs = 10_000;
/** A real worker can be terminated, including when its layout code is stuck. */
export function requestLayout(graph: Graph, expanded: string[], signal: AbortSignal,
  createWorker = () => new Worker(new URL('./layout.worker.ts', import.meta.url), { type: 'module' }), timeout = layoutTimeoutMs): Promise<Layout> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(new DOMException('Cancelled', 'AbortError')); return; }
    const worker = createWorker();
    const finish = (error?: Error, layout?: Layout) => {
      clearTimeout(timer); signal.removeEventListener('abort', abort); worker.terminate();
      if (error) reject(error); else resolve(layout!);
    };
    const abort = () => finish(new DOMException('Cancelled', 'AbortError'));
    const timer = setTimeout(() => finish(new Error('Layout exceeded 10 seconds. Retry or collapse groups.')), timeout);
    signal.addEventListener('abort', abort, { once: true });
    worker.onmessage = (event: MessageEvent<{ layout?: Layout; error?: string }>) => finish(event.data.error ? new Error(event.data.error) : undefined, event.data.layout);
    worker.onerror = () => finish(new Error('The graph layout worker failed. Retry or collapse groups.'));
    try { worker.postMessage({ graph, expanded }); } catch { finish(new Error('The graph could not be sent to layout.')); }
  });
}
