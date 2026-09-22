import { useEffect, useRef } from 'react';
import type { Dispatch, RefObject, SetStateAction } from 'react';
import type { Graph, GraphView, Layout } from './graph';
import type { ProjectionOptions } from './projection';
import { requestLayout } from './layout';
import { minimumOverviewScale, overviewScale, visibleBounds } from './overview';

/** A zero-sized mounting viewport is not evidence that the overview is too big. */
function currentViewport(container: RefObject<HTMLDivElement | null>, signal: AbortSignal) {
  return new Promise<{ width: number; height: number }>((resolve, reject) => {
    let frame = 0;
    const cleanup = () => { cancelAnimationFrame(frame); clearTimeout(timeout); signal.removeEventListener('abort', abort); };
    const abort = () => { cleanup(); reject(new DOMException('Cancelled', 'AbortError')); };
    const timeout = setTimeout(() => { cleanup(); reject(new Error('Viewport unavailable')); }, 10_000);
    const attempt = () => {
      const width = container.current?.offsetWidth ?? 0, height = container.current?.offsetHeight ?? 0;
      if (width > 0 && height > 0) { cleanup(); resolve({ width, height }); }
      else frame = requestAnimationFrame(attempt);
    };
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort(); else attempt();
  });
}

export interface LayoutResult {
  layout?: Layout; error?: string; options?: ProjectionOptions; invocation?: number; input?: Graph;
}

/** Keep the worker effect's cleanup outside Canvas's previous-layout closure. */
export function useLayoutRequest(input: { graph: Graph; error?: never } | { error: string; graph?: never }, options: ProjectionOptions,
  retry: number, view: GraphView, rememberAnchor: (id: string) => void,
  setResult: Dispatch<SetStateAction<LayoutResult>>, clearInspection: () => void, container: RefObject<HTMLDivElement | null>) {
  const count = useRef(0);
  useEffect(() => {
    const expansionAnchor = view.takeExpansionAnchor();
    if (expansionAnchor) rememberAnchor(expansionAnchor);
    const controller = new AbortController(); count.current++;
    if (!input.graph) {
      // Failure is local to this optional view; ordinary navigation stays available.
      void Promise.resolve().then(() => { if (!controller.signal.aborted) setResult({ error: input.error, options }); });
      return () => controller.abort();
    }
    void (async () => {
      const layout = await requestLayout(input.graph, options, controller.signal);
      if (view.initialOverview && !view.viewport) {
        const size = await currentViewport(container, controller.signal);
        if (controller.signal.aborted) return;
        if (view.initialOverview && overviewScale(visibleBounds(layout), size.width, size.height)! < minimumOverviewScale) {
          // Do not publish the candidate: exactly one fallback, never a visible snap
          // from full detail or a fit/collapse/retry loop.
          view.update({ initialOverview: false, expanded: [], modelCollapsed: true });
          return;
        }
        view.update({ initialOverview: false });
      }
      if (!controller.signal.aborted) {
        setResult({ layout, options, invocation: count.current, input: input.graph });
        if (view.edge && !layout.projection.edges.some((e) => e.id === view.edge)) {
          view.update({ edge: null }); clearInspection();
        }
      }
    })().catch(() => {
      if (!controller.signal.aborted) setResult({ options, input: input.graph, invocation: count.current,
        error: 'Layout failed or exceeded 10 seconds. Retry or collapse groups.' });
    });
    return () => controller.abort();
  }, [input, options, retry, view, rememberAnchor, setResult, clearInspection, container]);
}
