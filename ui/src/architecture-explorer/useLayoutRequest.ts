import { useEffect, useRef } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { Graph, GraphView, Layout } from './graph';
import type { ProjectionOptions } from './projection';
import { requestLayout } from './layout';

export interface LayoutResult {
  layout?: Layout; error?: string; options?: ProjectionOptions; invocation?: number; input?: Graph;
}

/** Keep the worker effect's cleanup outside Canvas's previous-layout closure. */
export function useLayoutRequest(input: { graph: Graph; error?: never } | { error: string; graph?: never }, options: ProjectionOptions,
  retry: number, view: GraphView, rememberAnchor: (id: string) => void,
  setResult: Dispatch<SetStateAction<LayoutResult>>, clearInspection: () => void) {
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
    void requestLayout(input.graph, options, controller.signal).then((layout) => {
      if (!controller.signal.aborted) {
        setResult({ layout, options, invocation: count.current, input: input.graph });
        if (view.edge && !layout.projection.edges.some((e) => e.id === view.edge)) {
          view.update({ edge: null }); clearInspection();
        }
      }
    }, () => {
      if (!controller.signal.aborted) setResult({ options, input: input.graph, invocation: count.current,
        error: 'Layout failed or exceeded 10 seconds. Retry or collapse groups.' });
    });
    return () => controller.abort();
  }, [input, options, retry, view, rememberAnchor, setResult, clearInspection]);
}
