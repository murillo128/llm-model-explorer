import { useLayoutEffect, useRef, useState } from 'react';
import { getViewportForBounds, useStoreApi } from '@xyflow/react';
import type { FitViewOptions, Node, ReactFlowInstance } from '@xyflow/react';
import type { RefObject } from 'react';
import type { Layout } from './graph';
import { useCanvasCallback } from './useCanvasCallback';

/** Commit a layout's camera only after the renderer has consumed its DOM geometry. */
export function useLayoutCamera<N extends Node>(layout: Layout | undefined, generation: object, current: boolean,
  nodes: N[], container: RefObject<HTMLDivElement | null>, flow: ReactFlowInstance<N>,
  apply: (fit: (options: FitViewOptions<N>) => Promise<boolean>, size: { width: number; height: number }) => Promise<unknown>, complete: () => void) {
  const store = useStoreApi<N>();
  const [settled, setSettled] = useState<{ layout: Layout; error?: string }>();
  const cancelled = useRef<object | undefined>(undefined);
  const work = useRef<{ layout: Layout; generation: object; deadline: number; promise?: Promise<string | undefined> }>(undefined);
  const run = useCanvasCallback((fit: (options: FitViewOptions<N>) => Promise<boolean>, size: { width: number; height: number }, finish: (error?: string) => void) => {
    void apply(fit, size).then(() => finish(), () => finish('Camera initialization failed. Retry or return to the previous view.'));
  });
  const done = useCanvasCallback(complete);
  const cancel = useCanvasCallback(() => { cancelled.current = generation; });
  useLayoutEffect(() => {
    if (!layout || !current || settled?.layout === layout) return;
    if (work.current?.layout !== layout || work.current.generation !== generation)
      work.current = { layout, generation, deadline: performance.now() + 10_000 };
    const job = work.current;
    let active = true, frame = 0;
    const finish = (error?: string) => {
      if (!active) return;
      active = false; clearTimeout(timeout); if (!error) done(); setSettled({ layout, ...(error ? { error } : {}) });
    };
    const timeout = setTimeout(() => {
      cancelAnimationFrame(frame);
      finish('Camera initialization could not use the current viewport. Retry or return to the previous view.');
    }, Math.max(0, job.deadline - performance.now()));
    const attempt = () => {
      if (!active) return;
      if (cancelled.current === generation) { finish(); return; }
      const state = store.getState(), element = container.current;
      // ELK supplies the complete fixed box geometry, including culled nodes.
      // Waiting for culled cards to mount would deadlock a scope outside the old camera.
      // Use authored width/height; supplying `measured` would retain stale handle
      // bounds when a projection changes ports without changing the card size.
      const usable = element && flow.viewportInitialized && state.width > 0 && state.height > 0 &&
        state.width === element.offsetWidth && state.height === element.offsetHeight &&
        state.nodes.length === nodes.length && state.nodes.every((node, i) => node === nodes[i] &&
          state.nodeLookup.get(node.id)?.width === node.width && state.nodeLookup.get(node.id)?.height === node.height);
      if (!usable) { frame = requestAnimationFrame(attempt); return; }
      // fitView queues work inside React Flow's next node update. Compute the same
      // fixed-node bounds/transform here so an obsolete queued fit cannot act
      // on a replacement layout. setViewport applies a zero-duration transform.
      const fit = (options: FitViewOptions<N>) => {
        const ids = options.nodes && new Set(options.nodes.map((node) => node.id));
        const targets = state.nodes.filter((node) => {
          const internal = state.nodeLookup.get(node.id)!;
          return internal.width && internal.height && !node.hidden && (!ids || ids.has(node.id));
        });
        return flow.setViewport(getViewportForBounds(flow.getNodesBounds(targets), state.width, state.height,
          options.minZoom ?? state.minZoom, options.maxZoom ?? state.maxZoom, options.padding ?? 0.1));
      };
      // Selection can replace React Flow node objects while completion is still
      // pending. Rejoin this generation's action instead of fitting twice or
      // extending its deadline. Obsolete effects can never commit completion.
      job.promise ??= new Promise((resolve) => run(fit, { width: state.width, height: state.height }, resolve));
      void job.promise.then(finish);
    };
    frame = requestAnimationFrame(attempt);
    return () => { active = false; clearTimeout(timeout); cancelAnimationFrame(frame); };
  }, [layout, generation, current, nodes, container, flow, store, settled, run, done]);
  return { ready: Boolean(layout && settled?.layout === layout),
    error: settled?.layout === layout ? settled?.error : undefined, cancel };
}
