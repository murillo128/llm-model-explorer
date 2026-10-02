import { layoutGraph } from './auto-layout';
import type { Graph } from './graph';
import type { ProjectionOptions } from './projection';
import { routeDisplays } from './route-display';
let controller: AbortController | undefined;
self.onmessage = async (event: MessageEvent<{ graph: Graph; options: ProjectionOptions } | { cancel: true }>) => {
  if ('cancel' in event.data) {
    // Terminate ELK before closing its owner. Abruptly terminating an owner alone
    // can leave its descendant worker finishing obsolete computation.
    controller?.abort(); self.postMessage({ cancelled: true }); self.close(); return;
  }
  controller = new AbortController();
  try {
    const layout = await layoutGraph(event.data.graph, event.data.options, controller.signal);
    const displays = await routeDisplays(layout, controller.signal);
    if (!controller.signal.aborted) self.postMessage({ layout, displays });
  }
  catch (error) { self.postMessage({ error: error instanceof Error ? error.message : 'The graph could not be laid out.' }); }
};
