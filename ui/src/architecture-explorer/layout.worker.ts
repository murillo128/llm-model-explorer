import { layoutGraph } from './auto-layout';
import type { Graph } from './graph';
import type { ProjectionOptions } from './projection';
let controller: AbortController | undefined;
self.onmessage = async (event: MessageEvent<{ graph: Graph; options: ProjectionOptions } | { cancel: true }>) => {
  if ('cancel' in event.data) {
    // Terminate ELK before closing its owner. Abruptly terminating an owner alone
    // can leave its descendant worker finishing obsolete computation.
    controller?.abort(); self.postMessage({ cancelled: true }); self.close(); return;
  }
  controller = new AbortController();
  try { self.postMessage({ layout: await layoutGraph(event.data.graph, event.data.options, controller.signal) }); }
  catch (error) { self.postMessage({ error: error instanceof Error ? error.message : 'The graph could not be laid out.' }); }
};
