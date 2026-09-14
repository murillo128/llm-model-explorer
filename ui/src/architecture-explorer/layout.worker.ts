import { layoutGraph } from './graph';
import type { Graph } from './graph';
self.onmessage = (event: MessageEvent<{ graph: Graph; expanded: string[] }>) => {
  try { self.postMessage({ layout: layoutGraph(event.data.graph, event.data.expanded) }); }
  catch { self.postMessage({ error: 'The graph could not be laid out.' }); }
};
