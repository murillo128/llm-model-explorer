import type { Graph, Layout } from '../src/architecture-explorer/graph';

/** Native worker observations only: no graph/layout copies or production hooks. */
export function installArchitectureProbe() {
  const graphs: WeakRef<Graph>[] = [], layouts: WeakRef<Layout>[] = [];
  const seen = new WeakSet<Graph>();
  let active = 0, created = 0, peak = 0;
  let geometry: { width: number; height: number; nodes: number; edges: number; layoutMs: number } | undefined;
  const Original = window.Worker;
  window.Worker = class extends Original {
    private stopped = false;
    constructor(url: string | URL, options?: WorkerOptions) {
      super(url, options);
      created++; active++; peak = Math.max(peak, active);
      this.addEventListener('message', (event: MessageEvent<{ layout?: Layout }>) => {
        const layout = event.data.layout;
        if (!layout) return;
        layouts.push(new WeakRef(layout));
        geometry = { width: layout.width, height: layout.height, nodes: layout.projection.nodes.length,
          edges: layout.projection.edges.length, layoutMs: layout.milliseconds };
      });
    }
    override postMessage(message: unknown, transfer: Transferable[] | StructuredSerializeOptions = []) {
      if (message && typeof message === 'object' && 'graph' in message) {
        const graph = message.graph as Graph;
        if (!seen.has(graph)) { seen.add(graph); graphs.push(new WeakRef(graph)); }
      }
      if (Array.isArray(transfer)) super.postMessage(message, transfer); else super.postMessage(message, transfer);
    }
    override terminate() {
      if (!this.stopped) { this.stopped = true; active--; }
      super.terminate();
    }
  };
  window.__architectureProbe = () => ({ active, created, peak, geometry,
    observedGraphs: graphs.length, observedLayouts: layouts.length,
    retainedGraphs: graphs.filter((ref) => ref.deref()).length,
    retainedLayouts: layouts.filter((ref) => ref.deref()).length });
}

declare global {
  interface Window {
    __architectureProbe: () => {
      active: number; created: number; peak: number; observedGraphs: number; observedLayouts: number;
      retainedGraphs: number; retainedLayouts: number;
      geometry: { width: number; height: number; nodes: number; edges: number; layoutMs: number } | undefined;
    };
  }
}
