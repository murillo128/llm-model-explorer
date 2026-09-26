import type { Page } from '@playwright/test';
import type { ReactFlowInstance, ReactFlowState } from '@xyflow/react';

type Store = { getState: () => ReactFlowState; subscribe: (fn: (state: ReactFlowState, previous: ReactFlowState) => void) => () => void };
export interface CameraSample {
  graph: string | undefined; layout: string | undefined; busy: string | null | undefined;
  actual: number[]; consumed: number[]; nodes: number; measured: number;
  loading: boolean; camera: number[];
}
interface Probe {
  holdSizes: boolean; holdLayout: boolean; holdCompletion: boolean;
  sizes: Map<ResizeObserver, () => void>; layouts: (() => void)[]; completions: (() => void)[];
  store: Store; flow: ReactFlowInstance; events: (CameraSample & { event: string })[];
  sample: () => CameraSample; install: (store: Store, flow: ReactFlowInstance) => void;
}
declare global { interface Window { cameraProbe: Probe } }

/** Observe the real renderer and control notification delivery, without replacing geometry or camera math. */
export async function cameraProbe(page: Page) {
  await page.addInitScript(() => {
    const p = window.cameraProbe = { holdSizes: false, holdLayout: false, holdCompletion: false,
      sizes: new Map(), layouts: [], completions: [], events: [] } as unknown as Probe;
    const stores = new WeakSet<Store>(), flows = new WeakSet<ReactFlowInstance>();
    p.sample = () => {
      const s = p.store.getState(), element = document.querySelector<HTMLElement>('.react-flow__renderer');
      const graph = document.querySelector<HTMLElement>('.architecture-explorer');
      return { graph: graph?.dataset.graphId, layout: graph?.dataset.layoutCount, busy: graph?.getAttribute('aria-busy'),
        actual: [element?.offsetWidth ?? 0, element?.offsetHeight ?? 0], consumed: [s.width, s.height],
        nodes: s.nodes.length, measured: [...s.nodeLookup.values()].filter((n) => n.measured.width && n.measured.height).length,
        loading: [...document.querySelectorAll('[role=status]')].some((e) => e.textContent?.includes('Laying out')),
        camera: [...s.transform] };
    };
    p.install = (store, flow) => {
      p.store = store; p.flow = flow;
      if (!stores.has(store)) {
        stores.add(store);
        store.subscribe((s, previous) => {
          if (s.width !== previous.width || s.height !== previous.height) p.events.push({ event: 'size', ...p.sample() });
        });
      }
      if (!flows.has(flow)) {
        flows.add(flow);
        for (const method of ['fitView', 'setViewport', 'setCenter'] as const) {
          const original = flow[method].bind(flow) as (...args: unknown[]) => Promise<boolean>;
          const wrapped = async (...args: unknown[]) => {
            p.events.push({ event: `${method} requested`, ...p.sample() });
            const result = await original(...args);
            if (p.holdCompletion) await new Promise<void>((resolve) => p.completions.push(resolve));
            p.events.push({ event: `${method} completed`, ...p.sample() });
            return result;
          };
          flow[method] = wrapped;
        }
      }
    };
    const OriginalResize = window.ResizeObserver;
    window.ResizeObserver = class extends OriginalResize {
      constructor(callback: ResizeObserverCallback) {
        super((entries, observer) => {
          if (p.holdSizes && entries.some((e) => e.target.matches('.react-flow__renderer'))) p.sizes.set(observer, () => callback(entries, observer));
          else callback(entries, observer);
        });
      }
    };
    const OriginalWorker = window.Worker;
    window.Worker = class extends OriginalWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener('message', (event: MessageEvent<{ layout?: unknown }>) => {
          if (p.holdLayout && event.data.layout) {
            event.stopImmediatePropagation();
            const callback = this.onmessage;
            p.layouts.push(() => callback?.call(this, event));
          }
        });
      }
    };
  });
  await page.route('**/src/architecture-explorer/ArchitectureCanvas.tsx*', async (route) => {
    const response = await route.fetch(), body = await response.text();
    // Inject a test observer into the existing provider; no alternate canvas or fit implementation.
    const instrumented = body.replace('ReactFlowProvider, useReactFlow }', 'ReactFlowProvider, useReactFlow, useStoreApi }')
      .replace('const flow = useReactFlow();', 'const flow = useReactFlow(); window.cameraProbe.install(useStoreApi(), flow);');
    if (instrumented === body) throw new Error('Camera probe did not bind to the canvas');
    await route.fulfill({ response, body: instrumented });
  });
}

export async function releaseSizes(page: Page) {
  await page.evaluate(() => { const p = window.cameraProbe; p.holdSizes = false; p.sizes.forEach((release) => release()); p.sizes.clear(); });
}
export async function releaseLayouts(page: Page) {
  await page.evaluate(() => { const p = window.cameraProbe; p.holdLayout = false; p.layouts.splice(0).forEach((release) => release()); });
}
export async function releaseCompletions(page: Page) {
  await page.evaluate(() => { const p = window.cameraProbe; p.holdCompletion = false; p.completions.splice(0).forEach((release) => release()); });
}
