import { GridRenderer } from '../src/rendering/tensor-renderer';
import { EditorView } from '@codemirror/view';
import { EditorSelection } from '@codemirror/state';
import { CameraHistory } from '../src/rendering/camera-history';
import { TensorViewport } from '../src/rendering/tensor-viewport';
import { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ApiClient } from '../src/api/client';
import { TokenizerWorkspace } from '../src/tokenizer/TokenizerExplorer';
import { PromptTokenizer } from '../src/tokenizer/PromptTokenizer';
import type { Tokenization } from '../src/tokenizer/annotations';
import '../src/app/styles.css';

interface Pending { session: string; text: string; add_special_tokens: boolean; aborted: boolean }
declare global {
  interface Window {
    embeddingHarness: {
      historyCalls: number;
      transferCalls: number;
      renderers: GridRenderer[];
      viewports: TensorViewport[];
      countRenderers: GridRenderer[];
      requests: { session: string; token_ids: number[]; aborted: boolean }[];
      auxiliary: { kind: 'statistics' | 'distributions'; session: string; token_ids: number[]; aborted: boolean }[];
      auxiliaryHeaders: (index: number, status?: number) => void;
      auxiliarySend: (index: number, bytes: number[], close?: boolean) => void;
      cancelled: string[];
      headers: (index: number, status?: number) => void;
      send: (index: number, bytes: number[], close?: boolean) => void;
    };
    tokenizerHarness: {
      requests: Pending[];
      source: () => string;
      selection: () => number[];
      select: (anchor: number, head: number) => void;
      coords: (position: number) => { x: number; y: number };
      complete: (index: number, data: Tokenization, status?: number) => void;
      fail: (index: number) => void;
    };
  }
}
const resolvers: ((value: Response) => void)[] = [];
const rejecters: ((reason: unknown) => void)[] = [];
window.tokenizerHarness = {
  requests: [],
  source: () => EditorView.findFromDOM(document.querySelector('.cm-content')!)!.state.doc.toString(),
  selection: () => { const selection = EditorView.findFromDOM(document.querySelector('.cm-content')!)!.state.selection.main; return [selection.from, selection.to]; },
  select: (anchor, head) => {
    const view = EditorView.findFromDOM(document.querySelector('.cm-content')!)!;
    view.focus(); view.dispatch({ selection: EditorSelection.single(anchor, head) });
  },
  coords: position => {
    const rect = EditorView.findFromDOM(document.querySelector('.cm-content')!)!.coordsAtPos(position)!;
    return { x: rect.left, y: (rect.top + rect.bottom) / 2 };
  },
  complete: (index, data, status = 200) => resolvers[index]!(new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } })),
  fail: (index) => rejecters[index]!(new Error('simulated late transport failure')),
};
const embeddingResolvers: ((response: Response) => void)[] = [];
const streams: ReadableStreamDefaultController<Uint8Array>[] = [];
const auxiliaryResolvers: ((response: Response) => void)[] = [];
const auxiliaryStreams: ReadableStreamDefaultController<Uint8Array>[] = [];
const renderers: GridRenderer[] = [];
const countRenderers: GridRenderer[] = [];
const setView = GridRenderer.prototype.setView;
GridRenderer.prototype.setView = function (...args) {
  const collection = this.canvas.closest('.matrix-scroll') ? renderers : countRenderers;
  if (!collection.includes(this)) collection.push(this);
  return setView.apply(this, args);
};
window.embeddingHarness = {
  historyCalls: 0,
  transferCalls: 0,
  renderers, countRenderers,
  viewports: [],
  requests: [], cancelled: [], auxiliary: [],
  auxiliaryHeaders(index, status = 200) {
    auxiliaryResolvers[index]!(status === 200 ? new Response(new ReadableStream({ start(controller) { auxiliaryStreams[index] = controller; } }), {
      headers: { 'Content-Type': 'application/vnd.llm-model-explorer.stream', 'X-Operation-Id': `11111111-0000-4000-8000-${String(index).padStart(12, '0')}` },
    }) : new Response(JSON.stringify({ code: 'internal_error', message: 'Analysis fixture failure' }), { status, headers: { 'Content-Type': 'application/json' } }));
  },
  auxiliarySend(index, bytes, close = false) {
    try { auxiliaryStreams[index]!.enqueue(new Uint8Array(bytes)); if (close) auxiliaryStreams[index]!.close(); } catch { /* Deliberate late callbacks after cancellation. */ }
  },
  headers(index, status = 200) {
    embeddingResolvers[index]!(status === 200 ? new Response(new ReadableStream({ start(controller) { streams[index] = controller; } }), {
      headers: { 'Content-Type': 'application/vnd.llm-model-explorer.stream', 'X-Operation-Id': `00000000-0000-4000-8000-${String(index).padStart(12, '0')}` },
    }) : new Response(JSON.stringify({ code: 'unsupported_representation', message: 'Unsupported fixture' }), { status, headers: { 'Content-Type': 'application/json' } }));
  },
  send(index, bytes, close = false) {
    try { streams[index]!.enqueue(new Uint8Array(bytes)); if (close) streams[index]!.close(); } catch { /* A cancelled reader may be closed. */ }
  },
};
const refreshViewport = TensorViewport.prototype.refresh;
TensorViewport.prototype.refresh = function () {
  if (!window.embeddingHarness.viewports.includes(this)) window.embeddingHarness.viewports.push(this);
  return refreshViewport.call(this);
};
const setTransfer = GridRenderer.prototype.setTransfer;
GridRenderer.prototype.setTransfer = function (...args) {
  window.embeddingHarness.transferCalls++;
  return setTransfer.apply(this, args);
};
for (const method of ['record', 'pop', 'clear'] as const) {
  const original = CameraHistory.prototype[method];
  Object.assign(CameraHistory.prototype, { [method]: function (this: CameraHistory, ...args: unknown[]) {
    window.embeddingHarness.historyCalls++;
    return Reflect.apply(original, this, args);
  } });
}
// Use the production typed client, but deliberately ignore abort when resolving
// transport promises so a real browser exercises the generation fence as well.
const client = new ApiClient({ backendBaseUrl: 'https://fixture.example' }, async (input, init) => {
  const path = new URL(String(input)).pathname;
  if (init?.method === 'DELETE') { window.embeddingHarness.cancelled.push(path); return new Response(null, { status: 204 }); }
  if (/\/embeddings\/(statistics|distributions)$/.test(path)) {
    const request = { kind: path.endsWith('/statistics') ? 'statistics' as const : 'distributions' as const,
      session: path.split('/')[2]!, ...JSON.parse(init!.body as string) as { token_ids: number[] }, aborted: false };
    window.embeddingHarness.auxiliary.push(request);
    init?.signal?.addEventListener('abort', () => { request.aborted = true; });
    return new Promise<Response>(resolve => auxiliaryResolvers.push(resolve));
  }
  if (path.endsWith('/embeddings')) {
    const request = { session: path.split('/')[2]!, ...JSON.parse(init!.body as string) as { token_ids: number[] }, aborted: false };
    window.embeddingHarness.requests.push(request);
    init?.signal?.addEventListener('abort', () => { request.aborted = true; });
    return new Promise<Response>(resolve => embeddingResolvers.push(resolve));
  }
  const request: Pending = { session: new URL(String(input)).pathname.split('/')[2]!, ...JSON.parse(init!.body as string), aborted: false };
  window.tokenizerHarness.requests.push(request);
  init?.signal?.addEventListener('abort', () => { request.aborted = true; });
  return new Promise<Response>((resolve, reject) => { resolvers.push(resolve); rejecters.push(reject); });
});
function Harness() {
  const [session, setSession] = useState('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa');
  const [special, setSpecial] = useState(true);
  const [available, setAvailable] = useState(true);
  const [mounted, setMounted] = useState(true);
  const embeddings = new URLSearchParams(location.search).has('embeddings');
  const controls = <div><button onClick={() => setMounted(false)}>Close workspace</button><button onClick={() => setSession((value) => value.startsWith('a') ? 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb' : 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa')}>Change session</button>
    <label><input type="checkbox" checked={special} onChange={(event) => setSpecial(event.target.checked)} />Add special tokens</label>
    <label><input type="checkbox" checked={available} onChange={(event) => setAvailable(event.target.checked)} />Tokenizer available</label></div>;
  const Surface = embeddings ? TokenizerWorkspace : PromptTokenizer;
  return <div className="app-shell"><div>{controls}</div><div className="workspace-frame"><main><section className="working-surface"><div className="surface-content">
    {mounted && <Surface client={client} sessionId={session} addSpecialTokens={special} tokenizerAvailable={available} />}
  </div></section></main></div><footer>Fixture</footer></div>;
}
createRoot(document.getElementById('root')!).render(<StrictMode><Harness /></StrictMode>);
