import { createRoot } from 'react-dom/client';
import { App } from '../src/app/App';
import { ApiClient, streamMediaType } from '../src/api/client';
import type { StreamOptions } from '../src/api/client';
import type { TensorDescriptor } from '../src/app/session-controller';
import { DistributionRenderer, GridRenderer } from '../src/rendering/tensor-renderer';
import type { Metadata } from '../src/api/validation';
import { models, sessionA } from '../src/test/shell-fixtures';
import '../src/app/styles.css';

const tensors: TensorDescriptor[] = [
  ['distribution-outliers', [2, 100]], ['inspection', [17, 19]], ['A', [2, 3]], ['B', [3, 2]], ['reference', [576, 1536]], ['vector', [5]], ['wide-vector', [1536]], ['short-matrix', [2, 1536]], ['empty', [0, 3]], ['unsupported', [2, 2, 2]],
].map(([name, dimensions]) => {
  const shape = dimensions as number[];
  return { id: name as string, name: name as string, path: [name as string], shape, rank: shape.length, numel: shape.reduce((a, b) => a * b, 1), storage_dtype: 'float32', logical_dtype: 'float32' };
});
const metrics = { textureCreates: 0, float32Allocations: 0, fetches: 0, displayAllocations: 0, liveDisplays: new Set<WebGLRenderbuffer>(), scalarAllocations: 0, integerAllocations: 0, uploads: 0, live: new Set<WebGLTexture>(), failAllocation: false, bandLimit: Infinity };
const OriginalFloat32Array = Float32Array;
window.Float32Array = new Proxy(OriginalFloat32Array, { construct(target, args, newTarget) {
  metrics.float32Allocations++;
  return Reflect.construct(target, args, newTarget);
} });
const renderers: GridRenderer[] = [];
const proto = WebGL2RenderingContext.prototype;
const createDisplay = proto.createRenderbuffer;
proto.createRenderbuffer = function () {
  const resource = createDisplay.call(this);
  if (resource) { metrics.displayAllocations++; metrics.liveDisplays.add(resource); }
  return resource;
};
const deleteDisplay = proto.deleteRenderbuffer;
proto.deleteRenderbuffer = function (resource) {
  if (resource) metrics.liveDisplays.delete(resource);
  deleteDisplay.call(this, resource);
};
const storage = proto.texStorage2D;
proto.texStorage2D = function (target, levels, format, width, height) {
  if (format === this.R32F) metrics.scalarAllocations++;
  if (format === this.R32UI) metrics.integerAllocations++;
  storage.call(this, target, levels, metrics.failAllocation ? this.RGBA : format, width, height);
};
const parameter = proto.getParameter;
proto.getParameter = function (name: number) { return name === this.MAX_TEXTURE_SIZE ? Math.min(parameter.call(this, name), metrics.bandLimit) : parameter.call(this, name); };
const create = proto.createTexture;
proto.createTexture = function () { const texture = create.call(this); if (texture) { metrics.textureCreates++; metrics.live.add(texture); } return texture; };
const remove = proto.deleteTexture;
proto.deleteTexture = function (texture) { if (texture) metrics.live.delete(texture); remove.call(this, texture); };
const upload = proto.texSubImage2D;
proto.texSubImage2D = function (this: WebGL2RenderingContext, ...args: Parameters<typeof upload>) {
  metrics.uploads++; upload.apply(this, args);
} as typeof upload;
const setView = GridRenderer.prototype.setView;
GridRenderer.prototype.setView = function (...args) {
  if (!renderers.includes(this)) renderers.push(this);
  return setView.apply(this, args);
};
const callbacks: StreamOptions[] = [];
for (const name of ['streamTensor', 'streamTensorStatistics', 'streamTensorDistributions'] as const) {
  const original = ApiClient.prototype[name];
  ApiClient.prototype[name] = function (session, tensor, options = {}) {
    callbacks.push(options);
    return original.call(this, session, tensor, options);
  };
}
interface Request { id: string; tensor: string; kind: string; stream: ReadableStreamDefaultController<Uint8Array> }
const requests: Request[] = [];
const cancelled: string[] = [];
let releaseCatalogue: (() => void) | undefined;
const catalogue = { paused: false, resume() { this.paused = false; releaseCatalogue?.(); } };
const originalFetch = window.fetch.bind(window);
window.fetch = async (input, options) => {
  metrics.fetches++;
  const url = String(input);
  if (!url.startsWith('https://fixture.example')) return originalFetch(input, options);
  const path = new URL(url).pathname;
  const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
  if (options?.method === 'DELETE') { cancelled.push(path.split('/').at(-1)!); return new Response(null, { status: 204 }); }
  if (path === '/models') {
    if (catalogue.paused) await new Promise<void>((resolve) => { releaseCatalogue = resolve; });
    return json({ models });
  }
  if (path === '/sessions') return json(sessionA, 201);
  if (path.endsWith('/tensors')) return json({ tensors });
  if (path === `/sessions/${sessionA.id}`) return json(sessionA);
  const id = `aaaaaaaa-aaaa-4aaa-8aaa-${String(requests.length + 1).padStart(12, '0')}`;
  const body = new ReadableStream<Uint8Array>({ start(stream) { requests.push({ id, tensor: path.split('/').at(-2)!, kind: path.split('/').at(-1)!, stream }); } });
  return new Response(body, { headers: { 'Content-Type': streamMediaType, 'X-Operation-Id': id } });
};
function frame(type: number, payload: Uint8Array = new Uint8Array(0)) {
  const bytes = new Uint8Array(12 + payload.length);
  bytes.set([76, 77, 69, 88, type]);
  new DataView(bytes.buffer).setUint32(8, payload.length, true);
  bytes.set(payload, 12);
  return bytes;
}
function emit(index: number, type: number, payload?: unknown, split = 0) {
  const bytes = frame(type, payload instanceof Uint8Array ? payload : payload === undefined ? undefined : new TextEncoder().encode(JSON.stringify(payload)));
  const stream = requests[index]!.stream;
  if (split) { for (let i = 0; i < bytes.length; i += split) stream.enqueue(bytes.slice(i, i + split)); }
  else stream.enqueue(bytes);
}
function end(index: number, type = 4) { emit(index, type); requests[index]!.stream.close(); }
function metadata(index: number): Metadata {
  const request = requests[index]!;
  const tensor = tensors.find((t) => t.id === request.tensor)!;
  if (request.kind === 'data') return { kind: 'tensor', tensor_id: tensor.id, name: tensor.name, shape: tensor.shape, dtype: 'float32', byte_order: 'little', layout: 'c', byte_length: tensor.numel * 4 };
  if (request.kind === 'statistics') return { kind: 'tensor_statistics', tensor_id: tensor.id, count: tensor.numel, finite_count: tensor.numel, non_finite_count: 0, minimum: -2, maximum: 2, mean: 0, stddev: 1, percentiles: { p01: -2, p05: -1, p50: 0, p95: 1, p99: 2 }, byte_length: 0 };
  const [rows, columns] = tensor.shape as [number, number];
  return { kind: 'tensor_distributions', tensor_id: tensor.id, rows, columns, bin_count: 100, binning: 'linear-full-range', domain_minimum: -2, domain_maximum: 2, dtype: 'uint32', byte_order: 'little', sections: [
    { name: 'row_counts', shape: [rows, 100], offset: 0, byte_length: rows * 400 },
    { name: 'column_counts', shape: [100, columns], offset: rows * 400, byte_length: columns * 400 },
  ], byte_length: (rows + columns) * 400 };
}
function data(index: number, values: number[], split = 0) {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, i) => requests[index]!.kind === 'data' ? view.setFloat32(i * 4, value, true) : view.setUint32(i * 4, value, true));
  emit(index, 2, bytes, split);
}
function pixels(renderer: GridRenderer) {
  renderer.draw();
  const gl = renderer.canvas.getContext('webgl2')!;
  const { width, height } = renderer.canvas;
  const bytes = new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
  return Array.from({ length: height }, (_, y) => Array.from({ length: width }, (_, x) => Array.from(bytes.slice(((height - 1 - y) * width + x) * 4, ((height - 1 - y) * width + x) * 4 + 4))));
}
const root = createRoot(document.getElementById('root')!);
root.render(<App config={{ backendBaseUrl: 'https://fixture.example' }} />);
window.explorerFixture = { catalogue, DistributionRenderer, tensors, metrics, renderers, requests, cancelled, callbacks, emit, end, metadata, data, pixels, unmount: () => root.unmount() };
declare global { interface Window { explorerFixture: {
  catalogue: typeof catalogue; DistributionRenderer: typeof DistributionRenderer; tensors: typeof tensors; metrics: typeof metrics; renderers: typeof renderers; requests: typeof requests;
  cancelled: typeof cancelled; callbacks: typeof callbacks; emit: typeof emit; end: typeof end;
  metadata: typeof metadata; data: typeof data; pixels: typeof pixels; unmount: () => void;
} } }
