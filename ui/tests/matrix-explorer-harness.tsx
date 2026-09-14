import { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { MatrixExplorer } from '../src/matrix-explorer';
import type { MatrixCell, MatrixSource, MatrixUpdates } from '../src/matrix-explorer';
import { TensorViewport } from '../src/rendering/tensor-viewport';
import { GridRenderer } from '../src/rendering/tensor-renderer';

// No application, session, model, tensor tree, API or global application stylesheet.
const metrics = { textures: new Set<WebGLTexture>(), displays: new Set<WebGLRenderbuffer>(),
  framebuffers: new Set<WebGLFramebuffer>(), buffers: new Set<WebGLBuffer>(), programs: new Set<WebGLProgram>(),
  uploads: 0, scalarAllocations: 0, integerAllocations: 0, detached: 0 };
const proto = WebGL2RenderingContext.prototype;
for (const [create, remove, live] of [
  ['createTexture', 'deleteTexture', metrics.textures],
  ['createRenderbuffer', 'deleteRenderbuffer', metrics.displays],
  ['createFramebuffer', 'deleteFramebuffer', metrics.framebuffers],
  ['createBuffer', 'deleteBuffer', metrics.buffers],
  ['createProgram', 'deleteProgram', metrics.programs],
] as const) {
  const allocate = proto[create];
  const release = proto[remove];
  Object.defineProperty(proto, create, { value: function (this: WebGL2RenderingContext) {
    const resource = allocate.call(this);
    if (resource) live.add(resource);
    return resource;
  } });
  Object.defineProperty(proto, remove, { value: function (this: WebGL2RenderingContext, resource: WebGLTexture | null) {
    if (resource) live.delete(resource);
    release.call(this, resource);
  } });
}
const storage = proto.texStorage2D;
proto.texStorage2D = function (target, levels, format, width, height) {
  if (format === this.R32F) metrics.scalarAllocations++;
  if (format === this.R32UI) metrics.integerAllocations++;
  storage.call(this, target, levels, format, width, height);
};
const upload = proto.texSubImage2D;
proto.texSubImage2D = function (this: WebGL2RenderingContext, ...args: Parameters<typeof upload>) {
  metrics.uploads++; upload.apply(this, args);
} as typeof upload;
const renderers: GridRenderer[] = [];
const view = GridRenderer.prototype.setView;
GridRenderer.prototype.setView = function (...args) {
  if (!renderers.includes(this)) renderers.push(this);
  return view.apply(this, args);
};
const viewports: TensorViewport[] = [];
const refresh = TensorViewport.prototype.refresh;
TensorViewport.prototype.refresh = function () {
  if (!viewports.includes(this)) viewports.push(this);
  return refresh.call(this);
};
const subscriptions: MatrixUpdates[] = [];
const cells: (MatrixCell | null)[] = [], rows: (number | null)[] = [], columns: (number | null)[] = [];
function source(shape: [number, number], distributions = false): MatrixSource {
  return { descriptor: { shape, rank: 2, numel: shape[0] * shape[1], logical_dtype: 'float32' }, distributions,
    subscribe(updates) {
      subscriptions.push(updates);
      return () => {
        metrics.detached++;
        // Even callbacks triggered by detach itself must already be fenced.
        updates.values(new Float32Array([999]), 0);
      };
    } };
}
const sources = { A: source([3, 5]), B: source([5, 3]), large: source([576, 1536]), full: source([3, 5], true), tall: source([900, 25], true), short: source([4, 128], true), square: source([120, 100], true) };
function Host({ source }: { source: MatrixSource }) {
  const [cell, setCell] = useState<MatrixCell | null>(null);
  return <><div id="workspace"><MatrixExplorer source={source} label="Synthetic result matrix"
    header={<span>Arbitrary result · {source.descriptor.shape.join(' × ')}</span>}
    onCellSelect={(next) => { cells.push(next); setCell(next); }}
    onRowSelect={(next) => rows.push(next)} onColumnSelect={(next) => columns.push(next)} />
    </div><output aria-label="Linked annotation">{cell ? `Row ${cell.row}, cell ${cell.row}:${cell.column}` : 'No selection'}</output></>;
}
const root = createRoot(document.getElementById('root')!);
const strict = new URLSearchParams(location.search).has('strict');
function render(name: keyof typeof sources | null) {
  const content = name ? <Host source={sources[name]} /> : null;
  root.render(strict ? <StrictMode>{content}</StrictMode> : content);
}
function resources() {
  return { textures: metrics.textures.size, displays: metrics.displays.size, framebuffers: metrics.framebuffers.size,
    buffers: metrics.buffers.size, programs: metrics.programs.size,
    cpuBytes: renderers.reduce((sum, r) => sum + r.diagnostics.cpuBytes, 0) };
}
function renderPair() { root.render(<><Host source={sources.square} /><Host source={sources.short} /></>); }
render('A');
window.matrixFixture = { renderPair, viewports, sources, subscriptions, renderers, metrics, cells, rows, columns, render, resources };
declare global { interface Window { matrixFixture: {
  renderPair: typeof renderPair; viewports: typeof viewports; sources: typeof sources; subscriptions: typeof subscriptions; renderers: typeof renderers; metrics: typeof metrics;
  cells: typeof cells; rows: typeof rows; columns: typeof columns; render: typeof render; resources: typeof resources;
} } }
