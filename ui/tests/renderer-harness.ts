import { TensorRenderer, TensorViewport } from '../src/rendering';
import type { RendererOptions, TensorDescriptor } from '../src/rendering';

const metrics = { allocations: 0, uploads: 0, deletes: 0, live: new Set<WebGLTexture>(), failAllocation: 0 };
const prototype = WebGL2RenderingContext.prototype;
const originalCreate = prototype.createTexture;
prototype.createTexture = function () {
  const texture = originalCreate.call(this);
  if (texture) metrics.live.add(texture);
  return texture;
};
const originalDelete = prototype.deleteTexture;
prototype.deleteTexture = function (texture) {
  if (texture && metrics.live.delete(texture)) metrics.deletes++;
  originalDelete.call(this, texture);
};
const originalStorage = prototype.texStorage2D;
prototype.texStorage2D = function (target, levels, format, width, height) {
  metrics.allocations++;
  if (metrics.failAllocation === metrics.allocations) format = this.RGBA; // actual GL INVALID_ENUM
  originalStorage.call(this, target, levels, format, width, height);
};
// The overload is preserved on the public prototype; forward all actual arguments unchanged.
const originalUpload = prototype.texSubImage2D;
prototype.texSubImage2D = function (this: WebGL2RenderingContext, ...args: Parameters<typeof originalUpload>) {
  metrics.uploads++;
  originalUpload.apply(this, args);
} as typeof originalUpload;

function descriptor(shape: number[]): TensorDescriptor {
  return { shape, rank: shape.length, numel: shape.reduce((a, b) => a * b, 1), logical_dtype: 'float32' };
}

function create(shape: number[], options: RendererOptions = {}) {
  const canvas = document.createElement('canvas');
  document.body.append(canvas);
  return new TensorRenderer(canvas, descriptor(shape), options);
}

function pixels(renderer: TensorRenderer) {
  renderer.draw();
  const gl = renderer.canvas.getContext('webgl2')!;
  const { width, height } = renderer.canvas;
  const bytes = new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
  const rows: number[][][] = [];
  for (let y = 0; y < height; y++) {
    const row: number[][] = [];
    for (let x = 0; x < width; x++) {
      const start = ((height - y - 1) * width + x) * 4;
      row.push(Array.from(bytes.subarray(start, start + 4)));
    }
    rows.push(row);
  }
  return rows;
}

window.harness = { TensorRenderer, TensorViewport, descriptor, create, pixels, metrics };
declare global {
  interface Window {
    harness: { TensorRenderer: typeof TensorRenderer; TensorViewport: typeof TensorViewport;
      descriptor: typeof descriptor; create: typeof create; pixels: typeof pixels; metrics: typeof metrics };
    renderer: TensorRenderer;
    viewport: TensorViewport;
    loss: WEBGL_lose_context;
  }
}
