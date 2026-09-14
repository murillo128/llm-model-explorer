/* eslint-disable @typescript-eslint/no-explicit-any -- Instrument native browser calls without changing their arguments/results. */
/** Test-only observers injected before the production bundle, never shipped. */
export function installProbe() {
  const textures = new Map<WebGLTexture, number>();
  const bound = new WeakMap<WebGL2RenderingContext, WebGLTexture>();
  const snapshots = new WeakMap<HTMLCanvasElement, { width: number; height: number; bytes: Uint8Array }>();
  const arrays: WeakRef<Float32Array | Uint32Array>[] = [];
  const scalarValues: number[] = [];
  const metrics = { readers: 0, peakReaders: 0, createdTextures: 0, peakTextures: 0,
    errors: [] as { code: number; stack: string | undefined }[], contextLosses: 0, peakGpuBytes: 0, uploads: 0, firstUpload: 0, firstRender: 0, maxUploadBytes: 0 };
  const wrap = (name: string, observe: (gl: WebGL2RenderingContext, args: any[], result: any) => void) => {
    const proto = WebGL2RenderingContext.prototype as any;
    const original = proto[name];
    proto[name] = function (...args: any[]) {
      const result = Reflect.apply(original, this, args);
      observe(this, args, result);
      return result;
    };
  };
  window.addEventListener('webglcontextlost', () => { metrics.contextLosses++; }, true);
  wrap('getError', (_gl, _args, result) => { if (result) metrics.errors.push({ code: result, stack: new Error().stack }); });
  wrap('createTexture', (_gl, _args, texture) => {
    textures.set(texture, 0); metrics.createdTextures++;
    metrics.peakTextures = Math.max(metrics.peakTextures, textures.size);
  });
  wrap('deleteTexture', (_gl, [texture]) => { textures.delete(texture); });
  wrap('bindTexture', (gl, [, texture]) => { bound.set(gl, texture); });
  wrap('texStorage2D', (gl, [, levels, , width, height]) => {
    if (levels !== 1) throw new Error('Acceptance expects one scalar mip level');
    textures.set(bound.get(gl)!, width * height * 4);
    metrics.peakGpuBytes = Math.max(metrics.peakGpuBytes, [...textures.values()].reduce((a, b) => a + b, 0));
  });
  wrap('texSubImage2D', (gl, args) => {
    metrics.uploads++;
    metrics.firstUpload ||= performance.now();
    const data = args[8];
    if ((window as any).__acceptance.captureScalars && args[7] === gl.FLOAT) {
      scalarValues.push(...data.subarray(args[9] ?? 0, (args[9] ?? 0) + args[4] * args[5]));
    }
    metrics.maxUploadBytes = Math.max(metrics.maxUploadBytes, data?.byteLength ?? 0);
  });
  wrap('drawArrays', (gl) => {
    if (gl.getParameter(gl.FRAMEBUFFER_BINDING)) return;
    const canvas = gl.canvas as HTMLCanvasElement;
    if (metrics.uploads && canvas.closest('.matrix-scroll')) metrics.firstRender ||= performance.now();
    if (!(window as any).__acceptance.capture) return;
    const width = gl.drawingBufferWidth, height = gl.drawingBufferHeight;
    const bytes = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
    snapshots.set(canvas, { width, height, bytes });
    if (metrics.uploads && canvas.closest('.matrix-scroll')) metrics.firstRender ||= performance.now();
  });
  for (const name of ['Float32Array', 'Uint32Array'] as const) {
    const original = window[name];
    (window as any)[name] = new Proxy(original, {
      construct(target, args) {
        const array = Reflect.construct(target, args);
        // Views are transient; track owning allocations only.
        if (typeof args[0] === 'number' && array.length >= 576) arrays.push(new WeakRef(array));
        return array;
      },
    });
  }
  const getReader = ReadableStream.prototype.getReader;
  (ReadableStream.prototype as any).getReader = function (...args: any[]) {
    const reader = Reflect.apply(getReader, this, args);
    metrics.readers++;
    metrics.peakReaders = Math.max(metrics.peakReaders, metrics.readers);
    const release = reader.releaseLock.bind(reader);
    let released = false;
    reader.releaseLock = () => { if (!released) { released = true; metrics.readers--; } release(); };
    return reader;
  };
  (window as any).__acceptance = {
    capture: true,
    captureScalars: false,
    scalarValues,
    metrics: () => ({ ...metrics, textures: textures.size,
      gpuBytes: [...textures.values()].reduce((a, b) => a + b, 0),
      arrays: arrays.flatMap((ref) => { const a = ref.deref(); return a ? [a.byteLength] : []; }),
    }),
    pixel: (selector: string, x: number, y: number) => {
      const canvas = document.querySelector(selector) as HTMLCanvasElement;
      const snapshot = snapshots.get(canvas)!;
      const offset = ((snapshot.height - 1 - y) * snapshot.width + x) * 4;
      return [...snapshot.bytes.slice(offset, offset + 4)];
    },
    limits: () => {
      const gl = document.querySelector<HTMLCanvasElement>('.matrix-scroll canvas')!.getContext('webgl2')!;
      const extension = gl.getExtension('WEBGL_debug_renderer_info');
      return { dpr: devicePixelRatio, texture: gl.getParameter(gl.MAX_TEXTURE_SIZE),
        renderbuffer: gl.getParameter(gl.MAX_RENDERBUFFER_SIZE), viewport: [...gl.getParameter(gl.MAX_VIEWPORT_DIMS)],
        renderer: extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER) };
    },
  };
}
