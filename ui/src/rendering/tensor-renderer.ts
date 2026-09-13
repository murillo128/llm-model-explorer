import { hitTest, tensorGeometry, viewGeometry } from './geometry';
import type { TensorDescriptor, ViewGeometry } from './geometry';
import { distributionFragmentShader, fragmentShader, vertexShader } from './shaders';
import { transferUniforms } from './transfer';
import type { TransferParameters } from './transfer';

export type RendererState = 'ready' | 'empty' | 'lost' | 'needs-reconstruction' | 'failed' | 'disposed';
export interface RendererOptions {
  /** One CPU array enables exact readout and explicit restoration of received data. */
  readonly retainValues?: boolean;
  /** Optional lower ceilings for resource budgets and deterministic band tests. */
  readonly textureLimit?: number;
  readonly framebufferLimit?: number;
  readonly onStateChange?: (state: RendererState) => void;
}

interface Band {
  x: number;
  y: number;
  width: number;
  height: number;
  texture: WebGLTexture;
}

/** Owns an exclusive WebGL2 canvas. No transport, React, or model dependencies. */
export class GridRenderer<T extends Float32Array | Uint32Array = Float32Array | Uint32Array> {
  readonly geometry;
  readonly limits;
  private readonly gl: WebGL2RenderingContext;
  private readonly options: RendererOptions;
  private readonly textureLimit: number;
  private readonly maxWidth: number;
  private readonly maxHeight: number;
  private bands: Band[] = [];
  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private uniforms: Record<string, WebGLUniformLocation> = {};
  private values: T | null = null;
  private transfer = transferUniforms();
  private _state: RendererState = 'needs-reconstruction';
  private _populated = 0;
  private allocations = 0;
  private uploads = 0;
  private generations = 0;
  private _view: ViewGeometry | null = null;

  protected constructor(readonly canvas: HTMLCanvasElement, descriptor: TensorDescriptor, options: RendererOptions = {}, private readonly densityAxisLength?: number) {
    this.geometry = tensorGeometry(descriptor);
    this.options = options;
    if (densityAxisLength !== undefined && (!Number.isSafeInteger(densityAxisLength) || densityAxisLength < 0)) {
      throw new Error('Density axis length must be a nonnegative integer.');
    }
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, depth: false, stencil: false });
    if (!gl) throw new Error('WebGL2 is unavailable. Enable WebGL2 or retry with a supported browser.');
    this.gl = gl;
    const viewport = gl.getParameter(gl.MAX_VIEWPORT_DIMS) as Int32Array;
    this.limits = Object.freeze({ textureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
      renderbufferSize: gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) as number,
      viewportWidth: viewport[0]!, viewportHeight: viewport[1]! });
    for (const limit of [options.textureLimit, options.framebufferLimit]) {
      if (limit !== undefined && (!Number.isSafeInteger(limit) || limit <= 0)) throw new Error('Resource ceilings must be positive integers.');
    }
    this.textureLimit = Math.min(options.textureLimit ?? Infinity, this.limits.textureSize);
    this.maxWidth = Math.min(options.framebufferLimit ?? Infinity, this.limits.renderbufferSize, this.limits.viewportWidth);
    this.maxHeight = Math.min(options.framebufferLimit ?? Infinity, this.limits.renderbufferSize, this.limits.viewportHeight);
    canvas.addEventListener('webglcontextlost', this.contextLost);
    canvas.addEventListener('webglcontextrestored', this.contextRestored);
    try {
      if (options.retainValues !== false && this.geometry.count > 0) this.values = (densityAxisLength === undefined ? new Float32Array(this.geometry.count) : new Uint32Array(this.geometry.count)) as T;
      this.allocate();
    } catch (error) {
      this.dispose();
      throw error;
    }
  }

  get state() { return this._state; }
  get populatedPrefix() { return this._populated; }
  get view() { return this._view; }
  get diagnostics() {
    return { state: this._state, scalarTextures: this.bands.length,
      scalarBytes: this.bands.reduce((sum, band) => sum + band.width * band.height * 4, 0),
      cpuBytes: this.values?.byteLength ?? 0, totalScalarAllocations: this.allocations,
      scalarUploadCalls: this.uploads, generations: this.generations, limits: this.limits };
  }

  private setState(state: RendererState) {
    this._state = state;
    this.options.onStateChange?.(state);
  }

  private assertReady() {
    if (this.gl.isContextLost() && this._state !== 'disposed') this.contextLost();
    if (this._state !== 'ready' && this._state !== 'empty') {
      throw new Error(`Renderer is ${this._state}; reconstruct after restoration or dispose and retry.`);
    }
  }

  private checkGL(action: string) {
    const error = this.gl.getError();
    if (error !== this.gl.NO_ERROR || this.gl.isContextLost()) {
      throw new Error(`WebGL2 ${action} failed (GL ${error}). Retry or release other GPU resources.`);
    }
  }

  private compile(type: number, source: string) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) throw new Error('WebGL2 shader allocation failed.');
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const message = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`WebGL2 shader compilation failed: ${message}`);
    }
    return shader;
  }

  private allocate() {
    const gl = this.gl;
    try {
      if (this.geometry.count === 0) {
        this.setState('empty');
        return;
      }
      const shaders: WebGLShader[] = [];
      try {
        shaders.push(this.compile(gl.VERTEX_SHADER, vertexShader));
        shaders.push(this.compile(gl.FRAGMENT_SHADER, this.densityAxisLength === undefined ? fragmentShader : distributionFragmentShader));
        this.program = gl.createProgram();
        if (!this.program) throw new Error('WebGL2 program allocation failed.');
        for (const shader of shaders) gl.attachShader(this.program, shader);
        gl.linkProgram(this.program);
        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error(`WebGL2 program linking failed: ${gl.getProgramInfoLog(this.program)}`);
      } finally {
        for (const shader of shaders) {
          if (this.program) gl.detachShader(this.program, shader);
          gl.deleteShader(shader);
        }
      }
      this.vao = gl.createVertexArray();
      if (!this.vao) throw new Error('WebGL2 vertex array allocation failed.');
      for (const name of ['weights', 'bandOffset', 'viewHeight', 'prefix', ...(this.densityAxisLength === undefined ? ['mode', 'slope', 'anchors', 'scale', 'span', 'correction'] : ['densityDenominator'])]) {
        const location = gl.getUniformLocation(this.program!, name);
        if (location === null) throw new Error(`Missing renderer uniform: ${name}`);
        this.uniforms[name] = location;
      }
      const { columns, rows } = this.geometry;
      for (let y = 0; y < rows; y += this.textureLimit) {
        for (let x = 0; x < columns; x += this.textureLimit) {
          const width = Math.min(columns - x, this.textureLimit);
          const height = Math.min(rows - y, this.textureLimit);
          const texture = gl.createTexture();
          if (!texture) throw new Error('WebGL2 scalar texture allocation failed.');
          this.bands.push({ x, y, width, height, texture });
          gl.bindTexture(gl.TEXTURE_2D, texture);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
          gl.texStorage2D(gl.TEXTURE_2D, 1, this.densityAxisLength === undefined ? gl.R32F : gl.R32UI, width, height);
          this.checkGL('scalar allocation');
          this.allocations++;
        }
      }
      this.generations++;
      this.setState('ready');
    } catch (error) {
      this.releaseGPU();
      this.setState(gl.isContextLost() ? 'lost' : 'failed');
      throw error;
    }
  }

  /** Consecutive row-major chunks only. The input view is never retained. */
  upload(chunk: T, offset = this._populated) {
    this.assertReady();
    if (this.densityAxisLength === undefined ? !(chunk instanceof Float32Array) : !(chunk instanceof Uint32Array)) {
      throw new Error('Chunk scalar type does not match the GPU storage.');
    }
    if (offset !== this._populated || offset + chunk.length > this.geometry.count) {
      throw new Error('Tensor chunks must form a consecutive prefix within the declared element count.');
    }
    try {
      this.uploadGPU(chunk, offset);
      this.values?.set(chunk, offset);
      this._populated += chunk.length;
    } catch (error) {
      this.releaseGPU();
      this.setState(this.gl.isContextLost() ? 'lost' : 'failed');
      throw error;
    }
  }

  private uploadGPU(chunk: T, offset: number) {
    const gl = this.gl;
    const { columns } = this.geometry;
    const end = offset + chunk.length;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    for (const band of this.bands) {
      const firstRow = Math.max(band.y, Math.floor(offset / columns));
      const lastRow = Math.min(band.y + band.height, Math.ceil(end / columns));
      gl.bindTexture(gl.TEXTURE_2D, band.texture);
      // Strided full rows go directly from the chunk to a band in one upload.
      // At most two partial edge rows need separate calls; no aggregate copies.
      for (let row = firstRow; row < lastRow;) {
        const start = Math.max(offset, row * columns + band.x);
        const stop = Math.min(end, row * columns + band.x + band.width);
        if (stop <= start) { row++; continue; }
        const fullRow = start === row * columns + band.x && stop === row * columns + band.x + band.width;
        const height = fullRow ? Math.min(lastRow - row,
          Math.floor((end - stop) / columns) + 1) : 1;
        gl.pixelStorei(gl.UNPACK_ROW_LENGTH, height > 1 ? columns : 0);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, start - row * columns - band.x, row - band.y,
          stop - start, height, this.densityAxisLength === undefined ? gl.RED : gl.RED_INTEGER,
          this.densityAxisLength === undefined ? gl.FLOAT : gl.UNSIGNED_INT, chunk, start - offset);
        this.uploads++;
        row += height;
      }
    }
    gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);
    this.checkGL('scalar upload');
  }

  /** Updates small draw state; never allocates or uploads scalar textures. */
  setTransfer(parameters: TransferParameters) {
    if (this._state === 'disposed') throw new Error('Renderer is disposed.');
    this.transfer = transferUniforms(parameters);
  }

  setView(cssWidth: number, cssHeight: number, scrollLeft = 0, scrollTop = 0, dpr = window.devicePixelRatio) {
    this.assertReady();
    const view = viewGeometry(this.geometry, cssWidth, cssHeight, scrollLeft, scrollTop, dpr, this.maxWidth, this.maxHeight);
    // Zero data dimensions get a hidden 1x1 backing buffer, never a zero-sized texture.
    const width = Math.max(1, view.width);
    const height = Math.max(1, view.height);
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    // Chromium rounds a fractional replaced-element CSS size before rasterizing.
    // A DPR-only transform from integer dimensions preserves odd edge pixels.
    this.canvas.style.width = `${view.width}px`;
    this.canvas.style.height = `${view.height}px`;
    this.canvas.style.transformOrigin = '0 0';
    this.canvas.style.transform = `scale(${1 / dpr})`;
    this._view = view;
    if (this.gl.drawingBufferWidth !== width || this.gl.drawingBufferHeight !== height) {
      this.releaseGPU();
      this.setState('failed');
      throw new Error('WebGL2 could not allocate the exact visible framebuffer.');
    }
    return view;
  }

  draw() {
    this.assertReady();
    const view = this._view;
    if (!view) throw new Error('Set the viewport before drawing.');
    if (view.dpr !== window.devicePixelRatio) throw new Error('Device pixel ratio changed. Call setView before drawing again.');
    if (!view.width || !view.height || this._state === 'empty') return;
    const gl = this.gl;
    gl.viewport(0, 0, view.width, view.height);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.DITHER);
    gl.enable(gl.SCISSOR_TEST);
    gl.bindVertexArray(this.vao);
    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    const u = this.uniforms;
    gl.uniform1i(u.weights!, 0);
    gl.uniform1i(u.viewHeight!, view.height);
    if (this.densityAxisLength === undefined) {
      gl.uniform1i(u.mode!, this.transfer.mode);
      gl.uniform1f(u.slope!, this.transfer.slope);
      gl.uniform2f(u.anchors!, this.transfer.low, this.transfer.high);
      gl.uniform1f(u.scale!, this.transfer.scale);
      gl.uniform1f(u.span!, this.transfer.span);
      gl.uniform1f(u.correction!, this.transfer.correction);
    } else gl.uniform1f(u.densityDenominator!, Math.log1p(this.densityAxisLength));
    const prefixRow = Math.floor(this._populated / this.geometry.columns);
    const prefixColumn = this._populated % this.geometry.columns;
    for (const band of this.bands) {
      const left = Math.max(view.x, band.x);
      const top = Math.max(view.y, band.y);
      const right = Math.min(view.x + view.width, band.x + band.width);
      const bottom = Math.min(view.y + view.height, band.y + band.height);
      if (right <= left || bottom <= top) continue;
      gl.scissor(left - view.x, view.height - (bottom - view.y), right - left, bottom - top);
      gl.uniform2i(u.bandOffset!, view.x - band.x, view.y - band.y);
      gl.uniform2i(u.prefix!, Math.max(0, Math.min(band.width, prefixColumn - band.x)),
        Math.max(-1, Math.min(band.height, prefixRow - band.y)));
      gl.bindTexture(gl.TEXTURE_2D, band.texture);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.disable(gl.SCISSOR_TEST);
    try {
      this.checkGL('draw');
    } catch (error) {
      this.releaseGPU();
      this.setState(gl.isContextLost() ? 'lost' : 'failed');
      throw error;
    }
  }

  cellAt(cssX: number, cssY: number) {
    return this._view ? hitTest(this._view, cssX, cssY) : null;
  }

  readCell(row: number, column: number) {
    this.assertReady();
    const { rows, columns } = this.geometry;
    if (!Number.isInteger(row) || !Number.isInteger(column) || row < 0 || row >= rows || column < 0 || column >= columns) return null;
    const index = row * columns + column;
    if (index >= this._populated) return { state: 'pending' as const, row, column };
    if (!this.values) return { state: 'not-retained' as const, row, column };
    const value = this.values[index]!;
    return { state: Number.isFinite(value) ? 'finite' as const : 'nonfinite' as const, row, column, value };
  }

  private contextLost = (event?: Event) => {
    event?.preventDefault();
    if (this._state === 'disposed' || this._state === 'lost') return;
    this.releaseGPU();
    this.setState('lost');
  };

  private contextRestored = () => {
    if (this._state !== 'disposed') this.setState('needs-reconstruction');
  };

  /** Explicit retry. Without CPU retention, restart the consecutive stream at zero. */
  reconstruct() {
    if (this._state !== 'needs-reconstruction' && this._state !== 'failed') {
      throw new Error('Reconstruction requires a restored context or a failed allocation.');
    }
    this.releaseGPU();
    this.allocate();
    try {
      if (this.values) this.uploadGPU(this.values.subarray(0, this._populated) as T, 0);
      else this._populated = 0;
    } catch (error) {
      this.releaseGPU();
      this.setState(this.gl.isContextLost() ? 'lost' : 'failed');
      throw error;
    }
  }

  private releaseGPU() {
    for (const band of this.bands) this.gl.deleteTexture(band.texture);
    this.bands = [];
    if (this.program) this.gl.deleteProgram(this.program);
    if (this.vao) this.gl.deleteVertexArray(this.vao);
    this.program = null;
    this.vao = null;
    this.uniforms = {};
  }

  dispose() {
    if (this._state === 'disposed') return;
    this.canvas.removeEventListener('webglcontextlost', this.contextLost);
    this.canvas.removeEventListener('webglcontextrestored', this.contextRestored);
    this.releaseGPU();
    this.values = null;
    this._populated = 0;
    this._view = null;
    // Release the potentially large visible drawing buffer as well.
    this.canvas.width = 1;
    this.canvas.height = 1;
    this.setState('disposed');
  }
}

/** Float32 logical tensor storage; keeps the original scalar API reusable. */
export class TensorRenderer extends GridRenderer<Float32Array> {
  constructor(canvas: HTMLCanvasElement, descriptor: TensorDescriptor, options: RendererOptions = {}) {
    super(canvas, descriptor, options);
  }
}

/** Exact uint32 counts. Conversion to visual density happens only in the shader. */
export class DistributionRenderer extends GridRenderer<Uint32Array> {
  constructor(canvas: HTMLCanvasElement, shape: readonly [number, number], axisLength: number, options: RendererOptions = {}) {
    super(canvas, { shape, rank: 2, numel: shape[0] * shape[1], logical_dtype: 'float32' }, options, axisLength);
  }
}
