/** The logical subset of the API descriptor needed by the renderer. */
export interface TensorDescriptor {
  readonly shape: readonly number[];
  readonly rank: number;
  readonly numel: number;
  readonly logical_dtype: 'float32';
}

export interface TensorGeometry {
  readonly columns: number;
  readonly rows: number;
  readonly count: number;
}

export function tensorGeometry(descriptor: TensorDescriptor): TensorGeometry {
  const { shape, rank, numel, logical_dtype } = descriptor;
  if (logical_dtype !== 'float32' || (rank !== 1 && rank !== 2) || rank !== shape.length ||
      shape.some((size) => !Number.isSafeInteger(size) || size < 0)) {
    throw new Error('Rendering requires a valid rank-1 or rank-2 float32 tensor.');
  }
  const columns = shape[rank - 1]!;
  const rows = rank === 1 ? 1 : shape[0]!;
  const count = columns * rows;
  if (!Number.isSafeInteger(count * 4) || count !== numel) {
    throw new Error('Tensor element/byte count is inconsistent or exceeds safe integer precision.');
  }
  return Object.freeze({ columns, rows, count });
}

export interface ViewGeometry {
  readonly dpr: number;
  readonly scaleX: number;
  readonly scaleY: number;
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  readonly cssWidth: number;
  readonly cssHeight: number;
  readonly scrollWidth: number;
  readonly scrollHeight: number;
}

/** Origins are logical (possibly fractional); sizes are integer device pixels.
 * Independent axis scales serve profiles; the matrix always supplies equal scales. */
export function viewGeometry(tensor: TensorGeometry, cssWidth: number, cssHeight: number,
  scrollLeft: number, scrollTop: number, dpr: number, maxWidth: number, maxHeight: number,
  scaleX = 1, scaleY = scaleX): ViewGeometry {
  if (![cssWidth, cssHeight, scrollLeft, scrollTop, dpr, scaleX, scaleY].every(Number.isFinite) ||
      dpr <= 0 || cssWidth < 0 || cssHeight < 0 || scaleX < 1 || scaleY < 1) throw new Error('Invalid viewport geometry.');
  const width = Math.min(Math.floor(tensor.columns * scaleX), maxWidth, Math.floor(cssWidth * dpr));
  const height = Math.min(Math.floor(tensor.rows * scaleY), maxHeight, Math.floor(cssHeight * dpr));
  const x = Math.max(0, Math.min(tensor.columns * scaleX - width, Math.round(scrollLeft * dpr))) / scaleX;
  const y = Math.max(0, Math.min(tensor.rows * scaleY - height, Math.round(scrollTop * dpr))) / scaleY;
  return Object.freeze({ dpr, scaleX, scaleY, x, y, width, height, cssWidth: width / dpr, cssHeight: height / dpr,
    scrollWidth: tensor.columns * scaleX / dpr, scrollHeight: tensor.rows * scaleY / dpr });
}

export function hitTest(view: ViewGeometry, cssX: number, cssY: number) {
  if (!Number.isFinite(cssX) || !Number.isFinite(cssY) || cssX < 0 || cssY < 0 ||
      cssX >= view.cssWidth || cssY >= view.cssHeight) return null;
  return { row: rasterCell(view.y, view.scaleY, Math.floor(cssY * view.dpr)),
    column: rasterCell(view.x, view.scaleX, Math.floor(cssX * view.dpr)) };
}

export function fitWidthScale(columns: number, cssWidth: number, dpr: number) {
  return columns > 0 ? Math.max(1, Math.floor(cssWidth * dpr) / columns) : 1;
}

/** Presentation only: center an underfilled axis on whole device pixels. */
export function centeredOffset(availableCSS: number, dataPixels: number, dpr: number) {
  return Math.max(0, Math.floor((Math.floor(availableCSS * dpr) - dataPixels) / 2)) / dpr;
}

/** Requested CSS scroll offset retaining a logical focal coordinate. Native bounds apply afterward. */
export function focalScroll(origin: number, focalCSS: number, oldScale: number, newScale: number, dpr: number) {
  return (origin + focalCSS * dpr / oldScale) * newScale / dpr - focalCSS;
}

/** Match GLSL highp float raster boundaries. Work relative to the first logical
 * cell so large scroll origins never lose their integer identity in float32. */
export function rasterOffset(origin: number, scale: number) {
  return Math.fround((origin - Math.floor(origin)) * scale);
}
export function rasterEdge(index: number, origin: number, scale: number) {
  return Math.ceil(Math.fround(Math.fround((index - Math.floor(origin)) * Math.fround(scale)) - rasterOffset(origin, scale)));
}
function rasterCell(origin: number, scale: number, pixel: number) {
  const first = Math.floor(origin);
  let cell = first + Math.floor((pixel + rasterOffset(origin, scale)) / Math.fround(scale));
  // Division may round either side of an exact edge. The rasterized boundary
  // is authoritative on both CPU and GPU, including at texture-band seams.
  if (pixel >= rasterEdge(cell + 1, origin, scale)) cell++;
  if (pixel < rasterEdge(cell, origin, scale)) cell--;
  return cell;
}
