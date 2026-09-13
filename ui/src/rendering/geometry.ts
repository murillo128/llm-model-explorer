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
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  readonly cssWidth: number;
  readonly cssHeight: number;
  readonly scrollWidth: number;
  readonly scrollHeight: number;
}

/** All origins and sizes are integer data/device pixels; CSS is only layout. */
export function viewGeometry(tensor: TensorGeometry, cssWidth: number, cssHeight: number,
  scrollLeft: number, scrollTop: number, dpr: number, maxWidth: number, maxHeight: number): ViewGeometry {
  if (![cssWidth, cssHeight, scrollLeft, scrollTop, dpr].every(Number.isFinite) ||
      dpr <= 0 || cssWidth < 0 || cssHeight < 0) throw new Error('Invalid viewport geometry.');
  const width = Math.min(tensor.columns, maxWidth, Math.floor(cssWidth * dpr));
  const height = Math.min(tensor.rows, maxHeight, Math.floor(cssHeight * dpr));
  const x = Math.max(0, Math.min(tensor.columns - width, Math.round(scrollLeft * dpr)));
  const y = Math.max(0, Math.min(tensor.rows - height, Math.round(scrollTop * dpr)));
  return Object.freeze({ dpr, x, y, width, height, cssWidth: width / dpr, cssHeight: height / dpr,
    scrollWidth: tensor.columns / dpr, scrollHeight: tensor.rows / dpr });
}

export function hitTest(view: ViewGeometry, cssX: number, cssY: number) {
  if (!Number.isFinite(cssX) || !Number.isFinite(cssY) || cssX < 0 || cssY < 0 ||
      cssX >= view.cssWidth || cssY >= view.cssHeight) return null;
  return { row: view.y + Math.floor(cssY * view.dpr), column: view.x + Math.floor(cssX * view.dpr) };
}
