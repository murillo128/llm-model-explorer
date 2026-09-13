/** Structurally compatible with the finite statistics in the API metadata. */
export interface TensorStatistics {
  readonly minimum: number | null;
  readonly maximum: number | null;
  readonly percentiles: { readonly p01: number | null; readonly p99: number | null };
}

export interface TransferParameters {
  readonly slope?: number;
  readonly statistics?: TensorStatistics | null;
  readonly anchors?: readonly [number, number] | null;
}

export interface TransferUniforms {
  readonly mode: 0 | 1 | 2; // provisional, anchored, constant
  readonly slope: number;
  readonly low: number;
  readonly high: number;
  readonly scale: number;
}

export function transferUniforms(parameters: TransferParameters = {}): TransferUniforms {
  const slope = parameters.slope ?? 8;
  // Keep the endpoint normalization well-conditioned even on float32 shaders.
  if (!Number.isFinite(slope) || slope < 0.01 || slope > 80) {
    throw new Error('Sigmoid slope must be between 0.01 and 80.');
  }
  const stats = parameters.statistics;
  let anchors = parameters.anchors;
  if (!anchors && stats) {
    const { p01, p99 } = stats.percentiles;
    if (p01 !== null && p99 !== null) anchors = [p01, p99];
    if ((!anchors || anchors[0] === anchors[1]) && stats.minimum !== null && stats.maximum !== null) {
      anchors = [stats.minimum, stats.maximum];
    }
  }
  if (!anchors) return { mode: 0, slope, low: 0, high: 1, scale: 1 };
  const [low, high] = anchors;
  if (![low, high].every((value) => Number.isFinite(value) && Number.isFinite(Math.fround(value))) || low > high) {
    throw new Error('Luminosity anchors must be ordered finite float32-range numbers.');
  }
  return { mode: low === high ? 2 : 1, slope, low, high, scale: Math.max(Math.abs(low), Math.abs(high), 1e-30) };
}
