export function frame(type: number, payload: Uint8Array = new Uint8Array()): number[] {
  const bytes = new Uint8Array(12 + payload.length);
  bytes.set([76, 77, 69, 88, type]);
  new DataView(bytes.buffer).setUint32(8, payload.length, true);
  bytes.set(payload, 12); return [...bytes];
}
export function meta(ids: number[], width = 7) {
  return frame(1, new TextEncoder().encode(JSON.stringify({ kind: 'input_embeddings', token_ids: ids, shape: [ids.length, width], dtype: 'float32', byte_order: 'little', layout: 'c', byte_length: ids.length * width * 4 })));
}
export const values = (ids: number[], width = 7) => ids.flatMap(id => Array.from({ length: width }, (_, column) => (id * 7 - column * 3) / 32));
export function data(ids: number[], width = 7) { return frame(2, new Uint8Array(new Float32Array(values(ids, width)).buffer)); }

/** Independent tiny-fixture oracle. Never used by the production viewer. */
export function analysis(ids: number[], width = 7) {
  const samples = values(ids, width), sorted = [...samples].sort((a, b) => a - b);
  const minimum = sorted[0]!, maximum = sorted.at(-1)!;
  const percentile = (q: number) => {
    const index = (samples.length - 1) * q, lower = Math.floor(index), fraction = index - lower;
    return sorted[lower]! * (1 - fraction) + sorted[Math.ceil(index)]! * fraction;
  };
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const statistics = { kind: 'input_embeddings_statistics', token_ids: ids, shape: [ids.length, width],
    count: samples.length, finite_count: samples.length, non_finite_count: 0, minimum, maximum, mean,
    stddev: Math.sqrt(samples.reduce((sum, value) => sum + (value - mean) ** 2, 0) / samples.length),
    percentiles: { p01: percentile(.01), p05: percentile(.05), p50: percentile(.5), p95: percentile(.95), p99: percentile(.99) }, byte_length: 0 };
  const counts = new Uint32Array((ids.length + width) * 100);
  samples.forEach((value, index) => {
    const bin = minimum === maximum ? 50 : Math.min(99, Math.floor((value - minimum) / (maximum - minimum) * 100));
    counts[Math.floor(index / width) * 100 + bin]!++;
    counts[ids.length * 100 + bin * width + index % width]!++;
  });
  const distribution = { kind: 'input_embeddings_distributions', token_ids: ids, rows: ids.length, columns: width,
    bin_count: 100, binning: 'linear-full-range', domain_minimum: minimum, domain_maximum: maximum,
    dtype: 'uint32', byte_order: 'little', byte_length: counts.byteLength,
    sections: [{ name: 'row_counts', shape: [ids.length, 100], offset: 0, byte_length: ids.length * 400 },
      { name: 'column_counts', shape: [100, width], offset: ids.length * 400, byte_length: width * 400 }] };
  return { statistics, distribution, counts,
    statisticsStream: [...frame(1, new TextEncoder().encode(JSON.stringify(statistics))), ...frame(4)],
    distributionStream: [...frame(1, new TextEncoder().encode(JSON.stringify(distribution))), ...frame(2, new Uint8Array(counts.buffer)), ...frame(4)],
  };
}
