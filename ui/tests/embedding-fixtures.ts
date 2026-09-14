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
