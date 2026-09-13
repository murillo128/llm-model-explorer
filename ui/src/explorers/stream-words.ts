/** Convert borrowed, arbitrarily split little-endian bytes in bounded batches.
 * Only an unfinished word survives a callback; no full-result staging buffer. */
export class StreamWords<T extends Float32Array | Uint32Array> {
  private readonly tail = new Uint8Array(4);
  private tailUsed = 0;
  private received = 0;
  constructor(private readonly kind: 'float32' | 'uint32', private readonly accept: (values: T, elementOffset: number) => void) {}

  push(bytes: Uint8Array, byteOffset: number) {
    if (byteOffset !== this.received) throw new Error('Discontinuous stream bytes.');
    let position = 0;
    while (position < bytes.length) {
      const size = Math.min(16384, Math.floor((bytes.length - position + this.tailUsed) / 4));
      if (!size) {
        this.tail.set(bytes.subarray(position), this.tailUsed);
        this.tailUsed += bytes.length - position;
        this.received += bytes.length - position;
        break;
      }
      const values = (this.kind === 'float32' ? new Float32Array(size) : new Uint32Array(size)) as T;
      const start = (this.received - this.tailUsed) / 4;
      let index = 0;
      const read = (view: DataView, offset: number) => this.kind === 'float32' ? view.getFloat32(offset, true) : view.getUint32(offset, true);
      if (this.tailUsed) {
        const needed = 4 - this.tailUsed;
        this.tail.set(bytes.subarray(position, position + needed), this.tailUsed);
        position += needed;
        this.received += needed;
        values[index++] = read(new DataView(this.tail.buffer), 0);
        this.tailUsed = 0;
      }
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (; index < size; index++) {
        values[index] = read(view, position);
        position += 4;
        this.received += 4;
      }
      this.accept(values, start);
    }
  }
}
