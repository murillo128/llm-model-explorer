import { ApiFailure, requireProtocol } from './errors';
import type { ApiError } from './errors';
import { product, safeSize, validateSchema } from './validation';
import type { Metadata, Progress } from './validation';

export type StreamOutcome =
  | { kind: 'complete'; metadata: Metadata; byteLength: number }
  | { kind: 'backend'; error: ApiError }
  | { kind: 'cancelled' };

export interface DecoderCallbacks {
  onMetadata?: (metadata: Metadata) => void;
  /** Borrowed raw little-endian bytes; copy into destination before returning if needed.
   * Offset/length may split an element. No conversion through Float32Array is done. */
  onData?: (bytes: Uint8Array, byteOffset: number) => void;
  onProgress?: (progress: Progress) => void;
}

/** Independent from fetch and React. Storage is a 12-byte header, <=1 MiB JSON,
 * and a 4-byte distribution word. DATA is never staged in a frame-sized buffer. */
export class LmexDecoder {
  private readonly header = new Uint8Array(12);
  private headerUsed = 0;
  private type = 0;
  private remaining = 0;
  private control = new Uint8Array(0);
  private controlUsed = 0;
  private metadata?: Metadata;
  private received = 0;
  private terminal?: StreamOutcome;
  private ended = false;
  private failed = false;
  private readonly word = new Uint8Array(4);
  private wordUsed = 0;
  private rowSum = 0;
  private columnSum = 0;

  constructor(private readonly callbacks: DecoderCallbacks = {}) {}

  push(chunk: Uint8Array): void {
    requireProtocol(!this.failed && !this.ended, 'Decoder has already terminated');
    try {
      let position = 0;
      while (position < chunk.length) {
        requireProtocol(!this.terminal, 'Bytes after terminal frame');
        if (this.headerUsed < 12) {
          const n = Math.min(12 - this.headerUsed, chunk.length - position);
          this.header.set(chunk.subarray(position, position + n), this.headerUsed);
          this.headerUsed += n;
          position += n;
          if (this.headerUsed < 12) continue;
          this.startFrame();
          if (this.remaining === 0) { this.endFrame(); continue; }
        }
        if (position === chunk.length) break;
        const n = Math.min(this.remaining, chunk.length - position);
        const bytes = chunk.subarray(position, position + n);
        if (this.type === 2) {
          requireProtocol(this.received + n <= this.metadata!.byte_length, 'DATA overrun');
          this.checkCounts(bytes);
          this.callbacks.onData?.(bytes, this.received);
          this.received += n;
        } else {
          this.control.set(bytes, this.controlUsed);
          this.controlUsed += n;
        }
        position += n;
        this.remaining -= n;
        if (this.remaining === 0) this.endFrame();
      }
    } catch (error) {
      this.failed = true;
      this.control = new Uint8Array(0);
      throw error;
    }
  }

  /** Only EOF establishes a valid terminal outcome, including ERROR/CANCELLED. */
  finish(): StreamOutcome {
    requireProtocol(!this.failed && !this.ended, 'Decoder has already terminated');
    this.ended = true;
    this.control = new Uint8Array(0);
    if (!this.terminal || this.headerUsed !== 0) throw new ApiFailure('transport', 'Incomplete LMEX body');
    return this.terminal;
  }

  private startFrame(): void {
    const h = this.header;
    const view = new DataView(h.buffer);
    requireProtocol(h[0] === 0x4c && h[1] === 0x4d && h[2] === 0x45 && h[3] === 0x58, 'Invalid LMEX magic');
    requireProtocol(h[5] === 0 && view.getUint16(6, true) === 0, 'Nonzero flags or reserved bits');
    this.type = h[4]!;
    this.remaining = view.getUint32(8, true);
    requireProtocol(this.type >= 1 && this.type <= 6, 'Unknown frame type');
    requireProtocol(this.metadata || [1, 5, 6].includes(this.type), 'Frame before metadata');
    requireProtocol(this.type !== 1 || !this.metadata, 'Duplicate metadata');
    if (this.type === 2) {
      requireProtocol(this.metadata!.kind !== 'tensor_statistics' && this.metadata!.kind !== 'input_embeddings_statistics', 'Statistics cannot contain DATA');
      requireProtocol(this.remaining % 4 === 0, 'Unaligned DATA frame');
      requireProtocol(this.remaining <= this.metadata!.byte_length - this.received, 'Declared DATA overrun');
    } else if (this.type === 4 || this.type === 6) {
      requireProtocol(this.remaining === 0, 'Terminal payload must be empty');
    } else {
      requireProtocol(this.remaining <= 1024 * 1024, 'JSON control frame exceeds 1 MiB');
      this.control = new Uint8Array(this.remaining);
      this.controlUsed = 0;
    }
  }

  private json(): unknown {
    try {
      return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(this.control));
    } catch (cause) {
      throw new ApiFailure('protocol', 'Invalid UTF-8 or JSON control payload', undefined, undefined, { cause });
    }
  }

  private endFrame(): void {
    switch (this.type) {
      case 1:
        this.metadata = validateSchema('StreamMetadata', this.json());
        this.callbacks.onMetadata?.(this.metadata);
        break;
      case 3: {
        const progress = validateSchema('StreamProgress', this.json());
        this.callbacks.onProgress?.(progress);
        break;
      }
      case 4:
        requireProtocol(this.received === this.metadata!.byte_length, 'COMPLETE byte count mismatch');
        if (this.metadata!.kind === 'tensor_distributions' || this.metadata!.kind === 'input_embeddings_distributions') {
          const m = this.metadata!;
          requireProtocol(this.rowSum === this.columnSum && this.rowSum <= product([m.rows, m.columns]), 'Distribution count totals disagree');
          requireProtocol(m.domain_minimum === null || this.rowSum > 0, 'Finite domain requires finite counts');
        }
        this.terminal = { kind: 'complete', metadata: this.metadata!, byteLength: this.received };
        break;
      case 5:
        this.terminal = { kind: 'backend', error: validateSchema('Error', this.json()) };
        break;
      case 6:
        this.terminal = { kind: 'cancelled' };
        break;
    }
    this.headerUsed = 0;
    this.control = new Uint8Array(0);
  }

  private checkCounts(bytes: Uint8Array): void {
    if (this.metadata?.kind !== 'tensor_distributions' && this.metadata?.kind !== 'input_embeddings_distributions') return;
    const m = this.metadata;
    const accept = (count: number, endOffset: number) => {
      requireProtocol(m.domain_minimum !== null || count === 0, 'Nonzero count in null domain');
      if (endOffset <= m.sections[0].byte_length) this.rowSum = safeSize(this.rowSum + count);
      else this.columnSum = safeSize(this.columnSum + count);
    };
    let i = 0;
    while (this.wordUsed > 0 && i < bytes.length) {
      this.word[this.wordUsed++] = bytes[i++]!;
      if (this.wordUsed === 4) {
        accept(new DataView(this.word.buffer).getUint32(0, true), this.received + i);
        this.wordUsed = 0;
      }
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (; i + 4 <= bytes.length; i += 4) accept(view.getUint32(i, true), this.received + i + 4);
    while (i < bytes.length) this.word[this.wordUsed++] = bytes[i++]!;
  }
}
