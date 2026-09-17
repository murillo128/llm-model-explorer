/** Test-only LMEX oracle; never imported by the runtime decoder. */
import { isDeepStrictEqual } from 'node:util';
import { LmexDecoder } from '../lmex-decoder';
import { ApiFailure } from '../errors';

interface WireFixture {
  readonly name: string;
  readonly expected: { readonly outcome: string; readonly data_hex?: string; readonly metadata?: unknown };
}

/** Buffer views preserve unaligned offsets, NaN payloads, signed zero and uint32 bits. */
export function compareByteRange(actual: Uint8Array, expected: Buffer, offset: number): void {
  if (!Number.isSafeInteger(offset) || offset < 0 || offset + actual.byteLength > expected.byteLength) {
    throw new Error(`DATA range offset ${offset}, length ${actual.byteLength}, expected length ${expected.byteLength}`);
  }
  const view = Buffer.from(actual.buffer, actual.byteOffset, actual.byteLength);
  if (view.compare(expected, offset, offset + view.length) === 0) return;
  // Only the failure path constructs detailed messages; success compares every byte natively.
  const details: string[] = [];
  for (let i = 0; i < view.length && details.length < 8; i++) {
    if (view[i] !== expected[offset + i]) details.push(`DATA byte ${offset + i} (chunk byte ${i}): expected ${expected[offset + i]}, actual ${view[i]}`);
  }
  throw new Error(details.join('\n'));
}

function freeze(value: unknown): unknown {
  if (value && typeof value === 'object') {
    for (const item of Object.values(value)) freeze(item);
    Object.freeze(value);
  }
  return value;
}

export function prepareWireCheck(fixture: WireFixture): (chunks: Iterable<Uint8Array>) => void {
  const { outcome } = fixture.expected;
  // Private expected bytes are never passed to decoder callbacks or exposed to callers.
  const data = Buffer.from(fixture.expected.data_hex ?? '', 'hex');
  const metadata = freeze(structuredClone(fixture.expected.metadata));
  return (chunks) => {
    let offset = 0;
    const decoder = new LmexDecoder({ onData: (bytes, position) => {
      if (position !== offset) throw new Error(`DATA offset: expected ${offset}, actual ${position}`);
      if (outcome === 'complete') compareByteRange(bytes, data, offset);
      offset += bytes.length;
    } });
    let result;
    try {
      for (const chunk of chunks) decoder.push(chunk);
      result = decoder.finish();
    } catch (error) {
      // Oracle errors are plain Errors and must never become expected protocol rejections.
      if (outcome !== 'reject' || !(error instanceof ApiFailure)) throw error;
      return;
    }
    if (outcome === 'reject') throw new Error('Expected decoder rejection, received a terminal outcome');
    const kind = outcome === 'error' ? 'backend' : outcome;
    if (result.kind !== kind) throw new Error(`Terminal: expected ${kind}, actual ${result.kind}`);
    if (result.kind === 'complete') {
      if (!isDeepStrictEqual(result.metadata, metadata)) throw new Error(`Metadata: expected ${JSON.stringify(metadata)}, actual ${JSON.stringify(result.metadata)}`);
      if (offset !== data.length || result.byteLength !== data.length) throw new Error(`DATA length: expected ${data.length}, received ${offset}, terminal ${result.byteLength}`);
    }
  };
}
