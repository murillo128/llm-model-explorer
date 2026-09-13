// @vitest-environment node
import { describe, expect, it } from 'vitest';
import fixtures from '../../../api/fixtures/conformance.json';
import { LmexDecoder } from './lmex-decoder';
import { ApiFailure } from './errors';
import { validateSchema } from './validation';
import type { components } from './generated/types';

export function frame(type: number, payload: Uint8Array = new Uint8Array(0)): Uint8Array {
  const bytes = new Uint8Array(12 + payload.length);
  bytes.set([0x4c, 0x4d, 0x45, 0x58, type]);
  new DataView(bytes.buffer).setUint32(8, payload.length, true);
  bytes.set(payload, 12);
  return bytes;
}
function jsonFrame(type: number, value: unknown): Uint8Array {
  return frame(type, new TextEncoder().encode(JSON.stringify(value)));
}
function unaligned(hex: string): Uint8Array {
  const data = Uint8Array.from(Buffer.from(hex, 'hex'));
  const storage = new Uint8Array(data.length + 1);
  storage.set(data, 1);
  return storage.subarray(1);
}

function check(fixture: typeof fixtures.wire_cases[number], chunks: Iterable<Uint8Array>): void {
  const expected = fixture.expected;
  let offset = 0;
  const data = Buffer.from(expected.data_hex ?? '', 'hex');
  const decoder = new LmexDecoder({ onData: (bytes, position) => {
    expect(position).toBe(offset);
    if (expected.outcome === 'complete') expect(Buffer.from(bytes)).toEqual(data.subarray(offset, offset + bytes.length));
    offset += bytes.length;
  } });
  let result;
  try {
    for (const chunk of chunks) decoder.push(chunk);
    result = decoder.finish();
  } catch (error) {
    if (expected.outcome !== 'reject') throw error;
    expect(error).toBeInstanceOf(ApiFailure);
    return;
  }
  expect(expected.outcome).not.toBe('reject');
  expect(result.kind).toBe(expected.outcome === 'error' ? 'backend' : expected.outcome);
  if (result.kind === 'complete') {
    expect(result.metadata).toEqual(expected.metadata);
    expect(offset).toBe(data.length);
  }
}
function* lengths(bytes: Uint8Array, sizes: number[]) {
  let offset = 0;
  for (let i = 0; offset < bytes.length; i++) {
    const n = sizes[i % sizes.length]!;
    yield bytes.subarray(offset, offset + n);
    offset += n;
  }
}

describe('shared wire conformance', () => {
  for (const fixture of fixtures.wire_cases) {
    it(`${fixture.name}: coalesced, every split, single bytes, Fibonacci and seeded random chunks`, () => {
      const bytes = unaligned(fixture.wire_hex);
      check(fixture, [bytes]);
      for (let split = 0; split <= bytes.length; split++) check(fixture, [bytes.subarray(0, split), bytes.subarray(split)]);
      check(fixture, lengths(bytes, [1]));
      check(fixture, lengths(bytes, [1, 2, 3, 5, 8, 13]));
      let seed = 11;
      for (let run = 0; run < 5; run++) {
        const sizes = Array.from({ length: 23 }, () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % 97 + 1; });
        check(fixture, lengths(bytes, sizes));
      }
    }, 30_000);
  }
});

describe('shared schema conformance', () => {
  for (const fixture of fixtures.schema_cases) it(fixture.name, () => {
    const validate = () => validateSchema(fixture.schema as keyof components['schemas'], fixture.value);
    if (fixture.valid) expect(validate).not.toThrow();
    else expect(validate).toThrow(ApiFailure);
  });
});

it('forwards large DATA incrementally without copying its frame or network buffer', () => {
  const byteLength = 2 * 1024 * 1024;
  let received = 0;
  const payload = new Uint8Array(65537).subarray(1);
  const decoder = new LmexDecoder({ onData: (bytes, offset) => {
    expect(bytes.buffer === payload.buffer).toBe(true);
    expect(offset).toBe(received);
    received += bytes.length;
  } });
  decoder.push(jsonFrame(1, { kind: 'tensor', tensor_id: 't', name: 'é😀', shape: [byteLength / 4], dtype: 'float32', byte_order: 'little', layout: 'c', byte_length: byteLength }));
  const header = frame(2);
  new DataView(header.buffer).setUint32(8, byteLength, true);
  decoder.push(header);
  for (let i = 0; i < byteLength; i += payload.length) {
    decoder.push(payload);
    expect(received).toBe(i + payload.length);
  }
  decoder.push(frame(4));
  expect(decoder.finish().kind).toBe('complete');
});

it('handles split UTF-8 and does not accept terminals until clean EOF', () => {
  const metadata = { ...fixtures.wire_cases[1]!.expected.metadata, name: 'é😀' };
  const decoder = new LmexDecoder();
  for (const chunk of lengths(jsonFrame(1, metadata), [1])) decoder.push(chunk);
  decoder.push(frame(2, new Uint8Array(4)));
  decoder.push(frame(4));
  expect(() => decoder.push(new Uint8Array([0]))).toThrow('Bytes after terminal');
  expect(() => decoder.finish()).toThrow('already terminated');
});

it('keeps asymmetric uint32 values exact even above float32 integer precision', () => {
  const bytes = unaligned(fixtures.numeric_cases.uint32.little_endian_hex);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  expect([0, 4, 8].map(offset => view.getUint32(offset, true))).toEqual(fixtures.numeric_cases.uint32.values);
  // Exercise the decoder with two equal count sections and a count > 2^24.
  const count = 16909060;
  const metadata = { kind: 'tensor_distributions', tensor_id: 't', rows: 1, columns: count, bin_count: 100, binning: 'linear-full-range', domain_minimum: 0, domain_maximum: 0, dtype: 'uint32', byte_order: 'little', sections: [{ name: 'row_counts', shape: [1, 100], offset: 0, byte_length: 400 }, { name: 'column_counts', shape: [100, count], offset: 400, byte_length: 400 * count }], byte_length: 400 * (count + 1) };
  const seen: number[] = [];
  const decoder = new LmexDecoder({ onData: bytes => seen.push(...bytes) });
  decoder.push(jsonFrame(1, metadata));
  decoder.push(frame(2, bytes.subarray(4, 8)));
  decoder.push(frame(6));
  expect(decoder.finish().kind).toBe('cancelled');
  expect(seen).toEqual([...bytes.subarray(4, 8)]);
});

it('rejects nonzero counts in a null domain and inconsistent section totals', () => {
  const metadata = fixtures.wire_cases.find(f => f.name === 'distributions-all-nonfinite')!.expected.metadata!;
  const decoder = new LmexDecoder();
  decoder.push(jsonFrame(1, metadata));
  expect(() => decoder.push(frame(2, new Uint8Array([1, 0, 0, 0])))).toThrow('null domain');
  const valid = fixtures.wire_cases.find(f => f.name === 'distributions-range')!;
  const data = unaligned(valid.expected.data_hex!);
  new DataView(data.buffer, data.byteOffset).setUint32(0, 100, true);
  const other = new LmexDecoder();
  other.push(jsonFrame(1, valid.expected.metadata));
  other.push(frame(2, data));
  expect(() => other.push(frame(4))).toThrow('totals disagree');
});
