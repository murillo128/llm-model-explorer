// @vitest-environment node
import { expect, it, vi } from 'vitest';
import fixtures from '../../../../api/fixtures/conformance.json';
import { compareByteRange, prepareWireCheck } from './wire-conformance';
import { LmexDecoder } from '../lmex-decoder';
import type { DecoderCallbacks } from '../lmex-decoder';

it('preserves float32 special-value and exact uint32 bytes in unaligned chunks', () => {
  const expected = Buffer.from('000000800000807f000080ff0100c07f0403020101000001', 'hex');
  const source = new Uint8Array(expected.length + 1); source.set(expected, 1);
  const before = Buffer.from(expected);
  for (let split = 0; split <= expected.length; split++) {
    compareByteRange(source.subarray(1, split + 1), expected, 0);
    compareByteRange(source.subarray(split + 1), expected, split);
  }
  expect(expected).toEqual(before);
  for (const offset of [0, 3, 4, 11, 12, expected.length - 1]) {
    const corrupt = source.slice(1); corrupt[offset] = corrupt[offset]! ^ 1;
    expect(() => compareByteRange(corrupt, expected, 0)).toThrow(`DATA byte ${offset}`);
  }
});
it('rejects invalid ranges, offsets and truncated expected data', () => {
  for (const offset of [-1, .5, NaN, 2]) expect(() => compareByteRange(new Uint8Array(2), Buffer.alloc(3), offset)).toThrow('range');
  expect(() => compareByteRange(new Uint8Array(4), Buffer.alloc(3), 0)).toThrow('length');
  expect(() => compareByteRange(new Uint8Array([2]), Buffer.from([1, 2]), 0)).toThrow('byte 0');
});
const complete = fixtures.wire_cases.find(f => f.name === 'tensor-scalar') ?? fixtures.wire_cases.find(f => f.expected.outcome === 'complete' && f.expected.data_hex)!;
it('keeps preparation private and unchanged across fresh decoder variants', () => {
  const fixture = structuredClone(complete), wire = Buffer.from(fixture.wire_hex, 'hex');
  const check = prepareWireCheck(fixture);
  fixture.expected.data_hex = 'ff';
  Object.assign(fixture.expected.metadata!, { name: 'mutated caller metadata' });
  for (let split = 0; split <= wire.length; split++) check([wire.subarray(0, split), wire.subarray(split)]);
  expect(wire.toString('hex')).toBe(complete.wire_hex);
});
it('rejects wrong payload bytes, total length, metadata and terminal expectation', () => {
  const wire = Buffer.from(complete.wire_hex, 'hex');
  const expected = complete.expected;
  const data = Buffer.from(expected.data_hex!, 'hex'); data[0] = data[0]! ^ 1;
  expect(() => prepareWireCheck({ name: 'wrong-byte', expected: { ...expected, data_hex: data.toString('hex') } })([wire])).toThrow('byte 0');
  expect(() => prepareWireCheck({ name: 'wrong-length', expected: { ...expected, data_hex: expected.data_hex + '00' } })([wire])).toThrow('DATA length');
  expect(() => prepareWireCheck({ name: 'wrong-metadata', expected: { ...expected, metadata: {} } })([wire])).toThrow('Metadata');
  expect(() => prepareWireCheck({ name: 'wrong-terminal', expected: { ...expected, outcome: 'cancelled' } })([wire])).toThrow('Terminal');
  expect(() => prepareWireCheck({ name: 'missing-rejection', expected: { outcome: 'reject' } })([wire])).toThrow('Expected decoder rejection');
});
it('never swallows callback comparison failures as expected decoder rejections', () => {
  const push = vi.spyOn(LmexDecoder.prototype, 'push').mockImplementation(function (this: LmexDecoder) {
    (this as unknown as { callbacks: DecoderCallbacks }).callbacks.onData!(new Uint8Array([0]), 1);
  });
  try {
    expect(() => prepareWireCheck({ name: 'offset corruption before malformed frame', expected: { outcome: 'reject' } })([new Uint8Array(1)])).toThrow('DATA offset: expected 0, actual 1');
  } finally { push.mockRestore(); }
});
