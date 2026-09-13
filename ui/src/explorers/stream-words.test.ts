import { expect, it } from 'vitest';
import { StreamWords } from './stream-words';

it('decodes split, unaligned little-endian float32 words without losing the prefix', () => {
  const source = new Uint8Array(16);
  const view = new DataView(source.buffer);
  const expected = [-3.25, 0, 100.5, Infinity];
  expected.forEach((value, index) => view.setFloat32(index * 4, value, true));
  for (let split = 1; split < 16; split++) {
    const actual: number[] = [];
    const reader = new StreamWords<Float32Array>('float32', (values, offset) => {
      expect(offset).toBe(actual.length);
      actual.push(...values);
    });
    reader.push(source.subarray(0, split), 0);
    reader.push(source.subarray(split), split);
    expect(actual).toEqual(expected);
  }
});

it('retains uint32 precision and bounds transient decoding batches', () => {
  const source = new Uint8Array(4 * 40000);
  const view = new DataView(source.buffer);
  for (let i = 0; i < 40000; i++) view.setUint32(i * 4, 0xffffffff - i, true);
  let received = 0;
  const reader = new StreamWords<Uint32Array>('uint32', (values, offset) => {
    expect(values).toBeInstanceOf(Uint32Array);
    expect(values.length).toBeLessThanOrEqual(16384);
    expect(offset).toBe(received);
    for (const value of values) expect(value).toBe(0xffffffff - received++);
  });
  reader.push(source, 0);
  expect(received).toBe(40000);
  expect(() => reader.push(source, 1)).toThrow('Discontinuous');
});
