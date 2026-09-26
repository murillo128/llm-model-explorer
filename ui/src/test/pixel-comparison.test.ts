// @vitest-environment node
import { expect, it } from 'vitest';
import { comparePixels } from '../../tests/pixel-comparison';

function scenario() {
  return { scenario: 'scale 3.25 / texture seam', width: 9, height: 3,
    frame: Array.from({ length: 3 }, (_, y) => Array.from({ length: 9 }, (_, x) => [y, x, 76, 255])),
    cells: Array.from({ length: 3 }, (_, row) => Array.from({ length: 9 }, (_, column) => ({ row, column }))),
    expected: ({ row, column }: { row: number; column: number }) => [row, column, 76, 255] };
}
it('compares every RGBA channel and preserves input arrays', () => {
  const input = scenario(), before = structuredClone({ frame: input.frame, cells: input.cells });
  expect(comparePixels(input)).toBe(3 * 9 * 4);
  expect({ frame: input.frame, cells: input.cells }).toEqual(before);
});
for (const [x, y] of [[0, 0], [4, 1], [8, 2], [7, 0], [8, 0]]) {
  it(`rejects a wrong pixel at ${x}:${y}, including first/middle/last and band edges`, () => {
    const input = scenario(); input.frame[y!]![x!]![2] = input.frame[y!]![x!]![2]! + 1;
    expect(() => comparePixels(input)).toThrow(`device pixel ${x}:${y}, cell ${y}:${x}, channel 2: expected 76, actual 77`);
  });
}
it('enforces exactness and the explicitly selected tolerance', () => {
  const input = scenario(); input.frame[0]![0]![2] = input.frame[0]![0]![2]! + 1;
  expect(() => comparePixels(input)).toThrow('tolerance 0');
  expect(comparePixels({ ...input, tolerance: 1 })).toBe(108);
  input.frame[0]![0]![2] = input.frame[0]![0]![2]! + 1;
  expect(() => comparePixels({ ...input, tolerance: 1 })).toThrow('tolerance 1');
});
it('bounds failure details while still counting every mismatch', () => {
  const input = scenario(); input.expected = () => [50, 50, 50, 50];
  try { comparePixels(input); throw new Error('comparison should reject'); } catch (error) {
    expect(String(error)).toContain('108 mismatched channels in 108 comparisons');
    expect(String(error).split('\n')).toHaveLength(9);
    expect(String(error)).toContain(input.scenario);
  }
});
it('rejects empty, truncated, oversized and ragged evidence', () => {
  for (const edit of [
    (v: ReturnType<typeof scenario>) => { v.frame = []; },
    (v: ReturnType<typeof scenario>) => { v.width = 0; },
    (v: ReturnType<typeof scenario>) => { v.height++; },
    (v: ReturnType<typeof scenario>) => { v.frame[1]!.pop(); },
    (v: ReturnType<typeof scenario>) => { v.frame[1]!.push([0, 0, 0, 255]); },
    (v: ReturnType<typeof scenario>) => { v.frame[1]![2]!.pop(); },
    (v: ReturnType<typeof scenario>) => { v.cells[1]!.pop(); },
    (v: ReturnType<typeof scenario>) => { delete v.cells[1]![2]; },
    (v: ReturnType<typeof scenario>) => { v.cells[1]![2]!.row = -1; },
    (v: ReturnType<typeof scenario>) => { v.frame[1]![2]![0] = NaN; },
  ]) {
    const input = scenario(); edit(input); expect(() => comparePixels(input)).toThrow();
  }
  expect(() => comparePixels({ ...scenario(), expected: () => [] })).toThrow('four channels');
  expect(() => comparePixels({ ...scenario(), expected: () => [NaN, 0, 0, 255] })).toThrow('mismatched');
  expect(() => comparePixels({ ...scenario(), tolerance: NaN })).toThrow('invalid tolerance');
});
