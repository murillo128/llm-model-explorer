import { expect, it } from 'vitest';
import { selectedRowIntervals } from './matrix-row-selection';

it('coalesces only adjacent valid native rows and preserves disjoint ranges', () => {
  expect(selectedRowIntervals([9, 1, 3, 2, 2, 0, 6, -1, NaN, Infinity, 1.5, 10], 10))
    .toEqual([[0, 4], [6, 7], [9, 10]]);
  expect(selectedRowIntervals([], 10)).toEqual([]);
  expect(selectedRowIntervals([0, 1], 0)).toEqual([]);
});
