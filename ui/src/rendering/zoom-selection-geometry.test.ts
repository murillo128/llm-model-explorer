import { describe, expect, it } from 'vitest';
import { selectionBoundary, selectionCamera } from './zoom-selection-geometry';
import { rasterEdge, viewGeometry } from './geometry';

const tensor = { rows: 1000, columns: 1000, count: 1000000 };
const view = viewGeometry(tensor, 400, 200, 200, 300, 1, 4096, 4096, 2);
describe('selection camera', () => {
  it('fits a mismatched rectangle uniformly and centers the spare dimension', () => {
    expect(selectionCamera({ columns: [100, 120], rows: [150, 170] }, view, tensor, 400, 200))
      .toEqual({ scale: 10, x: 90, y: 150 });
  });
  it('fits either axis while retaining the orthogonal logical center', () => {
    expect(selectionCamera({ columns: [100, 120] }, view, tensor, 400, 200)).toEqual({ scale: 20, x: 100, y: 195 });
    expect(selectionCamera({ rows: [150, 170] }, view, tensor, 400, 200)).toEqual({ scale: 10, x: 180, y: 150 });
  });
  it('clamps edges and retains native minimum', () => {
    expect(selectionCamera({ columns: [0, 10], rows: [0, 20] }, view, tensor, 400, 200))
      .toEqual({ scale: 10, x: 0, y: 0 });
    expect(selectionCamera({ columns: [990, 1000], rows: [980, 1000] }, view, tensor, 400, 200))
      .toEqual({ scale: 10, x: 960, y: 980 });
    expect(selectionCamera({ columns: [0, 1000] }, view, tensor, 400, 200)?.scale).toBe(1);
  });
  it('rejects empty, degenerate, fractional and out-of-bounds regions', () => {
    for (const bounds of [{}, { rows: [2, 2] as const }, { columns: [2, 1] as const },
      { rows: [-1, 2] as const }, { rows: [0, 1001] as const }, { columns: [0.1, 5] as const }]) {
      expect(selectionCamera(bounds, view, tensor, 400, 200)).toBeNull();
    }
  });
  it('fits a one-cell range even above the wheel camera historical 64x-fit ceiling', () => {
    expect(selectionCamera({ rows: [2, 3] }, view, tensor, 400, 200)?.scale).toBe(200);
  });
});
for (const dpr of [1, 1.25, 2]) for (const scale of [1, 3.7, 12]) {
  it(`snaps exact raster boundaries at DPR ${dpr}, scale ${scale}`, () => {
    const v = viewGeometry(tensor, 400, 200, 20, 30, dpr, 4096, 4096, scale);
    for (const axis of ['columns', 'rows'] as const) {
      const origin = axis === 'columns' ? v.x : v.y;
      const index = Math.ceil(origin) + 3;
      const edge = rasterEdge(index, origin, scale) / dpr;
      expect(selectionBoundary(v, edge, axis, 1000)).toBe(index);
      expect(selectionBoundary(v, edge + .1 / dpr, axis, 1000)).toBe(index);
    }
  });
}
