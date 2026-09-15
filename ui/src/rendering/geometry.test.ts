import { describe, expect, it } from 'vitest';
import { centeredOffset, fitWidthScale, focalScroll, hitTest, viewGeometry } from './geometry';

const matrix = { rows: 900, columns: 100, count: 90000 };
describe('square-cell camera', () => {
  it.each([1, 1.25, 2, 3])('centers only underfilled axes on device pixels at DPR %s', (dpr) => {
    for (const available of [101, 300.5, 600]) {
      const offset = centeredOffset(available, 25, dpr);
      expect(Number.isInteger(offset * dpr)).toBe(true);
      expect(Math.abs(offset * 2 + 25 / dpr - available)).toBeLessThan(2 / dpr);
      expect(centeredOffset(available, Math.ceil(available * dpr), dpr)).toBe(0);
      expect(centeredOffset(available, 2000, dpr)).toBe(0);
    }
  });
  it.each([1, 1.25, 2, 3])('fits width without minification at DPR %s', (dpr) => {
    expect(fitWidthScale(100, 500, dpr)).toBe(5 * dpr);
    expect(fitWidthScale(2000, 500, dpr)).toBe(1);
    expect(fitWidthScale(0, 500, dpr)).toBe(1);
  });
  it.each([1, 1.25, 2])('retains exact focal coordinates and bounds at DPR %s', (dpr) => {
    const scale = 7.5;
    const view = viewGeometry(matrix, 60, 80, 40, 90, dpr, 4096, 4096, scale);
    const focal = 20;
    const scroll = focalScroll(view.y, focal, scale, 15, dpr);
    expect(scroll * dpr / 15 + focal * dpr / 15).toBeCloseTo(view.y + focal * dpr / scale);
    expect(view.scaleX).toBe(view.scaleY);
    expect(hitTest(view, 20.2, 30.7)).toEqual({
      row: Math.floor(view.y + Math.floor(30.7 * dpr) / scale),
      column: Math.floor(view.x + Math.floor(20.2 * dpr) / scale),
    });
    expect(hitTest(view, view.cssWidth, 0)).toBeNull();
    const end = viewGeometry(matrix, 60, 80, 1e8, 1e8, dpr, 4096, 4096, scale);
    expect(end.x + end.width / scale).toBeCloseTo(matrix.columns);
    expect(end.y + end.height / scale).toBeCloseTo(matrix.rows);
  });
  it('rejects subpixel scale and shares only the data axis with profiles', () => {
    expect(() => viewGeometry(matrix, 50, 50, 0, 0, 1, 4096, 4096, .5)).toThrow();
    const main = viewGeometry(matrix, 50, 50, 12, 33, 2, 4096, 4096, 8);
    const row = viewGeometry({ rows: 900, columns: 100, count: 90000 }, 50, 50, 0, 33, 2, 4096, 4096, 1, 8);
    expect(row.y).toBe(main.y);
    expect(row.height).toBe(main.height);
    expect(row.width).toBe(100);
    expect(row.scaleX).toBe(1);
  });
});
