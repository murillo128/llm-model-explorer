import { expect, it } from 'vitest';
import { INSPECTION, inspectionPosition, magnifierVisible } from './inspection-layout';

const bounds = { left: 200, top: 100, right: 900, bottom: 800 };
const matrix = { left: 200, top: 100, right: 780, bottom: 680 };
const panels = [
  { left: 790, top: 100, right: 890, bottom: 680 },
  { left: 200, top: 690, right: 780, bottom: 790 },
];
const intersects = (a: typeof bounds, b: typeof bounds) =>
  a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;

it('keeps all four corners and panel-adjacent inspections inside the pane, clear of panels and neighborhood', () => {
  for (const cellSize of [0.5, 1, 7.9]) {
    for (const [x, y] of [[201, 101], [779, 101], [201, 679], [779, 679], [779, 390], [490, 679]]) {
      const rect = inspectionPosition(x!, y!, bounds, matrix, panels, cellSize)!;
      expect(rect).not.toBeNull();
      expect(rect.left).toBeGreaterThanOrEqual(bounds.left);
      expect(rect.top).toBeGreaterThanOrEqual(bounds.top);
      expect(rect.right).toBeLessThanOrEqual(bounds.right);
      expect(rect.bottom).toBeLessThanOrEqual(bounds.bottom);
      for (const panel of panels) expect(intersects(rect, panel)).toBe(false);
      const radius = 4.5 * cellSize;
      expect(intersects(rect, { left: x! - radius, right: x! + radius, top: y! - radius, bottom: y! + radius })).toBe(false);
      if (x === 779) expect(rect.right).toBeLessThan(x);
      if (y === 679) expect(rect.bottom).toBeLessThan(y);
    }
  }
});

it('uses empty scientific space before covering a short matrix', () => {
  const short = { ...matrix, right: 219, bottom: 117 };
  const rect = inspectionPosition(209, 109, bounds, short, [], 1)!;
  expect(intersects(rect, short)).toBe(false);
});

it('reports insufficient magnifier space so a compact readout can be placed without panel collisions', () => {
  const small = { left: 0, top: 0, right: 360, bottom: 180 };
  const panel = { left: 260, top: 0, right: 360, bottom: 180 };
  expect(inspectionPosition(240, 80, small, small, [panel], 1)).toBeNull();
  const rect = inspectionPosition(240, 80, small, small, [panel], 1, INSPECTION.readoutHeight)!;
  expect(rect.bottom).toBeLessThanOrEqual(small.bottom);
  expect(intersects(rect, panel)).toBe(false);
});

it('uses exact 10/8 CSS-pixel thresholds and holds state throughout the hysteresis band', () => {
  let visible = true;
  for (const [size, expected] of [[7.9, true], [8, true], [9.99, true], [10, false],
    [9.99, false], [10.01, false], [9, false], [8, false], [7.99, true], [8.01, true]]) {
    visible = magnifierVisible(visible, size as number);
    expect(visible).toBe(expected);
  }
});
