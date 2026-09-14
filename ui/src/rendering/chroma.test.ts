import { describe, expect, it } from 'vitest';
import { chromaRGB, scalarGreen, encodeSRGB, luminance } from './chroma';
import { formatFloat32 } from './matrix-inspection';

describe('green scalar and amber overlay', () => {
  it('orders luminance throughout the green scale and keeps compositing in gamut', () => {
    let previous = -1;
    for (let i = 0; i <= 1000; i++) {
      const t = i / 1000;
      const green = scalarGreen(t);
      expect(green[1]).toBeGreaterThan(green[0]!);
      expect(green[1]).toBeGreaterThan(green[2]!);
      const y = green.reduce((sum, v, j) => sum + v * luminance[j]!, 0);
      expect(y).toBeGreaterThan(previous); previous = y;
      for (const opacity of [0, 0.65, 0.9, 1]) {
        const rgb = chromaRGB(t, opacity);
        expect(rgb.every((c) => c >= 0 && c <= 1)).toBe(true);
        if (opacity >= 0.65) expect(rgb[0]).toBeGreaterThan(rgb[1]!);
      }
    }
    expect(encodeSRGB(0.5)).toBeCloseTo(0.735356983, 8);
  });
});

it('readout round-trips exact float32s including signed zero and nonfinite values', () => {
  const words = new Uint32Array([0, 0x80000000, 1, 0x7f7fffff, 0xff7fffff, 0x3f800001, 0x7f800000, 0xff800000, 0x7fc00000]);
  for (const value of new Float32Array(words.buffer)) expect(Object.is(Math.fround(Number(formatFloat32(value))), value)).toBe(true);
  expect(formatFloat32(-0)).toBe('-0');
});
