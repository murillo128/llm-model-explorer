/** Linear-sRGB luminance basis and display-only interaction color. */
export const luminance = [0.2126, 0.7152, 0.0722] as const;
export const amber = [1, 0.32, 0.015] as const;
export interface Selection { readonly row: number; readonly column: number }

/** Monotonic near-black green → green → pale green, in linear sRGB. */
export function scalarGreen(intensity: number): number[] {
  const t = Math.max(0, Math.min(1, intensity));
  return [0.001 + 0.819 * t ** 3, 0.006 + 0.994 * t ** 1.5, 0.002 + 0.858 * t ** 3];
}

export function chromaRGB(intensity: number, opacity: number): number[] {
  return scalarGreen(intensity).map((component, i) => component * (1 - opacity) + amber[i]! * opacity);
}

export function encodeSRGB(linear: number) {
  return linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055;
}
