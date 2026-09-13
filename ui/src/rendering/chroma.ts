/** Linear-sRGB luminance basis; the amber displacement is orthogonal to it. */
export const luminance = [0.2126, 0.7152, 0.0722] as const;
export const amber = [1, (0.0722 * 0.6 - 0.2126) / 0.7152, -0.6] as const;
export interface Selection { readonly row: number; readonly column: number }

export function chromaRGB(y: number, strength: number): number[] {
  let t = strength;
  for (const component of amber) {
    if (component > 0) t = Math.min(t, (1 - y) / component);
    if (component < 0) t = Math.min(t, -y / component);
  }
  return amber.map((component) => y + t * component);
}

export function encodeSRGB(linear: number) {
  return linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055;
}
