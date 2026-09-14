// Double-precision oracle for the normative display curve (no renderer imports).
export const decode = (byte: number) => byte / 255 <= 0.04045 ? byte / 255 / 12.92 : ((byte / 255 + 0.055) / 1.055) ** 2.4;
export const luminance = (rgb: number[]) => rgb.slice(0, 3).reduce((sum, v, i) => sum + decode(v) * [0.2126, 0.7152, 0.0722][i]!, 0);
export function intensity(value: number, low = -1, high = 1, slope = 12) {
  const u = Math.max(0, Math.min(1, (value - low) / (high - low)));
  const logistic = (x: number) => 1 / (1 + Math.exp(-x));
  const endpoint = logistic(-slope / 2);
  return (logistic(slope * (u - 0.5)) - endpoint) / (1 - 2 * endpoint);
}
export function color(t: number, opacity = 0) {
  const base = [0.001 + 0.819 * t ** 3, 0.006 + 0.994 * t ** 1.5, 0.002 + 0.858 * t ** 3];
  return base.map((c, i) => {
    const v = c * (1 - opacity) + [1, 0.32, 0.015][i]! * opacity;
    return Math.round(255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055));
  });
}
export function green(value: number, low = -1, high = 1, slope = 12) {
  return [...color(intensity(value, low, high, slope)), 255];
}
