import { expect, it } from 'vitest';
import { compactValue, DistributionScale, distributionZero } from './distribution-scale';

it.each([
  [-2, 2, 0.5], [-2, 6, 0.25], [-6, 2, 0.75], [0, 8, 0], [-8, 0, 1],
  [-3.4028234663852886e38, 3.4028234663852886e38, 0.5],
  [-(2 ** -149), 3 * 2 ** -149, 0.25],
  [1, 8, null], [-8, -1, null], [0, 0, null], [3, 3, null], [null, null, null],
])('places zero in domain [%s, %s] without inventing a constant span', (minimum, maximum, zero) => {
  expect(distributionZero({ minimum, maximum })).toBe(zero);
});

it('labels authoritative endpoints and clears finite scale marks for unavailable data', () => {
  const panel = document.createElement('div');
  const scale = new DistributionScale('rows', panel);
  expect(scale.ruler.getAttribute('aria-label')).toContain('unavailable');
  scale.setDomain({ minimum: -2, maximum: 6 });
  expect(scale.ruler.getAttribute('aria-label')).toBe('Row bin domain: -2 to 6; linear, full finite range');
  expect(scale.ruler.style.getPropertyValue('--distribution-zero')).toBe('25%');
  expect(scale.guide.hidden).toBe(false);
  scale.setDomain({ minimum: 0, maximum: 0 });
  expect(scale.ruler.getAttribute('aria-label')).toContain('constant, samples in bin 50');
  expect(scale.guide.hidden).toBe(true);
  scale.setDomain({ minimum: null, maximum: null });
  expect(scale.ruler.textContent).toBe('');
  expect(scale.ruler.hasAttribute('data-minimum')).toBe(false);
  expect(scale.ruler.getAttribute('aria-label')).toContain('no finite values');
});

it('keeps small nonzero and extreme endpoints legible without rounding them to zero', () => {
  expect(compactValue(2 ** -149)).toBe('1.4e-45');
  expect(compactValue(-3.4028234663852886e38)).toBe('-3.4e38');
  expect(compactValue(-0)).toBe('0');
  expect(compactValue(0.125)).toBe('0.125');
});
