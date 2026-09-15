/** Authoritative value-bin endpoints, independent from the matrix luminosity transfer. */
export interface DistributionDomain {
  readonly minimum: number | null;
  readonly maximum: number | null;
}

export function distributionZero(domain: DistributionDomain): number | null {
  const { minimum: low, maximum: high } = domain;
  if (low === null || high === null || low > 0 || high < 0) return null;
  // A constant has no numeric span. The API places its samples in bin 50.
  if (low === high) return null;
  return low === 0 ? 0 : -low / (high - low); // JS float64 safely spans opposite float32 extremes.
}

export function compactValue(value: number): string {
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  return magnitude >= 1e4 || magnitude < 0.001
    ? value.toExponential(1).replace(/\.0e/, 'e').replace('e+', 'e')
    : String(Number(value.toPrecision(3)));
}

/** DOM chrome only: no count storage, density transfer, or camera state. */
export class DistributionScale {
  readonly ruler = document.createElement('div');
  readonly guide = document.createElement('div');
  private captionKey = '';
  constructor(private readonly orientation: 'rows' | 'columns', panel: HTMLElement) {
    this.ruler.className = `distribution-scale distribution-scale-${orientation}`;
    this.ruler.setAttribute('role', 'img');
    this.guide.className = `distribution-zero distribution-zero-${orientation}`;
    this.guide.setAttribute('aria-hidden', 'true');
    panel.append(this.guide);
    this.setDomain();
  }

  setDomain(domain?: DistributionDomain) {
    this.captionKey = '';
    this.ruler.replaceChildren();
    delete this.ruler.dataset.minimum;
    delete this.ruler.dataset.maximum;
    this.ruler.classList.remove('distribution-scale-constant');
    this.guide.hidden = true;
    const label = this.orientation === 'rows' ? 'Row' : 'Column';
    if (!domain || domain.minimum === null || domain.maximum === null) {
      this.ruler.setAttribute('aria-label', `${label} bin domain: ${domain ? 'no finite values' : 'unavailable'}`);
      return;
    }
    const { minimum: low, maximum: high } = domain;
    const constant = low === high;
    this.ruler.setAttribute('aria-label', `${label} bin domain: ${low} to ${high}${constant ? '; constant, samples in bin 50' : '; linear, full finite range'}`);
    this.ruler.dataset.minimum = String(low);
    this.ruler.dataset.maximum = String(high);
    this.ruler.classList.toggle('distribution-scale-constant', constant);
    for (const [name, value] of [['low', low], ['high', high]] as const) {
      const tick = document.createElement('span');
      tick.className = `distribution-endpoint distribution-${name}`;
      tick.textContent = compactValue(value);
      tick.title = `Bin domain ${name}: ${value}`;
      this.ruler.append(tick);
    }
    this.fitCaptions();
    const zero = distributionZero(domain);
    if (zero === null) return;
    const position = `${zero * 100}%`;
    this.ruler.style.setProperty('--distribution-zero', position);
    this.guide.style.setProperty('--distribution-zero', position);
    this.guide.hidden = false;
    const tick = document.createElement('span');
    tick.className = 'distribution-zero-tick';
    // Endpoint labels already identify boundary zero. Keep its exact anchor
    // without repeating that label or shifting it into the finite domain.
    tick.textContent = this.orientation === 'rows' && (zero === 0 || zero === 1) ? '' : '0';
    this.ruler.append(tick);
  }

  /** Follow the renderer's existing physical-pixel alignment without moving data. */
  alignBinAxis(offset: string) {
    this.ruler.style.setProperty('--distribution-offset', offset);
    this.guide.style.setProperty('--distribution-offset', offset);
    this.fitCaptions();
  }

  private fitCaptions() {
    if (this.orientation !== 'rows' || !this.ruler.hasAttribute('data-minimum')) return;
    const width = this.ruler.getBoundingClientRect().width;
    if (!width) return;
    const key = `${width},${this.ruler.dataset.minimum},${this.ruler.dataset.maximum}`;
    if (key === this.captionKey) return;
    this.captionKey = key;
    const endpoints = Array.from(this.ruler.querySelectorAll<HTMLElement>('.distribution-endpoint'));
    const values = [Number(this.ruler.dataset.minimum), Number(this.ruler.dataset.maximum)];
    this.ruler.querySelector('.distribution-exponent')?.remove();
    endpoints.forEach((node, i) => { node.style.maxWidth = 'none'; node.textContent = compactValue(values[i]!); });
    const fits = () => endpoints.reduce((sum, node) => sum + node.getBoundingClientRect().width, 0) <= width - 2;
    if (fits()) return;
    // Share a power of ten when the narrow physical bin axis cannot fit two
    // decimal captions. Only display precision changes; endpoints/ticks do not.
    const magnitude = Math.max(...values.map(Math.abs));
    const exponent = magnitude ? Math.floor(Math.log10(magnitude)) : 0;
    for (const precision of [2, 1]) {
      endpoints.forEach((node, i) => {
        node.textContent = String(Number((values[i]! / 10 ** exponent).toPrecision(precision)))
          .replace(/^(-?)0\./, '$1.').replace('e+', 'e');
      });
      if (fits()) break;
    }
    if (exponent) {
      const scale = document.createElement('span');
      scale.className = 'distribution-exponent';
      const superscript = String(exponent).split('').map(c => '⁰¹²³⁴⁵⁶⁷⁸⁹' [Number(c)] ?? '⁻').join('');
      scale.textContent = `×10${superscript}`;
      scale.title = `Endpoint captions multiplied by 10^${exponent}`;
      scale.setAttribute('aria-hidden', 'true');
      this.ruler.append(scale);
    }
    // Extremely constrained tracks still keep exact values in tooltips/ARIA.
    if (!fits()) endpoints.forEach(node => { node.style.maxWidth = '50%'; });
  }
}
