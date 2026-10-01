import type { ModelSummary } from './session-controller';

const encodings: Record<string, string> = {
  'bnb-nf4-dq': 'NF4 4-bit, double quantization',
  'gptq-int4': 'GPTQ 4-bit',
  nvfp4: 'NVFP4 4-bit',
  'compressed-tensors-w4a16-int4': 'W4A16 INT4',
};

function displayLabel(name: string, shorten: boolean): string {
  return name.split(' + LoRA ').map((part) => {
    const label = shorten ? part.replace(/^[^/\s]+\/(?=[^/]+$)/, '') : part;
    return label.replace(/\(([^()]+)\)$/, (match, encoding: string) =>
      Object.hasOwn(encodings, encoding) ? `(${encodings[encoding]})` : match);
  }).join(' + LoRA ');
}

export function modelLabels(models: readonly ModelSummary[]): Map<string, string> {
  const short = models.map((model) => displayLabel(model.display_name, true));
  const qualified = models.map((model, index) => short.indexOf(short[index]!) === short.lastIndexOf(short[index]!)
    ? short[index]! : displayLabel(model.display_name, false));
  return new Map(models.map((model, index) => {
    const label = qualified[index]!;
    // Different revisions or custom display names can still coincide. Preserve
    // unambiguous selection without changing the underlying API identifiers.
    return [model.id, qualified.indexOf(label) === qualified.lastIndexOf(label)
      ? label : `${label} (${model.id})`];
  }));
}
