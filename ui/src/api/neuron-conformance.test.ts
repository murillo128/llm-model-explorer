// @vitest-environment node
import { expect, it } from 'vitest';
import fixtures from '../../../api/fixtures/neurons.json';
import { validateSchema } from './validation';
import { validateNeuronRequest } from './neuron-validation';
import { LmexDecoder } from './lmex-decoder';
import { ApiFailure } from './errors';
import type { components } from './generated/types';

type S = components['schemas'];

function edited(base: unknown, edits: unknown[][]): unknown {
  const value = structuredClone(base);
  for (const [path, replacement] of edits) {
    const keys = path as (string | number)[];
    let target = value as Record<string | number, unknown>;
    for (const key of keys.slice(0, -1)) target = target[key] as typeof target;
    target[keys.at(-1)!] = structuredClone(replacement);
  }
  return value;
}

for (const fixture of fixtures.cases) it(`neuron contract: ${fixture.name}`, () => {
  const value = edited(fixtures.bases[fixture.base as keyof typeof fixtures.bases], fixture.edits);
  const cap = edited(fixtures.bases.capabilities, 'context_edits' in fixture ? fixture.context_edits! : []);
  const validate = () => {
    validateSchema(fixture.schema as keyof S, value);
    if (fixture.schema === 'NeuronActivationsRequest' || fixture.schema === 'NeuronComparisonRequest') {
      const capability = validateSchema('NeuronCapabilities', cap);
      validateNeuronRequest(value as S['NeuronActivationsRequest'] | S['NeuronComparisonRequest'], capability);
    }
  };
  if (fixture.valid) expect(validate).not.toThrow();
  else expect(validate).toThrow(ApiFailure);
});

function frame(type: number, payload: Uint8Array = new Uint8Array(0)): Uint8Array {
  const bytes = new Uint8Array(12 + payload.length);
  bytes.set([0x4c, 0x4d, 0x45, 0x58, type]);
  new DataView(bytes.buffer).setUint32(8, payload.length, true);
  bytes.set(payload, 12);
  return bytes;
}

for (const fixture of fixtures.payload_cases) it(`neuron payload: ${fixture.name}`, () => {
  const meta = fixtures.bases[fixture.base as 'activations' | 'probabilities'];
  const validate = () => {
    const decoder = new LmexDecoder();
    decoder.push(frame(1, new TextEncoder().encode(JSON.stringify(meta))));
    // One-byte chunks exercise split float32 validation, independent of DATA framing.
    for (const byte of frame(2, Buffer.from(fixture.data_hex, 'hex'))) decoder.push(Uint8Array.of(byte));
    decoder.push(frame(4));
    expect(decoder.finish().kind).toBe('complete');
  };
  if (fixture.valid) expect(validate).not.toThrow();
  else expect(validate).toThrow(ApiFailure);
});

it.each([NaN, Infinity, -Infinity])('rejects non-finite neuron parameters: %s', factor => {
  const value = structuredClone(fixtures.bases.comparison);
  const conditions = [value.conditions[0], { id: 'scale', mode: 'scale', scope: 'all', factor }];
  expect(() => validateSchema('NeuronComparisonRequest', { ...value, conditions })).toThrow(ApiFailure);
});
