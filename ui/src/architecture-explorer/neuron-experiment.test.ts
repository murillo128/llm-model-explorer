// @vitest-environment node
import { expect, it } from 'vitest';
import fixtures from '../../../api/fixtures/neurons.json';
import { validateSchema } from '../api/validation';
import { ApiClient } from '../api/client';
import { Lifetime } from '../app/lifetime';
import { conditionLabel, experimentKey, experimentSnapshot, exportConfiguration, initialExperiment, percentagePoints } from './neuron-experiment';
import type { NeuronSelectionContext } from './neuron-selection';
import type { CurrentTokenization } from '../tokenizer/PromptTokenizer';

export function fixture() {
  const capabilities = validateSchema('NeuronCapabilities', structuredClone(fixtures.bases.capabilities));
  const context: NeuronSelectionContext = { client: new ApiClient({ backendBaseUrl: 'https://fixture.example' }), capabilities, module: capabilities.modules[0]!, neuronIndices: [2, 0], selection: new Lifetime(),
    session: { id: capabilities.session_id, model_id: capabilities.model_id, model_revision: capabilities.model_revision } };
  const signal = new AbortController();
  const current: CurrentTokenization = { signal: signal.signal, data: { text: 'x x', add_special_tokens: false,
    tokens: [0, 1].map(index => ({ index, id: 1, token: 'x', decoded: 'x', special: false, start: index * 2, end: index * 2 + 1 })) } };
  return { context, current, signal, config: initialExperiment(capabilities) };
}

it('captures immutable exact input, duplicate IDs, ordered neuron binding and session; edits make a different identity', () => {
  const { context, current, config } = fixture();
  const snapshot = experimentSnapshot(context, current, config), key = experimentKey(snapshot);
  current.data.text = 'edited'; current.data.tokens[0]!.id = 2;
  context.neuronIndices = [0]; context.module.layer_index = 9;
  expect(snapshot.input.text).toBe('x x'); expect(snapshot.request).toEqual(fixtures.bases.observation);
  expect(snapshot.capabilities.modules[0]!.layer_index).toBe(7);
  expect(experimentKey(snapshot)).toBe(key);
  expect(experimentKey(experimentSnapshot(context, current, config))).not.toBe(key);
});
it.each(['NaN', 'Infinity', '-Infinity', '1e39', ''])('rejects non-finite/out-of-float32/empty intervention input %s before sending', value => {
  const { context, current, config } = fixture(); config.mode = 'compare'; config.conditions[1]!.value = value;
  expect(() => experimentSnapshot(context, current, config)).toThrow('finite float32');
});
it('distinguishes multiply/add and scope, accepts table-only and finite negative factors, and fixes baseline first', () => {
  const { context, current, config } = fixture(); config.mode = 'compare'; config.maxNewTokens = '0';
  config.conditions = [{ id: 'multiply', mode: 'scale', value: '-2', scope: 'continuation' }, { id: 'offset', mode: 'add', value: '2', scope: 'prompt' }];
  const snapshot = experimentSnapshot(context, current, config);
  expect(snapshot.request).toMatchObject({ max_new_tokens: 0, decoding: 'greedy', conditions: [
    { id: 'baseline', mode: 'baseline' }, { id: 'multiply', mode: 'scale', factor: -2, scope: 'continuation' }, { id: 'offset', mode: 'add', delta: 2, scope: 'prompt' },
  ] });
  expect(conditionLabel({ id: 's', mode: 'scale', factor: 2, scope: 'all' })).toContain('scale × 2');
  expect(conditionLabel({ id: 'a', mode: 'add', delta: 2, scope: 'all' })).toContain('add +2');
});
it('rejects invalid index, empty/stale tokenization and over-context generation; unavailable generation permits table-only', () => {
  const { context, current, config, signal } = fixture();
  expect(() => experimentSnapshot({ ...context, neuronIndices: [] }, current, config)).toThrow('valid neuron');
  expect(() => experimentSnapshot({ ...context, neuronIndices: [3] }, current, config)).toThrow();
  expect(() => experimentSnapshot(context, { ...current, data: { ...current.data, tokens: [] } }, config)).toThrow('at least one');
  config.mode = 'compare'; context.capabilities.limits!.context_length = 32;
  expect(() => experimentSnapshot(context, current, config)).toThrow('must fit the 32 token context');
  config.maxNewTokens = '0'; context.capabilities.generation = { status: 'unavailable', reason: { code: 'unsupported_model', message: 'Unavailable', recovery: 'Use table only' } };
  expect(() => experimentSnapshot(context, current, config)).not.toThrow();
  signal.abort(); expect(() => experimentSnapshot(context, current, config)).toThrow('current tokenization');
});
it('uses percentage points, without normalizing the candidate subset', () => {
  expect(percentagePoints(.5, .125)).toBe('+37.5000 pp');
  expect(percentagePoints(.125, .5)).toBe('-37.5000 pp');
  expect(percentagePoints(.25, .25)).toBe('0.0000 pp');
});
it('exports only a bounded public recipe; prompt text is opt-in and outputs/private extras are excluded', () => {
  const { context, current, config } = fixture();
  const snapshot = experimentSnapshot(context, current, config);
  const metadata = { ...validateSchema('NeuronComparisonMetadata', fixtures.bases.probabilities), private_path: '/private/checkpoint', activations: [1, 2], probabilities: [0.5] };
  const json = exportConfiguration(snapshot, metadata), manifest = JSON.parse(json);
  expect(Object.keys(manifest).sort()).toEqual(['format', 'format_revision', 'operation_kind', 'model_id', 'session_id', 'model_revision', 'execution', 'limits', 'module', 'token_ids', 'neuron_indices', 'tokenizer', 'conditions', 'top_k', 'max_new_tokens', 'decoding', 'replay_notice'].sort());
  expect(manifest).toMatchObject({ format: 'llmex-neuron-experiment', format_revision: 1, token_ids: [1, 1], neuron_indices: [2, 0], module: fixtures.bases.capabilities.modules[0], execution: fixtures.bases.capabilities.execution });
  expect(json).not.toContain('/private'); expect(json).not.toContain('continuations'); expect(json).not.toContain('user_text');
  expect(JSON.parse(exportConfiguration(snapshot, metadata, true)).user_text).toBe('x x');
  snapshot.input.text = 'x'.repeat(1_048_576);
  expect(() => exportConfiguration(snapshot, metadata, true)).toThrow('1 MiB');
});
