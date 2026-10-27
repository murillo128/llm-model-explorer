import type { components } from '../api/generated/types';
import { validateSchema } from '../api/validation';
import { validateNeuronRequest } from '../api/neuron-validation';
import type { CurrentTokenization } from '../tokenizer/PromptTokenizer';
import type { NeuronSelectionContext } from './neuron-selection';

type S = components['schemas'];
export type ExperimentMode = 'activations' | 'compare';
export type ConditionDraft = { id: string; mode: 'zero' | 'scale' | 'add'; value: string; scope: 'prompt' | 'continuation' | 'all' };
export type ExperimentConfig = { mode: ExperimentMode; conditions: ConditionDraft[]; topK: string; maxNewTokens: string };
export type ExperimentMetadata = S['NeuronActivationsMetadata'] | S['NeuronComparisonMetadata'];
export interface ExperimentSnapshot {
  session: S['Session']; capabilities: S['NeuronCapabilities'];
  input: S['TokenizeResponse']; request: S['NeuronActivationsRequest'] | S['NeuronComparisonRequest'];
  mode: ExperimentMode;
}
export function initialExperiment(capabilities: S['NeuronCapabilities']): ExperimentConfig {
  return { mode: 'activations', conditions: capabilities.intervention.status === 'available' ? [
    { id: 'zero', mode: 'zero', value: '0', scope: 'all' },
    { id: 'scale', mode: 'scale', value: '2', scope: 'all' },
  ] : [], topK: String(Math.min(5, capabilities.limits?.max_top_k ?? 1)),
  maxNewTokens: String(capabilities.generation.status === 'available' ? Math.min(32, capabilities.limits?.max_new_tokens ?? 0) : 0) };
}
function integer(value: string, name: string, minimum: number, maximum: number): number {
  const number = Number(value);
  if (!value.trim() || !Number.isSafeInteger(number) || number < minimum || number > maximum) throw new Error(`${name} must be a whole number from ${minimum} to ${maximum}.`);
  return number;
}
export function experimentSnapshot(context: NeuronSelectionContext, current: CurrentTokenization | undefined, config: ExperimentConfig): ExperimentSnapshot {
  if (!context.selection.isCurrent() || !current || current.signal.aborted) throw new Error('Waiting for current tokenization.');
  const { capabilities, session, module } = context, limits = capabilities.limits;
  if (capabilities.observation.status !== 'available' || !limits) throw new Error('Neuron observation unavailable.');
  if (!context.neuronIndices.length) throw new Error('Choose a valid neuron index in the inspector.');
  const token_ids = current.data.tokens.map(token => token.id);
  if (!token_ids.length) throw new Error('Enter a prompt with at least one token.');
  if (token_ids.length > limits.max_input_tokens) throw new Error(`Prompt exceeds the ${limits.max_input_tokens} token limit.`);
  const request: S['NeuronActivationsRequest'] = { model_revision: session.model_revision, module_id: module.module_id,
    neuron_indices: [...context.neuronIndices], token_ids };
  let body: ExperimentSnapshot['request'] = request;
  if (config.mode === 'compare') {
    const max_new_tokens = integer(config.maxNewTokens, 'New tokens', 0, limits.max_new_tokens);
    if (token_ids.length + max_new_tokens > limits.context_length) throw new Error(`Prompt and new tokens must fit the ${limits.context_length} token context.`);
    const conditions: S['NeuronCondition'][] = [{ id: 'baseline', mode: 'baseline' }, ...config.conditions.map(condition => {
      const common = { id: condition.id, scope: condition.scope };
      if (condition.mode === 'zero') return { ...common, mode: 'zero' as const };
      const value = Number(condition.value);
      if (!condition.value.trim() || !Number.isFinite(value) || Math.abs(value) > 3.4028234663852886e38) throw new Error(`${condition.id}: enter a finite float32 ${condition.mode === 'scale' ? 'factor' : 'delta'}.`);
      return condition.mode === 'scale' ? { ...common, mode: 'scale' as const, factor: value } : { ...common, mode: 'add' as const, delta: value };
    })];
    body = { ...request, conditions, max_new_tokens, top_k: integer(config.topK, 'Top-k', 1, limits.max_top_k), decoding: 'greedy' };
    if (conditions.length > limits.max_conditions) throw new Error(`At most ${limits.max_conditions} conditions, including baseline.`);
    if (conditions.length > 1 && capabilities.intervention.status !== 'available') throw new Error('Interventions are unavailable. Remove intervention conditions to compare baseline only.');
    if (max_new_tokens > 0 && capabilities.generation.status !== 'available') throw new Error('Generation is unavailable. Set new tokens to 0 for the probability table.');
  }
  validateSchema(config.mode === 'compare' ? 'NeuronComparisonRequest' : 'NeuronActivationsRequest', body);
  validateNeuronRequest(body, capabilities);
  return structuredClone({ session, capabilities, input: current.data, request: body, mode: config.mode });
}
export function experimentKey(snapshot: ExperimentSnapshot): string {
  return JSON.stringify([snapshot.session, snapshot.input, snapshot.mode, snapshot.request]);
}
export function signed(value: number): string { return Object.is(value, -0) ? '-0' : value > 0 ? `+${value}` : String(value); }
export function percentagePoints(value: number, baseline: number): string {
  const delta = (value - baseline) * 100;
  return `${delta > 0 ? '+' : ''}${delta.toFixed(4)} pp`;
}
export function conditionLabel(condition: S['NeuronCondition']): string {
  return condition.mode === 'baseline' ? 'Baseline' : `${condition.id} · ${condition.mode}${condition.mode === 'scale' ? ` × ${condition.factor}` : condition.mode === 'add' ? ` ${signed(condition.delta)}` : ' = 0'} · ${condition.scope}`;
}

/** Explicit allowlist: this is a bounded recipe, never a result/transport dump. */
export function exportConfiguration(snapshot: ExperimentSnapshot, metadata: ExperimentMetadata, includeText = false): string {
  const manifest = {
    format: 'llmex-neuron-experiment', format_revision: 1, operation_kind: metadata.kind,
    model_id: metadata.model_id, session_id: metadata.session_id, model_revision: metadata.model_revision,
    execution: metadata.execution, limits: metadata.limits, module: metadata.module,
    token_ids: metadata.token_ids, neuron_indices: metadata.neuron_indices,
    tokenizer: { add_special_tokens: snapshot.input.add_special_tokens },
    ...(includeText ? { user_text: snapshot.input.text } : {}),
    ...(metadata.kind === 'neuron_comparison' ? { conditions: metadata.conditions, top_k: metadata.top_k,
      max_new_tokens: metadata.max_new_tokens, decoding: metadata.decoding } : {}),
    replay_notice: 'Configuration only. Token IDs can disclose the prompt. Requires independently matching local checkpoint content and versions and a fresh valid session. The opaque revision is not a portable hash; bitwise equality is not guaranteed.',
  };
  const json = JSON.stringify(manifest, null, 2);
  if (new TextEncoder().encode(json).byteLength > 1_048_576) throw new Error('Configuration exceeds the 1 MiB export limit. Exclude prompt text or shorten the input.');
  return json;
}
