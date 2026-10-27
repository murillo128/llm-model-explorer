import type { components } from './generated/types';
import { requireProtocol } from './errors';

type S = components['schemas'];
type NeuronMetadata = S['NeuronActivationsMetadata'] | S['NeuronComparisonMetadata'];

// JSON object member order is not part of the wire contract; array order is.
function sameRecord(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const left = a as Record<string, unknown>, right = b as Record<string, unknown>;
  const keys = Object.keys(left);
  return keys.length === Object.keys(right).length && keys.every(key => Object.hasOwn(right, key) && sameRecord(left[key], right[key]));
}

/** Producer metadata must describe this exact admitted request, not just equal-sized data. */
export function validateNeuronAssociation(metadata: NeuronMetadata, session: S['Session'],
  capabilities: S['NeuronCapabilities'], request: S['NeuronActivationsRequest'] | S['NeuronComparisonRequest']): void {
  requireProtocol(metadata.session_id === session.id && metadata.model_id === session.model_id
    && metadata.model_revision === session.model_revision && metadata.model_revision === request.model_revision
    && sameRecord(metadata.module, capabilities.modules.find(module => module.module_id === request.module_id))
    && sameRecord(metadata.execution, capabilities.execution) && sameRecord(metadata.limits, capabilities.limits)
    && sameRecord(metadata.neuron_indices, request.neuron_indices) && sameRecord(metadata.token_ids, request.token_ids), 'Neuron response identity mismatch');
  if ('conditions' in request) requireProtocol(metadata.kind === 'neuron_comparison'
    && sameRecord(metadata.conditions, request.conditions) && metadata.top_k === request.top_k
    && metadata.max_new_tokens === request.max_new_tokens && metadata.decoding === request.decoding, 'Neuron comparison configuration mismatch');
  else requireProtocol(metadata.kind === 'neuron_activations', 'Unexpected neuron result kind');
}

function limits(value: S['NeuronLimits']): void {
  requireProtocol(value.max_input_tokens === Math.min(512, value.context_length)
    && value.max_new_tokens === Math.min(64, value.context_length - 1)
    && value.max_top_k === Math.min(20, value.vocabulary_size), 'Invalid effective neuron limits');
  requireProtocol(value.eos_token_ids.every((id, i, ids) => id < value.vocabulary_size
    && (i === 0 || id > ids[i - 1]!)), 'Invalid EOS identities');
}

function moduleBinding(value: S['NeuronModule']): void {
  requireProtocol(Number.isSafeInteger(value.intermediate_size * value.hidden_size * 4), 'Unsafe neuron binding size');
  const ids: string[] = [];
  for (const name of ['gate', 'up', 'down'] as const) {
    const binding = value.weights[name];
    const shape = name === 'down' ? [value.hidden_size, value.intermediate_size] : [value.intermediate_size, value.hidden_size];
    requireProtocol(binding.axis === (name === 'down' ? 'column' : 'row')
      && binding.shape.every((size, i) => size === shape[i]), 'Invalid neuron weight binding');
    ids.push(binding.tensor_id);
  }
  requireProtocol(new Set(ids).size === 3, 'Repeated projection binding');
}

function conditions(value: S['NeuronComparisonRequest'] | S['NeuronComparisonMetadata']): void {
  requireProtocol(value.conditions[0]?.mode === 'baseline'
    && value.conditions.slice(1).every(c => c.mode !== 'baseline')
    && new Set(value.conditions.map(c => c.id)).size === value.conditions.length, 'Invalid baseline or condition order');
}

/** Contextual checks follow schema validation; no execution or tokenization occurs here. */
export function validateNeuronRequest(value: S['NeuronActivationsRequest'] | S['NeuronComparisonRequest'],
  capabilities: S['NeuronCapabilities']): void {
  requireProtocol(capabilities.observation.status === 'available' && capabilities.limits !== null, 'Neuron observation unavailable');
  requireProtocol(value.model_revision === capabilities.model_revision, 'Neuron revision mismatch');
  const selected = capabilities.modules.find(m => m.module_id === value.module_id);
  requireProtocol(selected, 'Unknown neuron module');
  requestBounds(value, capabilities.limits, selected);
  if ('conditions' in value) {
    requireProtocol(value.conditions.length === 1 || capabilities.intervention.status === 'available', 'Neuron intervention unavailable');
    requireProtocol(value.max_new_tokens === 0 || capabilities.generation.status === 'available', 'Neuron generation unavailable');
  }
}

function requestBounds(value: Pick<S['NeuronActivationsRequest'], 'token_ids' | 'neuron_indices'>
  & Partial<Pick<S['NeuronComparisonRequest'], 'conditions' | 'top_k' | 'max_new_tokens'>>,
  bounds: S['NeuronLimits'], selected: S['NeuronModule']): void {
  requireProtocol(value.token_ids.length <= bounds.max_input_tokens
    && value.token_ids.every(id => id < bounds.vocabulary_size)
    && value.neuron_indices.every(id => id < selected.intermediate_size), 'Neuron input out of bounds');
  if (value.conditions) {
    requireProtocol(value.top_k! <= bounds.max_top_k && value.max_new_tokens! <= bounds.max_new_tokens
      && value.token_ids.length + value.max_new_tokens! <= bounds.context_length, 'Neuron comparison out of bounds');
  }
}

function metadata(value: NeuronMetadata): void {
  limits(value.limits);
  moduleBinding(value.module);
  requestBounds(value, value.limits, value.module);
  let name = 'activations';
  let shape = [value.token_ids.length, value.neuron_indices.length];
  if (value.kind === 'neuron_comparison') {
    conditions(value);
    const candidates = value.candidate_token_ids;
    requireProtocol(candidates.every((id, i) => id < value.limits.vocabulary_size
      && (i === 0 || id > candidates[i - 1]!)), 'Invalid neuron candidate order/range');
    requireProtocol(value.top_token_ids.length === value.conditions.length
      && value.top_token_ids.every(ids => ids.length === value.top_k), 'Invalid neuron top-k shape');
    const union = [...new Set(value.top_token_ids.flat())].sort((a, b) => a - b);
    requireProtocol(union.length === candidates.length && union.every((id, i) => id === candidates[i]), 'Invalid neuron candidate union');
    requireProtocol(value.continuations.length === value.conditions.length, 'Invalid continuation count');
    value.continuations.forEach((continuation, i) => {
      const ids = continuation.token_ids;
      const isEos = (id: number) => value.limits.eos_token_ids.includes(id);
      requireProtocol(continuation.condition_id === value.conditions[i]!.id
        && ids.length <= value.max_new_tokens && ids.every(id => id < value.limits.vocabulary_size), 'Invalid continuation identity/range');
      if (value.max_new_tokens === 0) {
        requireProtocol(ids.length === 0 && continuation.text === '' && continuation.finish_reason === 'not_requested', 'Unexpected continuation');
      } else if (continuation.finish_reason === 'eos') {
        requireProtocol(ids.length > 0 && isEos(ids.at(-1)!) && !ids.slice(0, -1).some(isEos), 'Invalid EOS stopping');
      } else {
        requireProtocol(continuation.finish_reason === 'length' && ids.length === value.max_new_tokens && !ids.some(isEos), 'Invalid length stopping');
      }
    });
    name = 'probabilities';
    shape = [value.conditions.length, candidates.length];
  }
  const section = value.sections[0]!;
  const size = 4 * shape[0]! * shape[1]!;
  requireProtocol(section.name === name && section.offset === 0 && section.byte_length === size
    && value.byte_length === size && section.shape.every((n, i) => n === shape[i]), 'Invalid neuron section geometry');
}

export function validateNeuronFields(name: string, value: unknown): void {
  if (name === 'NeuronLimits') limits(value as S['NeuronLimits']);
  else if (name === 'NeuronModule') moduleBinding(value as S['NeuronModule']);
  else if (name === 'NeuronComparisonRequest') conditions(value as S['NeuronComparisonRequest']);
  else if (name === 'NeuronActivationsMetadata' || name === 'NeuronComparisonMetadata') metadata(value as NeuronMetadata);
  else if (name === 'NeuronCapabilities') {
    const cap = value as S['NeuronCapabilities'];
    if (cap.observation.status === 'unavailable') {
      requireProtocol(cap.intervention.status === 'unavailable' && cap.generation.status === 'unavailable'
        && cap.execution === null && cap.limits === null && cap.modules.length === 0, 'Invalid unavailable neuron capability');
    } else {
      requireProtocol(cap.execution !== null && cap.limits !== null && cap.modules.length > 0, 'Incomplete neuron capability');
      limits(cap.limits);
      cap.modules.forEach(moduleBinding);
      requireProtocol(new Set(cap.modules.map(m => m.module_id)).size === cap.modules.length
        && new Set(cap.modules.map(m => m.module.name)).size === cap.modules.length
        && cap.modules.every((m, i) => i === 0 || m.layer_index > cap.modules[i - 1]!.layer_index), 'Invalid module identities/order');
    }
  }
}
