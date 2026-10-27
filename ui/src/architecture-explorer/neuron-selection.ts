import type { components } from '../api/generated/types';
import type { ExplorerContextValue } from '../app/explorer-context';
import type { ArchitectureSelection } from './ArchitectureCanvas';

type S = components['schemas'];
export type NeuronModule = S['NeuronModule'];
export type ProjectionWeight = keyof NeuronModule['weights'];
export const projectionWeights = ['gate', 'up', 'down'] as const;

/** Presentation only; the backend-scoped GraphViews owner retains this across explorer switches. */
export interface NeuronViewState {
  sessionId: string;
  modelId: string;
  modelRevision: string;
  open: boolean;
  moduleId: string | null;
  index: number;
  scroll: number;
}

/** Stable composition boundary for the experimental view. No inferred graph/tensor identity. */
export interface NeuronSelectionContext extends Pick<ExplorerContextValue, 'client' | 'session' | 'selection'> {
  capabilities: S['NeuronCapabilities'];
  module: NeuronModule;
  neuronIndices: readonly number[];
}

export function validNeuronIndex(index: number, module: NeuronModule): boolean {
  return Number.isSafeInteger(index) && index >= 0 && index < module.intermediate_size;
}

export function neuronForInspection(capabilities: S['NeuronCapabilities'] | undefined, inspected: ArchitectureSelection): NeuronModule | undefined {
  if (inspected.structureOnly || inspected.boundary || !inspected.node || !capabilities
    || capabilities.session_id !== inspected.sessionId || capabilities.model_id !== inspected.modelId) return;
  const matches = capabilities.modules.filter(module => module.graph_binding?.graph_id === inspected.graphId
    && module.graph_binding.node_id === inspected.node!.id);
  return matches.length === 1 ? matches[0] : undefined;
}

export function neuronWeight(module: NeuronModule, projection: ProjectionWeight, inventory: S['TensorInventory']):
  { tensor: S['TensorDescriptor']; axis: 'row' | 'column' } | { reason: string } {
  const binding = module.weights[projection];
  const matches = inventory.tensors.filter(tensor => tensor.id === binding.tensor_id);
  if (matches.length !== 1) return { reason: 'The bound tensor is unavailable in the current inventory.' };
  const tensor = matches[0]!;
  const axis = projection === 'down' ? 'column' : 'row';
  const shape = projection === 'down' ? [module.hidden_size, module.intermediate_size] : [module.intermediate_size, module.hidden_size];
  if (binding.axis !== axis || binding.shape.length !== 2 || tensor.rank !== 2 || tensor.shape.length !== 2
    || shape.some((size, i) => size !== binding.shape[i] || size !== tensor.shape[i])
    || tensor.numel !== shape[0]! * shape[1]! || tensor.logical_dtype !== 'float32') {
    return { reason: 'The weight shape or orientation does not match this neuron binding.' };
  }
  return { tensor, axis };
}
