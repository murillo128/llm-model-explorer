import fixtures from '../../../api/fixtures/neurons.json';
import architecture from '../../../api/fixtures/architecture.json';
import { validateSchema } from '../api/validation';

/** Contract-owned layer-7 identities and asymmetric shapes, shared by UI tests. */
export const neuronCapabilities = validateSchema('NeuronCapabilities', structuredClone(fixtures.bases.capabilities));
export const neuronSession = { id: neuronCapabilities.session_id, model_id: neuronCapabilities.model_id, model_revision: neuronCapabilities.model_revision };
export const neuronInventory = validateSchema('TensorInventory', { coverage: 'complete', diagnostics: [],
  tensors: Object.entries(neuronCapabilities.modules[0]!.weights).map(([projection, binding]) => ({
    id: binding.tensor_id, name: `${neuronCapabilities.modules[0]!.module.name}.${projection}_proj.weight`,
    path: ['layer7', projection], shape: binding.shape, rank: 2, numel: 6, storage_dtype: 'float32', logical_dtype: 'float32',
  })),
});
export const neuronArchitecture = validateSchema('ArchitectureResponse', structuredClone(architecture.response));
if (neuronArchitecture.status !== 'available') throw new Error('Expected fixture graph');
export const neuronGraph = neuronArchitecture.graph;
neuronArchitecture.model_id = neuronSession.model_id;
const node = neuronArchitecture.graph.nodes.find(n => n.id === 'linear1')!;
node.label = 'MLP layer 7';
node.references = [neuronCapabilities.modules[0]!.module];
node.parameter_ids = [];
neuronCapabilities.modules[0]!.graph_binding = { graph_id: neuronArchitecture.graph.graph_id, node_id: node.id };
export const unavailableNeurons = validateSchema('NeuronCapabilities', { ...fixtures.bases.unavailable,
  session_id: neuronSession.id, model_id: neuronSession.model_id, model_revision: neuronSession.model_revision });
