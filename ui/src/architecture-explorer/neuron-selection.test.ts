import { expect, it } from 'vitest';
import { neuronCapabilities, neuronInventory, neuronSession } from '../test/neuron-fixture';
import { GraphViews } from './graph';
import { neuronForInspection, neuronWeight, validNeuronIndex } from './neuron-selection';
import type { ArchitectureSelection } from './ArchitectureCanvas';
import { neuronArchitecture } from '../test/neuron-fixture';

const module = neuronCapabilities.modules[0]!;
const graph = neuronArchitecture.status === 'available' ? neuronArchitecture.graph : undefined;
const inspected: ArchitectureSelection = { sessionId: neuronSession.id, modelId: neuronSession.model_id, graphId: graph!.graph_id,
  node: graph!.nodes.find(n => n.id === 'linear1')!, trigger: document.body };

it('binds only the exact concrete graph instance, never a representative or another session', () => {
  expect(neuronForInspection(neuronCapabilities, inspected)?.layer_index).toBe(7);
  for (const patch of [{ structureOnly: { label: 'Shared MLP', role: 'mlp' } }, { graphId: 'old' }, { sessionId: 'old' },
    { modelId: 'old' }, { node: graph!.nodes.find(n => n.id === 'linear0')! }]) {
    expect(neuronForInspection(neuronCapabilities, { ...inspected, ...patch })).toBeUndefined();
  }
  expect(neuronForInspection({ ...neuronCapabilities, modules: [module, { ...module, module_id: 'other' }] }, inspected)).toBeUndefined();
});
it('uses gate/up rows and down columns through the exact inventory IDs and native shapes', () => {
  expect(neuronWeight(module, 'gate', neuronInventory)).toMatchObject({ axis: 'row', tensor: { id: 'gate7', shape: [3, 2] } });
  expect(neuronWeight(module, 'up', neuronInventory)).toMatchObject({ axis: 'row', tensor: { id: 'up7', shape: [3, 2] } });
  expect(neuronWeight(module, 'down', neuronInventory)).toMatchObject({ axis: 'column', tensor: { id: 'down7', shape: [2, 3] } });
  expect(neuronWeight(module, 'down', { ...neuronInventory, tensors: [] })).toHaveProperty('reason');
  const wrong = structuredClone(neuronInventory);
  wrong.tensors[2]!.shape = [3, 2];
  expect(neuronWeight(module, 'down', wrong)).toHaveProperty('reason');
  const axis = structuredClone(module); axis.weights.down.axis = 'row';
  expect(neuronWeight(axis, 'down', neuronInventory)).toHaveProperty('reason');
});
it('accepts the real index endpoints and rejects invalid coordinates', () => {
  expect([0, 2].map(index => validNeuronIndex(index, module))).toEqual([true, true]);
  for (const index of [-1, 3, 1.5, NaN, Infinity]) expect(validNeuronIndex(index, module)).toBe(false);
});
it('retains only same-session coordinates across explorers and clears them on revision/model/backend replacement', () => {
  const views = new GraphViews();
  Object.assign(views.neuronView(neuronSession), { open: true, moduleId: 'mlp-7', index: 2, scroll: 80 });
  expect(views.neuronView(neuronSession)).toMatchObject({ open: true, index: 2, scroll: 80 });
  expect(views.neuronView({ ...neuronSession, model_revision: 'new' })).toMatchObject({ open: false, moduleId: null });
  expect(views.neuronView({ ...neuronSession, model_id: 'different' })).toMatchObject({ open: false, moduleId: null });
  expect(new GraphViews().neuronView(neuronSession).moduleId).toBeNull();
});
