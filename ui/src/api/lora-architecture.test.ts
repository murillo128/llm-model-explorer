import { expect, it } from 'vitest';
import type { TensorDescriptor } from '../app/session-controller';
import { validateArchitecture } from './architecture-validation';
import { validateResponse } from './validation';
import { inspectionFixture } from '../../tests/architecture-inspection-fixture';

const tensor = (id: string, shape: number[]): TensorDescriptor => ({
  id, name: id, path: [id], shape, rank: shape.length,
  numel: shape.reduce((count, dimension) => count * dimension, 1),
  storage_dtype: 'float32', logical_dtype: 'float32',
});

it('validates and exposes native logical LoRA factor bindings in the generic graph', () => {
  const tensors = [tensor('A', [2, 3]), tensor('B', [3, 2]), tensor('vector', [5]), tensor('lora-A', [2, 2]), tensor('lora-B', [3, 2])];
  const response = inspectionFixture(tensors, 'lab/alpha', true);
  for (const parameter of response.graph.parameters.filter((item) => item.id === 'lora-a' || item.id === 'lora-b')) {
    const factor = parameter.id === 'lora-a' ? 'A' : 'B';
    parameter.storage[0]!.name = `base_model.model.model.layers.1.self_attn.q_proj.lora_${factor}.default.weight`;
    parameter.storage[0]!.role = 'adapter_factor';
  }
  validateResponse('getArchitecture', response);
  const context = { modelId: 'lab/alpha' };
  const validated = validateArchitecture(response, context);
  if (validated.status !== 'available') throw new Error('LoRA graph unexpectedly unavailable.');
  const graph = validated.graph;
  expect(graph.nodes.map((node) => node.id)).toEqual(expect.arrayContaining(['lora_A', 'lora_B', 'lora_scale', 'lora_add']));
  expect(graph.parameters.find((parameter) => parameter.id === 'lora-a')?.inspection).toEqual({ status: 'available', tensor_id: 'lora-A' });
  expect(graph.parameters.find((parameter) => parameter.id === 'lora-b')?.inspection).toEqual({ status: 'available', tensor_id: 'lora-B' });

  const invalid = structuredClone(response);
  invalid.graph.parameters.find((parameter) => parameter.id === 'lora-a')!.storage[0]!.shape = [-1, 2];
  expect(() => validateArchitecture(invalid, context)).toThrow('Invalid ArchitectureResponse schema');
});
