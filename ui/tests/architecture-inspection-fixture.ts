import { contractResponse } from './architecture-fixtures';
import type { TensorDescriptor } from '../src/app/session-controller';

/** Synthetic bindings deliberately use different asymmetric weights per instance. */
export function inspectionFixture(tensors: TensorDescriptor[], modelId: string) {
  const response = structuredClone(contractResponse);
  response.model_id = modelId;
  const graph = response.graph;
  const native = (id: string, tensorId: string) => {
    const tensor = tensors.find((t) => t.id === tensorId)!;
    return { id, name: tensor.name, binding: 'native' as const,
      logical_shape: tensor.shape.map((value) => ({ kind: 'constant' as const, value })),
      storage: [{ name: tensor.name, dtype: tensor.storage_dtype, shape: tensor.shape }],
      inspection: { status: 'available' as const, tensor_id: tensor.id }, provenance: [] };
  };
  graph.parameters = [...graph.parameters.filter((p) => p.inspection.status === 'unavailable'), native('first', 'A'), native('second', 'B'), native('vector-weight', 'vector')];
  for (const [nodeId, parameter] of [['linear0', 'first'], ['linear1', 'second']] as const) {
    const node = graph.nodes.find((n) => n.id === nodeId)!;
    node.parameter_ids = [parameter, 'vector-weight', 'quantized', 'fused', 'unresolved', 'volume'];
    node.references = [{ kind: 'parameter', parameter_id: parameter }, { kind: 'module', name: `layers.${nodeId === 'linear0' ? 0 : 1}.linear` }];
    node.description = 'A concrete repeated-layer operation.';
    node.formula = 'Y = XWᵀ';
  }
  if (modelId === 'lab/beta') {
    graph.scope = 'visual_encoder_predictor';
    graph.nodes.find((n) => n.id === 'tokens')!.references = [];
    const patch = graph.parameters.find((p) => p.id === 'volume')!;
    patch.name = 'encoder.patch_embed.proj.weight';
    patch.logical_shape = [1024, 3, 2, 16, 16].map((value) => ({ kind: 'constant', value }));
    patch.storage = [{ name: patch.name, dtype: 'F32', shape: [1024, 3, 2, 16, 16] }];
  }
  return response;
}
