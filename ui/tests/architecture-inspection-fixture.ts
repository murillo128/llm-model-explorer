import { contractResponse } from './architecture-fixtures';
import type { TensorDescriptor } from '../src/app/session-controller';
import type { components } from '../src/api/generated/types';

/** Synthetic bindings deliberately use different asymmetric weights per instance. */
export function inspectionFixture(tensors: TensorDescriptor[], modelId: string, lora = false) {
  type S = components['schemas'];
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
  if (lora) {
    const layer = graph.nodes.find((node) => node.id === 'layer1');
    const base = graph.nodes.find((node) => node.id === 'linear1');
    const aTensor = tensors.find((tensor) => tensor.id === 'lora-A')!;
    const bTensor = tensors.find((tensor) => tensor.id === 'lora-B')!;
    if (layer?.kind !== 'group' || !base || base.kind !== 'operation') throw new Error('Invalid LoRA fixture.');
    const aName = '__peft__.org%2Fsmoltalk.model.layers.1.self_attn.q_proj.lora_A.weight';
    const bName = '__peft__.org%2Fsmoltalk.model.layers.1.self_attn.q_proj.lora_B.weight';
    aTensor.name = aName; aTensor.path = aName.split('.');
    bTensor.name = bName; bTensor.path = bName.split('.');
    const nativeFactor = (id: string, name: string, tensor: TensorDescriptor): S['ArchitectureDirectParameter'] => ({
      id, name, binding: 'native',
      logical_shape: tensor.shape.map((value) => ({ kind: 'constant', value })),
      storage: [{ name: tensor.name, dtype: tensor.storage_dtype, shape: tensor.shape, role: 'adapter_factor' }],
      inspection: { status: 'available', tensor_id: tensor.id }, provenance: [],
    });
    graph.parameters.push(nativeFactor('lora-a', aName, aTensor), nativeFactor('lora-b', bName, bTensor));
    const flowShape = layer.ports[0]!.shape;
    const port = (id: string, direction: 'input' | 'output', shape: S['ArchitectureShape']): S['ArchitecturePort'] => ({ id, direction, label: id, shape });
    const provenance: S['ArchitectureProvenance'][] = [{ kind: 'description', source: 'synthetic-smollm2-lora' }];
    const operation = (id: string, label: string, name: string, ports: S['ArchitecturePort'][], parameterIds: string[] = [], formula?: string): S['ArchitectureLeafNode'] => ({
      id, kind: 'operation', label, operation: name, parent_id: 'layer1', ports,
      parameter_ids: parameterIds,
      references: [
        ...(parameterIds.map((parameter_id) => ({ kind: 'parameter' as const, parameter_id }))),
        { kind: 'module', name: `model.layers.1.self_attn.q_proj.${id}` },
      ],
      attributes: [], provenance,
      ...(formula ? { formula } : {}),
    });
    const a = operation('lora_A', 'LoRA A projection', 'linear', [port('x', 'input', flowShape), port('out', 'output', flowShape)], ['lora-a'], 'y = x Aᵀ');
    const b = operation('lora_B', 'LoRA B projection', 'linear', [port('x', 'input', flowShape), port('out', 'output', flowShape)], ['lora-b'], 'y = x Bᵀ');
    const scale = operation('lora_scale', 'LoRA scale', 'scale', [port('x', 'input', flowShape), port('out', 'output', flowShape)], [], 'y = (alpha / r) * x');
    scale.attributes = [{ name: 'factor', value: 2, provenance }];
    const add = operation('lora_add', 'LoRA residual', 'add', [port('base', 'input', flowShape), port('adapter', 'input', flowShape), port('out', 'output', flowShape)], [], 'y = base + adapter');
    graph.nodes.push(a, b, scale, add);
    layer.children.push('lora_A', 'lora_B', 'lora_scale', 'lora_add');
    graph.edges = graph.edges.filter((edge) => !(edge.source.node_id === 'linear1' && edge.target.node_id === 'layer1'));
    const edge = (id: string, source: string, sourcePort: string, target: string, targetPort: string): S['ArchitectureEdge'] => ({
      id, source: { node_id: source, port_id: sourcePort }, target: { node_id: target, port_id: targetPort }, kind: 'data', provenance,
    });
    graph.edges.push(
      edge('lora-input-base', 'layer1', 'in', 'linear1', 'in'),
      edge('lora-input-a', 'layer1', 'in', 'lora_A', 'x'),
      edge('lora-a-b', 'lora_A', 'out', 'lora_B', 'x'),
      edge('lora-b-scale', 'lora_B', 'out', 'lora_scale', 'x'),
      edge('lora-base-add', 'linear1', 'out', 'lora_add', 'base'),
      edge('lora-scale-add', 'lora_scale', 'out', 'lora_add', 'adapter'),
      edge('lora-output', 'lora_add', 'out', 'layer1', 'out'),
    );
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
