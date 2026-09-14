import fixture from '../../api/fixtures/architecture.json' with { type: 'json' };
import type { components } from '../src/api/generated/types';
import type { Graph, GraphNode } from '../src/architecture-explorer/graph';

type S = components['schemas'];
export const contractResponse = fixture.response as S['ArchitectureAvailableResponse'];
export const contractInventory = fixture.context.inventory as S['TensorInventory'];
// Pinned configuration observations, not checkpoint/analyzer acceptance. Interiors are
// deliberately synthetic port/branch stress fixtures; production never imports this file.
export const references = [
  { name: 'qwen3', model: 'JunHowie/Qwen3-0.6B-GPTQ-Int4', revision: 'b9d87006067b0c0c2dea836370d6288e14f112ab', stacks: [28], hybrid: false, visual: false },
  { name: 'qwen35', model: 'AxionML/Qwen3.5-0.8B-NVFP4', revision: '2ac1e750cda67cc8538d731f6216f77b9c3a6f72', stacks: [24], hybrid: true, visual: false },
  { name: 'vjepa2', model: 'facebook/vjepa2-vitl-fpc64-256', revision: 'b3c1679b7c34d3255ef3547f27c7b226aefab26f', stacks: [24, 12], hybrid: false, visual: true },
  { name: 'smollm2', model: 'HuggingFaceTB/SmolLM2-135M', revision: '93efa2f097d58c2a74874c7e644dbc9b0cee75a2', stacks: [30], hybrid: false, visual: false },
] as const;
export function referenceFixture(name: string): S['ArchitectureAvailableResponse'] {
  const reference = references.find((r) => r.name === name)!;
  const graph: Graph = { graph_id: `fixture-${name}`, scope: reference.visual ? 'visual_encoder_predictor' : 'language_model', coverage: 'partial',
    symbols: [{ name: 'B', meaning: 'Batch' }, { name: 'S', meaning: reference.visual ? 'Visual sequence' : 'Token sequence' }],
    nodes: [], edges: [], parameters: [], repetitions: [], diagnostics: [{ code: 'ui_stress_fixture', message: 'Synthetic branch topology with reference configuration counts; not checkpoint support evidence.' }] };
  const port = (id: string, direction: 'input' | 'output'): S['ArchitecturePort'] => ({ id, direction, label: id,
    shape: [{ kind: 'symbol', name: 'B' }, { kind: 'symbol', name: 'S' }, { kind: 'unknown', reason: 'UI fixture' }] });
  function node(id: string, label: string, parent?: S['ArchitectureGroupNode'], group = false): GraphNode {
    const common = { id, label, ports: [port('in', 'input'), port('out', 'output')], parameter_ids: [], references: [], attributes: [], provenance: [], ...(parent ? { parent_id: parent.id } : {}) };
    const n: GraphNode = group ? { ...common, kind: 'group', children: [] } : { ...common, kind: 'operation', operation: label };
    graph.nodes.push(n); parent?.children.push(id); return n;
  }
  const group = (id: string, label: string, parent?: S['ArchitectureGroupNode']) => node(id, label, parent, true) as S['ArchitectureGroupNode'];
  function edge(source: GraphNode, target: GraphNode, from = 'out', to = 'in', kind: 'data' | 'state' | 'context' = 'data') {
    graph.edges.push({ id: `edge-${graph.edges.length}`, source: { node_id: source.id, port_id: from }, target: { node_id: target.id, port_id: to }, kind, provenance: [] });
  }
  const input = node('input', reference.visual ? 'Symbolic visual input' : 'Tokenization context');
  input.kind = 'context'; if (!reference.visual) input.references.push({ kind: 'tokenizer' });
  const root = group('root', 'Model'); edge(input, root, 'out', 'in', 'context');
  let previous: GraphNode | undefined;
  for (const [stackIndex, count] of reference.stacks.entries()) {
    const stack = group(`stack-${stackIndex}`, reference.visual ? stackIndex === 0 ? 'Encoder' : 'Predictor' : 'Decoder', root);
    if (previous) edge(previous, stack); else edge(root, stack, 'in');
    previous = stack;
    const repetition: S['ArchitectureRepetition'] = { id: `repeat-${stackIndex}`, parent_id: stack.id, label: `${stack.label} layers`, instances: [] };
    graph.repetitions.push(repetition);
    let last: GraphNode | undefined;
    for (let i = 0; i < count; i++) {
      const variant = reference.hybrid && i % 4 !== 3 ? 'linear_attention' : 'full_attention';
      const layer = group(`layer-${stackIndex}-${i}`, `${stack.label} layer ${i}`, stack);
      repetition.instances.push({ node_id: layer.id, index: i, variant });
      if (last) edge(last, layer); else edge(stack, layer, 'in'); last = layer;
      // 32/40 operations plus skip/branch/state crossings exercise full-size rendering.
      const operationCount = variant === 'linear_attention' ? 40 : 32;
      let tail: GraphNode = layer;
      const ops: GraphNode[] = [];
      for (let j = 0; j < operationCount; j++) {
        const op = node(`${layer.id}-op-${j}`, `${variant} operation ${j}`, layer);
        edge(tail, op, j === 0 ? 'in' : 'out'); tail = op; ops.push(op);
      }
      edge(tail, layer, 'out', 'out');
      edge(ops[0]!, ops[14]!); edge(ops[15]!, ops[31]!); // two long residual-like crossings
      edge(ops[3]!, ops[8]!); edge(ops[4]!, ops[8]!); // converging branches
      if (variant === 'linear_attention') edge(ops[25]!, ops[5]!, 'out', 'in', 'state');
    }
    edge(last!, stack, 'out', 'out');
  }
  edge(previous!, root, 'out', 'out');
  const output = node('output', reference.visual ? 'Representations' : 'Language output'); output.kind = 'output'; edge(root, output);
  return { model_id: reference.model, status: 'available', graph, diagnostics: [] };
}
