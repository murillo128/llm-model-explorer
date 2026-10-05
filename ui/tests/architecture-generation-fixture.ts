import type { Graph, GraphNode } from '../src/architecture-explorer/graph';

/** Independent phase/edge oracle from issue #294, not output of a producer. */
export function generationFixture(): Graph {
  const dim = (name: string) => ({ kind: 'symbol' as const, name });
  const ids = [dim('B'), dim('S')], mask = [dim('B'), { kind: 'constant' as const, value: 1 }, dim('S'), dim('S')];
  const initial = [dim('B'), dim('P')], final = [dim('B'), dim('F')];
  const next = [dim('B'), { kind: 'expression' as const, text: 'S + 1', symbols: ['S'] }];
  const token = [dim('B'), { kind: 'constant' as const, value: 1 }];
  const logits = [...ids, { kind: 'constant' as const, value: 8 }];
  const port = (id: string, direction: 'input' | 'output', shape: GraphNode['ports'][number]['shape'], label = id) => ({ id, direction, shape, label });
  const inputs = [port('tokens', 'input', ids), port('positions', 'input', ids), port('mask', 'input', mask)];
  const role = (value: string) => [{ name: 'semantic_role', value, provenance: [] }];
  const common = { parameter_ids: [], references: [], attributes: [], provenance: [] };
  const nodes: GraphNode[] = [
    { ...common, id: 'generation', kind: 'group', label: 'Generation', children: ['sequence', 'prepare', 'model', 'select', 'append'],
      attributes: [...role('autoregressive_generation'), { name: 'policy', value: 'greedy_no_cache', provenance: [] }],
      ports: [port('prompt_ids', 'input', initial), port('token_ids', 'output', final, 'token_ids[T]')] },
    { ...common, id: 'sequence', parent_id: 'generation', kind: 'state', label: 'Sequence state', operation: 'generation_sequence_state',
      attributes: role('generation_sequence_state'), ports: [port('initial', 'input', initial), port('current', 'output', ids), port('next', 'input', next), port('final', 'output', final)] },
    { ...common, id: 'prepare', parent_id: 'generation', kind: 'operation', label: 'Prepare inputs', operation: 'generation_prepare_inputs',
      ports: [port('sequence', 'input', ids, 'token_ids[t]'), port('tokens', 'output', ids, 'token_ids[t]'), port('positions', 'output', ids, 'positions[t]'), port('mask', 'output', mask, 'causal_mask[t]')] },
    { ...common, id: 'model', parent_id: 'generation', kind: 'group', label: 'model', children: ['neural'], ports: [...inputs, port('logits', 'output', logits)] },
    { ...common, id: 'neural', parent_id: 'model', kind: 'operation', label: 'Fixture neural computation', operation: 'fixture_model',
      ports: [...inputs, port('logits', 'output', logits)], parameter_ids: ['weight'] },
    { ...common, id: 'select', parent_id: 'generation', kind: 'operation', label: 'Next token', operation: 'generation_greedy_next_token',
      ports: [port('logits', 'input', logits, 'logits[t]'), port('token', 'output', token, 'next_token_id[t]')] },
    { ...common, id: 'append', parent_id: 'generation', kind: 'operation', label: 'Append token', operation: 'generation_append_token',
      ports: [port('sequence', 'input', ids, 'token_ids[t]'), port('token', 'input', token, 'next_token_id[t]'), port('updated', 'output', next, 'token_ids[t+1]')] },
  ];
  const edges: Graph['edges'] = [
    ['initial', 'generation', 'prompt_ids', 'sequence', 'initial', 'data'],
    ['current', 'sequence', 'current', 'prepare', 'sequence', 'state'],
    ['tokens', 'prepare', 'tokens', 'model', 'tokens', 'data'],
    ['positions', 'prepare', 'positions', 'model', 'positions', 'data'],
    ['mask', 'prepare', 'mask', 'model', 'mask', 'data'],
    ['logits', 'model', 'logits', 'select', 'logits', 'data'],
    ['bypass', 'prepare', 'tokens', 'append', 'sequence', 'data'],
    ['selected', 'select', 'token', 'append', 'token', 'data'],
    ['next', 'append', 'updated', 'sequence', 'next', 'state'],
    ['final', 'sequence', 'final', 'generation', 'token_ids', 'state'],
    ...['tokens', 'positions', 'mask'].map((p) => [`model-${p}`, 'model', p, 'neural', p, 'data']),
    ['model-logits', 'neural', 'logits', 'model', 'logits', 'data'],
  ].map(([id, sn, sp, tn, tp, kind]) => ({ id: id!, source: { node_id: sn!, port_id: sp! }, target: { node_id: tn!, port_id: tp! }, kind: kind as 'data' | 'state', provenance: [] }));
  return { graph_id: 'authored-generation-fixture', scope: 'language_model', coverage: 'complete', nodes, edges, repetitions: [],
    symbols: ['B', 'S', 'P', 'F'].map((name) => ({ name, meaning: name })), diagnostics: [],
    parameters: [{ id: 'weight', name: 'model.weight', binding: 'native', logical_shape: [{ kind: 'constant', value: 8 }, { kind: 'constant', value: 8 }],
      storage: [{ name: 'model.weight', dtype: 'F32', shape: [8, 8] }], inspection: { status: 'available', tensor_id: 'exact-weight' }, provenance: [] }] };
}
