import type { Graph, GraphNode } from '../src/architecture-explorer/graph';

/** Authored presentation fixture, not an analyzer or checkpoint claim. */
export function summaryFixture(long = false): Graph {
  const port = (id: string, direction: 'input' | 'output') => ({ id, label: id, direction,
    shape: [{ kind: 'symbol' as const, name: 'B' }, { kind: 'constant' as const, value: 3 }] });
  const node = (id: string, label: string): GraphNode => ({ id, label, kind: 'operation', parent_id: 'block',
    ports: [port('x', 'input'), port('out', 'output')], parameter_ids: [], references: [], attributes: [], provenance: [] });
  const norm = node('norm', 'layer norm'); norm.operation = 'layer_norm';
  norm.formula = 'LayerNorm(x; weight, bias, epsilon)'; norm.parameter_ids = ['norm-weight', 'norm-bias'];
  norm.references = [{ kind: 'module', name: 'layers.7.norm' }, { kind: 'parameter', parameter_id: 'norm-weight' }];
  norm.attributes = [{ name: 'epsilon', value: 0.00003, provenance: [] }];
  const linear = node('linear', 'linear'); linear.operation = 'linear'; linear.formula = 'x W^T + bias';
  linear.parameter_ids = ['linear-weight', 'linear-bias']; linear.references = [{ kind: 'module', name: 'layers.7.linear' }];
  const block: GraphNode = { ...node('block', 'Block 7'), kind: 'group', children: ['norm', 'linear'], ports: [] };
  delete block.parent_id;
  const graph: Graph = { graph_id: 'card-summaries', scope: 'language_model', coverage: 'complete',
    symbols: [{ name: 'B', meaning: 'Batch' }], nodes: [block, norm, linear],
    edges: [{ id: 'norm-to-linear', source: { node_id: 'norm', port_id: 'out' }, target: { node_id: 'linear', port_id: 'x' }, kind: 'data', provenance: [] }],
    repetitions: [], diagnostics: [],
    parameters: [norm, linear].flatMap((n) => n.parameter_ids.map((id) => ({ id, name: `layers.7.${n.id}.${id.endsWith('bias') ? 'bias' : 'weight'}`,
      logical_shape: (id === 'linear-weight' ? [3, 3] : [3]).map((value) => ({ kind: 'constant' as const, value })),
      binding: 'native' as const, storage: [], inspection: { status: 'available' as const, tensor_id: `inventory-${id}` }, provenance: [] }))),
  };
  if (long) {
    norm.formula = '<script>display only</script> ' + 'supplied long formula '.repeat(30);
    graph.parameters[0]!.name = 'layers.7.norm.q_proj.' + 'submodule.'.repeat(35) + 'weight';
    for (let i = 0; i < 7; i++) {
      const id = `extra-${i}`; norm.parameter_ids.push(id);
      graph.parameters.push({ id, name: `layers.7.norm.extra.${i}.weight`, binding: 'unresolved', logical_shape: null,
        storage: [], inspection: { status: 'unavailable', reason: 'unresolved_binding', message: 'No verified inventory tensor.' }, provenance: [] });
    }
    block.parameter_ids = ['norm-weight']; block.references = [{ kind: 'module', name: 'layers.7' }];
    block.ports = [port('x', 'input'), port('out', 'output')];
    graph.edges.push(
      { id: 'enter', source: { node_id: 'block', port_id: 'x' }, target: { node_id: 'norm', port_id: 'x' }, kind: 'data', provenance: [] },
      { id: 'leave', source: { node_id: 'linear', port_id: 'out' }, target: { node_id: 'block', port_id: 'out' }, kind: 'data', provenance: [] });
  }
  return graph;
}
