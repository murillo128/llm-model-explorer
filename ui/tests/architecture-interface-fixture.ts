import type { Graph, GraphNode } from '../src/architecture-explorer/graph';

/** Independently authored declaration/operation cases, not producer output. */
export function interfaceFixture(kind: 'dense' | 'hybrid' | 'visual' = 'dense', many = false): Graph {
  const shape = [{ kind: 'constant' as const, value: 3 }, { kind: 'constant' as const, value: 7 }];
  const port = (id: string, direction: 'input' | 'output') => ({ id, label: id, direction, shape });
  const leaf = (id: string, kind: 'input' | 'output' | 'operation', parent?: string): GraphNode => ({
    id, label: id, kind, operation: kind === 'operation' ? 'identity' : `declared_${id}`,
    ...(parent ? { parent_id: parent } : {}), ports: kind === 'input' ? [port('out', 'output')] : kind === 'output'
      ? [port('x', 'input'), ...(kind === 'output' && parent ? [port('out', 'output')] : [])] : [port('x', 'input'), port('out', 'output')],
    parameter_ids: [], references: [], attributes: [], provenance: [{ kind: 'description', source: `fixture.${id}` }],
  });
  const inputNames = kind === 'visual' ? ['video', 'context_indices', 'target_indices'] : ['Token IDs', 'positions', 'mask', 'current_mask'];
  if (many) for (let i = 0; i < 18; i++) inputNames.push(`auxiliary_${i}`);
  const root = kind === 'dense' ? 'model' : kind === 'hybrid' ? 'language' : 'encoder';
  const inputs = inputNames.map((name) => leaf(name, 'input', kind === 'dense' ? root : undefined));
  const operation = leaf(kind === 'visual' ? 'patch preparation' : 'Embedding', 'operation', root);
  operation.ports = inputNames.map((name) => port(name, 'input')).concat(port('out', 'output'));
  const group: GraphNode = { id: root, label: kind === 'hybrid' ? 'Language model' : root, kind: 'group',
    children: [...(kind === 'dense' ? inputs.map((n) => n.id) : []), operation.id],
    ports: kind === 'dense' ? [] : [...inputNames.map((name) => port(name, 'input')), port('out', 'output')],
    parameter_ids: [], references: [], attributes: [], provenance: [] };
  const head = leaf(kind === 'visual' ? 'predictor' : 'LM head', 'operation', kind === 'dense' ? root : undefined);
  const output = leaf(kind === 'visual' ? 'representations' : 'logits', 'output', kind === 'dense' ? root : undefined);
  if (kind === 'dense') group.children.push(head.id, output.id);
  const edges: Graph['edges'] = [];
  const link = (from: string, out: string, to: string, into: string) => edges.push({ id: `wire-${edges.length}`, kind: 'data',
    source: { node_id: from, port_id: out }, target: { node_id: to, port_id: into }, provenance: [{ kind: 'description', source: 'independent.fixture' }] });
  for (const node of inputs) {
    if (node.id === 'auxiliary_17') continue; // A declared but unconnected interface.
    if (kind !== 'dense') { link(node.id, 'out', root, node.id); link(root, node.id, operation.id, node.id); }
    else link(node.id, 'out', operation.id, node.id);
  }
  if (kind !== 'dense') { link(operation.id, 'out', root, 'out'); link(root, 'out', head.id, 'x'); }
  else link(operation.id, 'out', head.id, 'x');
  link(head.id, 'out', output.id, 'x');
  const tools: GraphNode[] = kind === 'visual' ? [] : [{ id: 'tokenizer', label: 'Tokenizer capability', kind: 'context',
    ports: [], parameter_ids: [], references: [{ kind: 'tokenizer' }], attributes: [], provenance: [] }];
  return { graph_id: `interface-fixture-${kind}-${many}`, scope: kind === 'visual' ? 'visual_encoder_predictor' : 'language_model',
    coverage: 'complete', symbols: [], nodes: [...inputs, group, operation, head, output, ...tools], edges, parameters: [], repetitions: [], diagnostics: [] };
}
