import type { Graph, GraphNode } from '../src/architecture-explorer/graph';

/** Small authored graphs exercise geometry, never checkpoint-size heuristics. */
export function overviewFixture(kind: 'compact' | 'synthetic' | 'wide' | 'fanout' | 'training'): Graph {
  const graph: Graph = { graph_id: `overview-${kind}`, scope: kind === 'training' ? 'model_defined' : 'visual_encoder_predictor',
    coverage: 'complete', nodes: [], edges: [], parameters: [], repetitions: [], symbols: [], diagnostics: [] };
  const node = (id: string, parent_id?: string, children?: string[]): GraphNode => ({ id, label: id,
    ...(children ? { kind: 'group', children } : { kind: 'operation', operation: 'identity' }),
    ...(parent_id ? { parent_id } : {}), ports: [{ id: 'x', label: 'x', direction: 'input', shape: null },
      { id: 'y', label: 'y', direction: 'output', shape: null }], parameter_ids: [], references: [], attributes: [],
    provenance: [{ kind: 'description', source: kind === 'training' ? 'model-supplied architecture.json' : 'authored overview fixture' }] });
  const link = (from: string, to: string, input = false, output = false) => graph.edges.push({ id: `edge-${graph.edges.length}`,
    source: { node_id: from, port_id: input ? 'x' : 'y' }, target: { node_id: to, port_id: output ? 'y' : 'x' }, kind: 'data', provenance: [] });
  if (kind === 'compact') {
    graph.nodes = [node('Model', undefined, ['Operation']), node('Operation', 'Model')];
    return graph;
  }
  if (kind === 'synthetic') {
    graph.nodes = [node('Encoder', undefined, ['Patch']), node('Patch', 'Encoder'), node('Predictor', undefined, ['Prediction']), node('Prediction', 'Predictor')];
    link('Encoder', 'Predictor'); return graph;
  }
  const names = kind === 'training' ? ['Student', 'Teacher', 'Loss'] : Array.from({ length: 24 }, (_, i) => `Branch ${i}`);
  graph.nodes.push(node('Model', undefined, names));
  for (const [index, name] of names.entries()) {
    graph.nodes.push(node(name, 'Model', [`${name} interior`]), node(`${name} interior`, name));
    if (kind === 'wide' && index > 0) link(names[index - 1]!, name);
    else link('Model', name, true);
    link(name, `${name} interior`, true); link(`${name} interior`, name, false, true);
  }
  return graph;
}
